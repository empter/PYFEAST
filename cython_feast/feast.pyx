import numpy as np
from scipy.sparse import csr_matrix

cdef extern from "../FEAST/include/feast_tools.h":
    void feastinit(int *)

cdef extern from "../FEAST/include/feast_sparse.h":
    void zfeast_hcsrev(char *,int *,double *,int *,int *,int *,double *,int *,double *,double *,int *,double *,double *,int *,double *,int *)
    void zfeast_gcsrev(int *,double *,int *,int *,int *,double *,int *,double *,double *,int *,double *,double *,int *,double *,int *)
    void dfeast_scsrev(char *,int *,double *,int *,int *,int *,double *,int *,double *,double *,int *,double *,double *,int *,double *,int *)
    void dfeast_gcsrev(int *,double *,int *,int *,int *,double *,int *,double *,double *,int *,double *,double *,int *,double *,int *)

def feastinit_py(fpm):
    # if not fpm.flags['C_CONTIGUOUS']: # contiguous check not needed
    #     fpm = np.ascontiguousarray(fpm)
    cdef int[:] fpm_view = fpm
    feastinit(&fpm_view[0])
    return fpm

class HSolver():   # solver for Hermitian matrix (real/complex auto detection)
    def __init__(self,H,M0=1,Em=0.0,Ex=0.1,which=-1):
        self.H=H
        self.N=self.H.shape[0]
        self.M0=M0
        self.Em=Em
        self.Ex=Ex
        self.which=which
        if self.which==-1:
            self.M0=self.M0*2
        if self.which==1:
            self.M0=self.M0*2
        self.fpm=feastinit_py(np.array([0 for i in range(64)], dtype=np.int32))
        self.loop=0
        self.info=255
        self.M=0
        self.epsout=1.0

        self.E=np.zeros(self.M0,dtype=np.float64) # eig_vals
        self.res=np.zeros(self.M0,dtype=np.float64) # residuals
        self.comp=1
        if self.H.dtype==np.complex128:
            self.comp=2
            self.sa=np.frombuffer(self.H.data,dtype=np.float64).data
        else:
            self.sa=self.H.data
        self.X = np.zeros(self.N*self.M0*self.comp,dtype=np.float64) # eig_vectors
        self.computed=0
    
    def fpm(self):
        return self.fpm
    
    def setfpm(self,cp=None,eps=None,ml=None,it=None,lsp=None,ifeast=None,ifp=None,ifl=None,Em=None,Ex=None):
        if cp!=None: # contour points (1 to 20, 24, 32, 40, 48, 56)
            self.fpm[1]=cp
        if eps!=None: # Stopping convergence criteria 10^(-eps)
            self.fpm[2]=eps
        if ml!=None: # Maximum number of FEAST refinement loop >=0
            self.fpm[3]=ml
        if it!=None: # Integration type (0: Gauss 1: Trapezoidal; 2: Zolotarev)
            self.fpm[15]=it
        if ifeast!=None: # Automatic switch 0:feast, 1:ifeast
            self.fpm[42]=ifeast
        if ifp!=None: # Accuracy of ifeast 10^(-ifp)
            self.fpm[44]=ifp
        if ifl!=None: # iterations of BiCGStab in ifeast
            self.fpm[45]=ifl
        if lsp!=None: # 1: single precision 0: double precision for linear solver
            self.fpm[41]=lsp
        if Em!=None: # reset E_min if needed
            self.Em=Em
        if Ex!=None: # reset E_max if needed
            self.Ex=Ex
    
    def para_check(self):
        return [self.comp,self.M0,self.Em,self.Ex,self.which,self.loop,self.epsout]
    
    def eig(self):
        if self.computed==0:
            return None
        else:
            return self.E[:self.M]
    
    def vec(self):
        if self.computed==0:
            return None
        else:
            if self.comp==2:
                Xc=np.frombuffer(self.X.data,dtype=np.complex128,count=self.N*self.M).reshape([self.M,self.N]).T
                return Xc
            else:
                Xc=np.frombuffer(self.X.data,dtype=np.float64,count=self.N*self.M).reshape([self.M,self.N]).T
                return Xc
    
    def resid(self):
        if self.computed==0:
            return None
        else:
            return self.res[:self.M]

    def eigsh(self,debug=0,se=0):
        if self.computed==1:
            return [self.eig(),self.vec(),self.M,self.info]
        dr,dc=self.H.shape
        if dr!=dc:
            return None
        cdef int[:] fpm = self.fpm
        self.fpm[0]=debug # 1 print info on screen, 0 none, -x log file
        self.fpm[13]=se # 2: Stochastic estimate eigenvalues inside search contour
        self.fpm[39]=self.which
        cdef int loop = self.loop
        cdef int info = self.info
        cdef int M = self.M
        cdef char UPLO = 'F'
        cdef double epsout = self.epsout
        cdef double[:] E = self.E
        cdef double[:] res = self.res
        cdef int N = self.N
        cdef int M0 = self.M0
        isapy=self.H.indptr+1 # feast is 1-indexed
        jsapy=self.H.indices+1
        cdef int[:] isa = isapy
        cdef int[:] jsa = jsapy
        cdef double[:] sa = self.sa
        cdef double Emin = self.Em
        cdef double Emax = self.Ex
        cdef double[:] X = self.X
        if self.comp==1:
            dfeast_scsrev(&UPLO,&N,&sa[0],&isa[0],&jsa[0],&fpm[0],&epsout,&loop,&Emin,&Emax,&M0,&E[0],&X[0],&M,&res[0],&info)
        else:
            zfeast_hcsrev(&UPLO,&N,&sa[0],&isa[0],&jsa[0],&fpm[0],&epsout,&loop,&Emin,&Emax,&M0,&E[0],&X[0],&M,&res[0],&info)
        if se==2:
            return [M,info]
        self.computed=1
        self.M=M
        self.M0=M0
        self.loop=loop
        self.info=info
        self.epsout=epsout
        self.Em=Emin
        self.Ex=Emax
        return [self.eig(),self.vec(),self.M,self.info]

class GSolver():  # solver for general matrix (real/complex auto detection)
    def __init__(self,H,M0=40,Emid=0.0,r=0.1,which=1):
        self.H=H
        self.N=self.H.shape[0]
        self.M0=M0
        self.M00=M0
        self.Emid=Emid+0.0*1j
        self.r=r
        self.which=which # 1: right eigenvetors; 0: both left/right
        self.fpm=feastinit_py(np.array([0 for i in range(64)], dtype=np.int32))
        self.fpm[14]=which
        self.loop=0
        self.info=255
        self.M=0
        self.epsout=1.0

        self.E=np.zeros(self.M0*2,dtype=np.float64) # eig_vals: complex=2*real
        self.res=np.zeros(self.M0*(2-self.which),dtype=np.float64) # residuals
        self.comp=1
        if self.H.dtype==np.complex128:
            self.comp=2
            self.sa=np.frombuffer(self.H.data,dtype=np.float64).data
        else:
            self.sa=self.H.data
        self.X = np.zeros(self.N*self.M0*2*(2-self.which),dtype=np.float64) # eig_vectors
        self.computed=0
        self.normed=0
    
    def fpm(self):
        return self.fpm
    
    def setfpm(self,cp=None,eps=None,ml=None,it=None,vh=None,ra=None,lsp=None,ifeast=None,ifp=None,ifl=None,Emid=None,r=None):
        if cp!=None: # contour points (2 to 40, 48, 64, 80, 96, 112)
            self.fpm[7]=cp
        if eps!=None: # Stopping convergence criteria 10^(-eps)
            self.fpm[2]=eps
        if ml!=None: # Maximum number of FEAST refinement loop >=0
            self.fpm[3]=ml
        if it!=None: # Integration type (0: Gauss, 1: Trapezoidal)
            self.fpm[15]=it
        if vh!=None: # vh/100='vertical axis'/'horizontal axis'
            self.fpm[17]=vh
        if ra!=None: # Ellipse rotation angle [-180:180]
            self.fpm[18]=ra
        if ifeast!=None: # Automatic switch 0:feast, 1:ifeast
            self.fpm[42]=ifeast
        if ifp!=None: # Accuracy of ifeast 10^(-ifp)
            self.fpm[44]=ifp
        if ifl!=None: # iterations of BiCGStab in ifeast
            self.fpm[45]=ifl
        if lsp!=None: # 1: single precision 0: double precision for linear solver
            self.fpm[41]=lsp
        if Emid!=None: # reset E_mid if needed
            self.Emid=Emid
        if r!=None: # reset r if needed
            self.r=r
    
    def para_check(self):
        return [self.comp,self.M0,self.Emid,self.r,self.which,self.loop,self.epsout]
    
    def eig(self):
        if self.computed==0:
            return None
        else:
            Ec=np.frombuffer(self.E.data,dtype=np.complex128)
            return Ec[:self.M]

    def eigl(self):
        if self.computed==0:
            return None
        elif self.which!=0:
            return None
        else:
            Ec=np.frombuffer(self.E.data,dtype=np.complex128)
            return Ec[:self.M].conj()

    def __normlize(self):
        Xc=np.frombuffer(self.X.data,dtype=np.complex128)
        l=self.N
        if self.which==0: # bi-orthonormal is done by FEAST
            offset=l*self.M0
            # for i in range(self.M):
            #     norm=np.sqrt(Xc[i*l+offset:(i+1)*l+offset].conj().dot(Xc[i*l:(i+1)*l]))
            #     Xc[i*l:(i+1)*l]=Xc[i*l:(i+1)*l]/norm
            #     Xc[i*l+offset:(i+1)*l+offset]=Xc[i*l+offset:(i+1)*l+offset]/norm
        else:
            for i in range(self.M):
                norm=np.sqrt(Xc[i*l:(i+1)*l].conj().dot(Xc[i*l:(i+1)*l]))
                Xc[i*l:(i+1)*l]=Xc[i*l:(i+1)*l]/norm
        self.normed=1

    def vec(self):
        if self.computed==0:
            return None
        else:
            if self.normed!=1:
                self.__normlize()
            Xc=np.frombuffer(self.X.data,dtype=np.complex128,count=self.N*self.M).reshape([self.M,self.N]).T
            return Xc

    def vecl(self):
        if self.computed==0:
            return None
        elif self.which!=0:
            return None
        else:
            if self.normed!=1:
                self.__normlize()
            Xc=np.frombuffer(self.X.data,dtype=np.complex128).reshape([self.M00*2,self.N]).T
            if self.fpm[15]==0: # bug? Gauss left_vec start at new M0; Trapezoidal left_vec start at origin M0
                offset=self.M0
            else:
                offset=self.M00
            return Xc[:,offset:offset+self.M]
    
    def resid(self):
        if self.computed==0:
            return None
        else:
            return self.res[:self.M]

    def residl(self):
        if self.computed==0:
            return None
        elif self.which!=0:
            return None
        else:
            return self.res[self.M0:self.M+self.M0]

    def eigs(self,debug=0,se=0):
        if self.computed==1:
            return [self.eig(),self.vec(),self.M,self.info]
        dr,dc=self.H.shape
        if dr!=dc:
            return None
        cdef int[:] fpm = self.fpm
        self.fpm[0]=debug # 1 print info on screen, 0 none, -x log file
        self.fpm[13]=se # 2: Stochastic estimate eigenvalues inside search contour
        cdef int loop = self.loop
        cdef int info = self.info
        cdef int M = self.M
        cdef double epsout = self.epsout
        cdef double[:] E = self.E
        cdef double[:] res = self.res
        cdef int N = self.N
        cdef int M0 = self.M0
        isapy=self.H.indptr+1 # feast is 1-indexed
        jsapy=self.H.indices+1
        cdef int[:] isa = isapy
        cdef int[:] jsa = jsapy
        cdef double[:] sa = self.sa
        Emid_array=np.array([self.Emid.real,self.Emid.imag])
        cdef double[:] Emid = np.frombuffer(Emid_array.data,dtype=np.float64)
        cdef double r = self.r
        cdef double[:] X = self.X
        if self.comp==1:
            dfeast_gcsrev(&N,&sa[0],&isa[0],&jsa[0],&fpm[0],&epsout,&loop,&Emid[0],&r,&M0,&E[0],&X[0],&M,&res[0],&info)
        else:
            zfeast_gcsrev(&N,&sa[0],&isa[0],&jsa[0],&fpm[0],&epsout,&loop,&Emid[0],&r,&M0,&E[0],&X[0],&M,&res[0],&info)
        if se==2:
            return [M,info]
        self.computed=1
        self.M=M
        self.M0=M0
        self.loop=loop
        self.info=info
        self.epsout=epsout
        return [self.eig(),self.vec(),self.M,self.info]