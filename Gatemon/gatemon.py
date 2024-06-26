import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import os

def mp(x):
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in x]))

#parameters
class gatemon_flux:
    def __init__(self, n, EC, t, ng):
        self.EC = EC
        self.n = n
        if(self.n%2 == 0): #This condition is to ensure that we have an uneven number of steps so that the Fourier transform works
            self.n += 1
        self.t = t
        self.ng = ng
       
        flux = 0
        self.eigvals = []   
        self.eigvecs = []

        self.model='beenakker'
   
    def w(self):
       return(np.sqrt(self.EC*2))
    
    def hamkin(self):
        vec = np.ones(self.n - 1)
        iden = np.identity(self.n)
        dx = 2.*np.pi/self.n

        first_deriv = 1j*np.diag(vec,-1)-1j*np.diag(vec,1) #Create the first derivative
        first_deriv[0,self.n-1] = -1j#Add periodic boundary conditions
        first_deriv[self.n-1,0] = 1j

        second_deriv = (2*iden-np.diag(vec,-1)-np.diag(vec,1)) #Create the second derivative matrix
        second_deriv[0,self.n-1] = -1 #Add periodic boundary conditions
        second_deriv[self.n-1,0] = -1

        hkin = 4.0*self.EC*(1/(dx**2)*second_deriv - 2*self.ng/(2*dx)*first_deriv + self.ng**2*iden) #Combine to create the kinetic energy term

        if self.model=='averin':
            hkin = np.kron(hkin,np.identity(2))    
        return(hkin)
      
      
    def hampot(self):
        dx = 2*np.pi/self.n
        x = np.array([-np.pi+dx*i for i in np.arange(self.n)])
        if self.model=='beenakker':
            v = -np.array([np.sqrt(1-self.t*(np.sin(xi/2.)**2)) for xi in x])          
            v = np.diag(v)
        if self.model=='averin':
            #sx=np.array([[0,1],[1,0]])
            sy=np.array([[0,-1j],[1j,0]]) #Define the Pauli matrices
            sz=np.array([[1,0],[0,-1]])
            r = np.sqrt(1-self.t)          
            cos = np.cos(x/2)
            sin = np.sin(x/2)
            v = np.kron(np.diag(cos),sz)+r*np.kron(np.diag(sin),sy)
        return(v)
      
    def ham(self):
        return(self.hamkin()+self.hampot())
    
    def spect(self):
        v =  np.linalg.eigvals(self.ham()).real
        v.sort()
        self.eigvals = v      
      
    def eigsys(self): #A function that calculates the eigenvalues and the eigenvectors for the hamiltonian
        v, w =  np.linalg.eigh(self.ham())
        v = v.real
        #order = np.argsort(v)
        self.eigvals = v #np.array([v[i] for i in order])
        self.eigvecs = w # np.array([w[:,i] for i in order])   
        #print(np.shape(self.eigvecs), np.shape(self.eigvecs[0]), np.shape(self.eigvecs[1]), np.shape(self.eigvecs[2]))  

    def getEigvals(self): #A function that returns the eigenvalues
        return self.eigvals

      
    def plotspect(self, xmin=-0.75*np.pi, xmax=0.75*np.pi, npoints=100, neigs=10,pr=False):#A function that plots the potential and the lowest eigenvalues
        dx = (xmax-xmin)/npoints
        x = np.array([xmin+i*dx for i in range(npoints+1)])
        v = -np.sqrt(1-self.t*(np.sin(x/2.)**2))      
        if len(self.eigvals)==0:
            self.spect()
        for i in range(neigs):
            plt.axhline(self.eigvals[i],xmin,xmax,color='red')                   
        plt.plot(x,v)
        if pr:
            plt.plot(x,-v)
        plt.title(self.model+', $\mathcal{t}$=' + str(self.t) + ', $E_C$ = ' + str(self.EC))
        plt.xlabel(r'$\varphi$')
        plt.ylabel(r'Energy')
        if pr:
            savefigure('spect.pdf')      
        plt.show()
       
    def plotwf(self,n,pr=False):#A function that plots the first couple of wavefunctions   
        dx = 2*np.pi/self.n
        x = np.array([-np.pi+dx*i for i in range(self.n)])
        if len(self.eigvecs)==0:#If the eigenvalues haven't been calculated yet then do it here
            self.eigsys()
        for i in range(n):
            if self.model=='beenakker':
                vi=self.eigvecs[i]
            elif self.model=='averin':
                vi = np.array([self.eigvecs[i][2*j+1].real for j in range(self.n)])
            plt.plot(x,vi)
        if pr:
            savefigure('wavefunctions.pdf')      
        plt.show()
      
    def dH01(self):
        dx = 2*np.pi/self.n
        x = np.array([-np.pi+dx*i for i in range(self.n)])
        self.eigsys()
        if self.model=='beenakker':
            hm = -self.t*(np.sin(x/2)**2)/np.sqrt(1-self.t*np.sin(x/2)**2)
            return np.sum(hm*self.eigvecs[0]*self.eigvecs[1])*dx
        if self.model=='averin':
            sy=np.array([[0,-1j],[1j,0]])
            r=np.sqrt(1-self.t)
            sin = np.sin(x/2)
            hm = r*np.kron(np.diag(sin),sy)
            return abs(np.matmul(self.eigvecs[0].T,np.matmul(hm,self.eigvecs[1])))
        

class gatemon_charge_averin(gatemon_flux):
    def __init__(self, n, EC, t, ng):
        super().__init__(n, EC, t, ng)#Here we call the initialization of the gatemon_flux to get all the variables initialized
        self.n_cut = int((self.n-1)/2) #Define the charge cut-off depending on the dimension of the calculation
        print("The charge cut-off is " + str(self.n_cut))

    def hamkin(self):
        n_array = np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = np.diag(n_array) #In charge basis the (n-ng)**2 term is a diagonal matrix
        hkin = np.kron(hkin,np.identity(2))    
        return hkin

    
    def hampot(self):
        sy=np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz=np.array([[1,0],[0,-1]])
        r = np.sqrt(1-self.t)     
        off_diag = np.ones(self.n-1)     
        cos = (np.diag(off_diag, 1)+np.diag(off_diag, -1))/2
        sin = (np.diag(off_diag, 1)-np.diag(off_diag, -1))/(2j)
        #print(np.shape(cos), np.shape(sz))
        v = np.kron(cos,sz)+r*np.kron(sin,sy)
        return v

    def plotwf(self,n,pr=False):#A function that plots the first couple of wavefunctions   
        dx = 2*np.pi/self.n
        x = np.arange(-self.n_cut, self.n_cut+1, 1)
        if len(self.eigvecs) == 0:#If the eigenvalues haven't been calculated yet then do it here
            self.eigsys()
        for i in range(n):
            vi = np.array([self.eigvecs[i][2*j+1].real for j in range(self.n)])
            plt.plot(x,vi, '-o')
        if pr:
            savefigure('wavefunctions.pdf')      
        plt.show()

class transmon(gatemon_charge_averin):
    def __init__(self, n, EC, t, ng):
        super().__init__(n, EC, t, ng)#The T variable should not be used now so we just parse 1

    def hamkin(self):
        n_array = np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = np.diag(n_array) #In charge basis the (n-ng)**2 term is a diagonal matrix
        return hkin
    
    def hampot(self):    
        off_diag = np.ones(self.n-1)     
        cos = (np.diag(off_diag, 1)+np.diag(off_diag, -1))/2
        return cos

    def ham(self):
        return 4*self.EC *self.hamkin()+self.hampot()
    

def savefigure(filename):
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(filename)     
    
