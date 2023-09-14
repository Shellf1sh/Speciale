import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

class Qubit:#The base qubit class with the things that all qubits require
    def __init__(self, N):
        self.N = N
        self.eigvals = np.array([])   
        self.eigvecs = np.array([])

    def Hamkin():
        pass

    def HamPot():
        pass

    def Hamiltonian(self):
        return self.Hamkin() + self.HamPot
    
    def solve(self):
        eigenvalues, eigenvectors =  np.linalg.eigh(self.ham())
        self.eigvecs = eigenvalues
        self.eigvecs = eigenvectors

    def plot(self, wavefuncs): #TO-DO
        pass

    
class transmon_charge(Qubit):
    def __init__(self, N, EC, EJ, ng):
        super().__init__(N)
        self.EC = EC
        self.EJ = EJ
        self.ng = ng

        if(self.ng%2 == 0):#Making sure that the resolution is uneven
            self.ng += 1

        self.n_cut = (self.ng-1)/2#Setting the charge cut-off


    def Hamkin(self):
        n_array = np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = 4*self.EC*np.diag(n_array) #In charge basis the (n-ng)**2 term is a diagonal matrix
        return hkin

    def HamPot(self):
        off_diag = np.ones(self.N-1)     
        hpot = self.EJ*(np.diag(off_diag, 1)+np.diag(off_diag, -1))/2
        return hpot
    
class gatemon_charge_averin(Qubit):
    def __init__(self, N, EC, gap, T, ng):
        super().__init__(N)
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T

    def Hamkin(self):
        n_array = np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = 4*self.EC*np.diag(n_array) #In charge basis the (n-ng)**2 term is a diagonal matrix
        hkin_2channel = np.kron(hkin, np.identity(2))
        return hkin_2channel
    
    def HamPot(self):
        sy=np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz=np.array([[1,0],[0,-1]])
        r = np.sqrt(1-self.T)#Why? Ask Karsten

        #If I can do change of variable I might be able to use these sine and cosine functions
        off_diag = np.ones(self.N-1)  
        cos = (np.diag(off_diag, 1) + np.diag(off_diag, -1))/2
        sin = (np.diag(off_diag, 1) - np.diag(off_diag, -1))/(2j)

        #These are the functions for if I need to use the sin-half and cosine-half functions
        coords = np.arange(0, self.N)
        x, y = np.meshgrid(coords, coords)

        cos_half = -2*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2 - 1))
        sin_half = 4j*(x-y)*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2-1))

        return self.gap*(np.kron(cos, sz) + r*np.kron(sin, sy))
