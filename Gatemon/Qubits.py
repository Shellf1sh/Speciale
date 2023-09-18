import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

#================================General Qubits==============================================
class Qubit:#The base qubit class with the things that all qubits require
    def __init__(self, N):
        self.N = int(N)
        self.verbose = False
        self.eigvals = np.array([])   
        self.eigvecs = np.array([])


    def Hamkin(self):
        return np.array([])

    def Hampot(self):
        return np.array([])


    def Hamiltonian(self):
        return self.Hamkin() + self.Hampot()
    
    def solve(self):
        eigenvalues, eigenvectors =  np.linalg.eigh(self.Hamiltonian())
        self.eigvals = eigenvalues
        self.eigvecs = eigenvectors

    def plot_wav(self, x, wavefuncs):
        if(len(self.eigvals) == 0):#If the Hamiltonian hasn't been solved yet then solve it
            self.solve()
        
        for i in range(wavefuncs):
            plt.plot(x, self.eigvecs[i] * self.eigvecs[i] + self.eigvals[i], '-o')

        plt.grid(True)

#================================Transmon in charge basis===========================================
class transmon_charge(Qubit):
    def __init__(self, N, EC, EJ, ng):
        super().__init__(N)
        self.EC = EC
        self.EJ = EJ
        self.ng = ng

        if(self.N%2 == 0):#Making sure that the resolution is uneven
            self.N += 1
            if(self.verbose):
                print("The resolution is now " + str(self.N))

        self.n_cut = int((self.N-1)/2)#Setting the charge cut-off
        if(self.verbose):
            print("The charge cut-off is " + str(self.n_cut))
        self.n_array = np.array(range(-self.n_cut, self.n_cut+1))

    def Hamkin(self):
        self.n_squared =  np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = 4*self.EC*np.diag(self.n_squared) #In charge basis the (n-ng)**2 term is a diagonal matrix
        return hkin

    def Hampot(self):
        off_diag = np.ones(self.N-1)     
        hpot = -self.EJ/2*(np.diag(off_diag, 1)+np.diag(off_diag, -1))
        return hpot
    
    def plot_wav(self, number_of_wavefuncs):
        print(self.eigvals[:number_of_wavefuncs])
        #plt.plot(self.n_array, self.n_squared, '-o', color="black")
        return super().plot_wav(self.n_array, number_of_wavefuncs)


#================================Transmon in flux basis===========================================
class transmon_flux(Qubit):
    def __init__(self, N, EC, EJ, ng):
        super().__init__(N)
        self.EC = EC
        self.EJ = EJ
        self.ng = ng
        self.off_diag = np.ones(self.N-1)  

    def Hamkin(self):
        iden = np.identity(self.N)
        dx = 2.*np.pi/self.N

        first_deriv = 1j*np.diag(self.off_diag,-1)-1j*np.diag(self.off_diag,1) #Create the first derivative
        first_deriv[0,self.N-1] = -1j#Add periodic boundary conditions
        first_deriv[self.N-1,0] = 1j

        second_deriv = (2*iden-np.diag(self.off_diag,-1)-np.diag(self.off_diag,1)) #Create the second derivative matrix
        second_deriv[0,self.N-1] = -1 #Add periodic boundary conditions
        second_deriv[self.N-1,0] = -1

        hkin = 4.0*self.EC*(1/(dx**2)*second_deriv - 2*self.ng/(2*dx)*first_deriv + self.ng**2*iden) #Combine to create the kinetic energy term
        return hkin
    
    def Hampot(self):  
        dx = 2.*np.pi/self.N
        x = np.array([-np.pi+dx*i for i in np.arange(self.N)])
        cos_arr = -self.EJ*np.array([np.cos(xi) for xi in x])          
        cos = np.diag(cos_arr)
        return cos

#================================Gatemon in charge basis===========================================
class gatemon_charge(Qubit):#Averins model for the Gatemon
    def __init__(self, N, EC, gap, T, ng): #Parse all the constants
        super().__init__(N)
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T
        self.transform_variabels = False

        if(self.N%2 == 0):#Making sure that the resolution is uneven
            self.N += 1

        self.n_cut = int((self.N-1)/2)#Setting the charge cut-off

    def Hamkin(self):
        if(self.transform_variabels):
            n_array = np.array([(2*i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        else:
            n_array = np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = 4*self.EC*np.diag(n_array) #In charge basis the (n-ng)**2 term is a diagonal matrix
        hkin_2channel = np.kron(np.identity(2), hkin)
        return hkin_2channel
    
    def Hampot(self):
        sx = np.array([[0, 1],[1, 0]])
        sy = np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz = np.array([[1,0],[0,-1]])
        r = np.sqrt(1-self.T)#From Kringhoej 2020

        if(self.transform_variabels):
            #If I can do change of variable I might be able to use these sine and cosine functions
            off_diag = np.ones(self.N-1)  
            cos = (np.diag(off_diag, 1) + np.diag(off_diag, -1))/2
            sin = (np.diag(off_diag, 1) - np.diag(off_diag, -1))/(2j)
        else:
            #These are the functions for if I need to use the sin-half and cosine-half functions
            coords = np.arange(0, self.N)
            x, y = np.meshgrid(coords, coords)

            #====Interesting note, these matrices lift the degeneracy even for T=1 and ng=0 WHY?
            #Another note: Changing phi/2 -> phi we should change n -> 2n, but do we change ng?
            cos = -2*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2 - 1))
            sin = 4j*(x-y)*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2-1))

        if(self.verbose):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cos.real)
            ax1.set(title = "Cosine (Real part)")
            ax2.imshow(sin.imag)
            ax2.set(title = "Sine (Imaginary part)")


        return self.gap*(np.kron(sz, cos) + r*np.kron(sx, sin))

    def plot_wav(self, x, wavefuncs):
        super().plot_wav(x, wavefuncs)
        plt.xlabel(r'$n$')
        plt.ylabel(r'\varphi')


#================================Gatemon in flux basis===========================================
class gatemon_flux(Qubit):#Averins model for the Gatemon
    def __init__(self, N, EC, gap, T, ng): #Parse all the constants
        super().__init__(N)
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T
        self.transform_variabels = False
        self.dx = 2.*np.pi/self.N

        self.phi_array = np.linspace(-np.pi, np.pi, self.N)


    def Hamkin(self):
        vec = np.ones(self.N - 1)
        iden = np.identity(self.N)
        s0 = np.array([[1, 0],[0, 1]])
        
        first_deriv = 1j*np.diag(vec,-1)-1j*np.diag(vec,1) #Create the first derivative
        first_deriv[0,self.N-1] = -1j#Add periodic boundary conditions
        first_deriv[self.N-1,0] = 1j

        second_deriv = (2*iden-np.diag(vec,-1)-np.diag(vec,1)) #Create the second derivative matrix
        second_deriv[0,self.N-1] = -1 #Add periodic boundary conditions
        second_deriv[self.N-1,0] = -1

        hkin = 4.0*self.EC*(1/(self.dx**2)*second_deriv - 2*self.ng/(2*self.dx)*first_deriv + self.ng**2*iden) #Combine to create the kinetic energy term
        return  np.kron(s0, hkin)


    def Hampot(self):
        sx = np.array([[0, 1],[1, 0]])
        sy = np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz = np.array([[1,0],[0,-1]])
        r = np.sqrt(1-self.T)#From Kringhoej 2020

        cos = np.diag(np.cos(self.phi_array/2))
        sin = np.diag(np.sin(self.phi_array/2))

        return self.gap*(np.kron(sz, cos) + r*np.kron(sx, sin))