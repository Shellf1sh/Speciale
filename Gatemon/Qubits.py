import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

kB = sc.constants.Boltzmann
hbar = sc.constants.hbar

#================================General Qubits==============================================
class Qubit:#The base qubit class with the things that all qubits require
    def __init__(self, N):
        self.N = int(N)
        self.verbose = False
        self.eigvals = np.array([])   
        self.eigvecs = np.array([])

    def Hamkin(self):
        return np.array([]) #This is basis and design specific, see child classes

    def Hampot(self):
        return np.array([]) #This is basis and design specific, see child classes

    def Hamiltonian(self):
        Hamtot = self.Hamkin() + self.Hampot()
        if(sc.linalg.ishermitian(Hamtot) != True):
            print("The Hamiltinian is not hermitian!!!")

        return Hamtot

    def solve(self):
        eigenvalues, eigenvectors =  np.linalg.eigh(self.Hamiltonian())
        self.eigvals = eigenvalues
        self.eigvecs = eigenvectors

    def matrix_element_C(self):
        return 0 #This is basis and design specific, see child classes
    
    def matrix_element_F(self):
        return 0 #This is basis and design specific, see child classes

    def T_1_gamma(self):
        #constant_1f_flux = 2*np.pi*A_flux**2/(sc.constants.hbar*abs(self.eigvals[1]-self.eigvals[0]))
        constant_1f_ng = 2*np.pi*10/(4*abs(self.eigvals[1]-self.eigvals[0]))
        constant_ohmic_ng = 5.2**2 * (self.eigvals[1]-self.eigvals[0])/(4*2*np.pi)
        pre_costant_dielectric = (self.eigvals[1]-self.eigvals[0])*100/(2*self.EC)
        post_costant_dielectric = 1/np.tanh((self.eigvals[1]-self.eigvals[0])*hbar*1e9/(2*kB*0.02)) + 1
        #The temperature is set at 20mK like in the litterature
        costant_dielectric = pre_costant_dielectric * post_costant_dielectric

        mel_C = self.matrix_element_C()
        mel_F = self.matrix_element_F()
        print("Charge matrix element: " + str(mel_C))
        print("Phi matrix element: " + str(mel_F))
        print("Qubit energies: " + str(self.eigvals[0]) + " and " + str(self.eigvals[1]))
        print("Qubit frequency: " + str(self.eigvals[1]-self.eigvals[0]))

        return np.array([constant_1f_ng*mel_C, constant_ohmic_ng*mel_C, costant_dielectric*mel_F])

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
        n_squared =  np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = 4*self.EC*np.diag(n_squared) #In charge basis the (n-ng)**2 term is a diagonal matrix
        return hkin

    def Hampot(self):
        off_diag = np.ones(self.N - 1)     
        hpot = -self.EJ/2*(np.diag(off_diag, 1) + np.diag(off_diag, -1))
        return hpot
    
    def plot_wav(self, number_of_wavefuncs):
        print(self.eigvals[:number_of_wavefuncs])
        #plt.plot(self.n_array, self.n_squared, '-o', color="black")
        return super().plot_wav(self.n_array, number_of_wavefuncs)
    
    def matrix_element_C(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        DH = np.diag(np.array([8*self.EC*(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))
        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2

        return mel
    
    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]
        off_diag = np.ones(self.N - 1)
        
        phi_matrix = -1j/2 * (np.diag(off_diag, 1) + np.diag(-1*off_diag, -1)) #Create the first derivative

        #The charge basis is not periodic
        #phi_matrix[0,self.N-1] = -1j/2#Add periodic boundary conditions
        #phi_matrix[self.N-1,0] = 1j/2

        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2

        return mel
    

#================================Transmon in flux basis===========================================
class transmon_flux(Qubit):
    def __init__(self, N, EC, EJ, ng):
        super().__init__(N)
        self.EC = EC
        self.EJ = EJ
        self.ng = ng
        self.off_diag = np.ones(self.N-1)  
        self.dx = 2*np.pi/self.N
        self.phi_array = np.linspace(-np.pi, np.pi, self.N)

    def Hamkin(self):
        off_diag = np.ones(self.N-1) 
        n = -1j*(np.diag(off_diag,-1)-np.diag(off_diag,1)) #Create the first derivative
        n[0,self.N-1] = -1j#Add periodic boundary conditions
        n[self.N-1,0] = 1j

        n2 = (2*np.eye(self.N)-np.diag(self.off_diag,-1)-np.diag(self.off_diag,1)) #Create the second derivative matrix
        n2[0,self.N-1] = -1 #Add periodic boundary conditions
        n2[self.N-1,0] = -1

        hkin = 4.0*self.EC*(1/(self.dx**2)*n2 - 2*self.ng*n/(2*self.dx) + self.ng**2*np.eye(self.N)) #Combine to create the kinetic energy term
        return hkin
    
    def Hampot(self):  
        cos_arr = np.cos(self.phi_array)        
        cos_matrix = -self.EJ*np.diag(cos_arr)
        return cos_matrix

    def matrix_element_C(self):
        off_diag = np.ones(self.N-1) 
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        #Derivative of the Hamiltonian
        n = 1j*np.diag(off_diag,-1)-1j*np.diag(off_diag,1) #Create the first derivative
        n[0,self.N-1] = 1j#Add periodic boundary conditions
        n[self.N-1,0] = -1j

        DH = 8*self.EC*(n/(2*self.dx) - self.ng*np.eye(self.N))

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2

        return mel
    
    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        phi_matrix = np.diag(np.linspace(-np.pi, np.pi, self.N))

        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2

        return mel
#================================Gatemon in charge basis===========================================
class gatemon_charge(Qubit):#Averin/Kringhøj model for the Gatemon
    def __init__(self, N, EC, gap, T, ng): #Parse all the constants
        super().__init__(N)
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T
        
        if(self.N%2 == 0):#Making sure that the resolution is uneven
            self.N += 1

        self.n_cut = int((self.N-1)/2)#Setting the charge cut-off


    def Hamkin(self):
        n_array = np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = 4*self.EC*np.diag(n_array) #In charge basis the (n-ng)**2 term is a diagonal matrix
        hkin_2channel = np.kron(np.identity(2), hkin)
        return hkin_2channel
    
    def Hampot(self):
        sx = np.array([[0, 1],[1, 0]])
        sy = np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz = np.array([[1,0],[0,-1]])

        self.r = np.sqrt(1-self.T)#From Kringhoej 2020

        self.coords = np.arange(-self.n_cut, self.n_cut+1)
        
        x, y = np.meshgrid(self.coords, self.coords)

        #The Fourier transformed cos(phi/2) and sin(phi/2)
        cos = -2*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2 - 1))
        sin = -4j*(x-y)*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2-1))

        if(self.verbose):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cos.real)
            ax1.set(title = "Cosine (Real part)")
            ax2.imshow(sin.imag)
            ax2.set(title = "Sine (Imaginary part)")

        return self.gap*(np.kron(sz, cos) + self.r*np.kron(sx, sin))

    def matrix_element_C(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        DH = np.diag(np.array([8*self.EC*(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))

        DH = np.kron(DH, np.eye(2))

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2

        return mel
    
    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]
        off_diag = np.ones(self.N - 1)
        
        phi_matrix = -1j/(2) * (np.diag(off_diag, 1) + np.diag(-1*off_diag, -1)) #Create the first derivative

        phi_matrix = np.kron(phi_matrix, np.eye(2))

        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2

        return mel

    def plot_wav(self, x, wavefuncs):
        super().plot_wav(x, wavefuncs)
        plt.xlabel(r'$n$')
        plt.ylabel(r'\varphi')

    def set_resolution(self, N):
        self.N = N
        if(self.N%2 == 0):#Making sure that the resolution is uneven
            self.N += 1

        self.n_cut = int((self.N-1)/2)#Setting the charge cut-off

#================================Gatemon in flux basis===========================================
class gatemon_flux(Qubit):#Averins model for the Gatemon
    def __init__(self, N, EC, gap, T, ng): #Parse all the parameters
        super().__init__(N)
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T
        self.beenakker = False
        self.dx = 2.0*np.pi/self.N

        self.phi_array = np.linspace(-np.pi, np.pi, self.N)

    def Hamkin(self):
        off_diag = np.ones(self.N - 1)
        iden = np.identity(self.N)
        
        n = -1j/(2*self.dx) * (np.diag(off_diag, 1) + np.diag(-1*off_diag, -1)) #Create the first derivative
        n[0,self.N-1] = -1j/(2*self.dx)#Add periodic boundary conditions
        n[self.N-1,0] = 1j/(2*self.dx)

        n2 = -1/(self.dx**2)*(-2*iden+np.diag(off_diag,-1)+np.diag(off_diag,1)) #Create the second derivative matrix
        n2[0,self.N-1] = -1/(self.dx**2) #Add periodic boundary conditions
        n2[self.N-1,0] = -1/(self.dx**2)

        n_const = self.ng*np.identity(self.N)

        hkin = 4.0*self.EC*(n2 - 2*self.ng*n + n_const@n_const) #Combine to create the kinetic energy term
        
        if(self.beenakker):
            return hkin

        return  np.kron(np.identity(2), hkin)


    def Hampot(self):
        sx = np.array([[0, 1],[1, 0]])
        sy = np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz = np.array([[1,0],[0,-1]])
        r = np.sqrt(1-self.T)#From Kringhoej 2020

        cos = np.diag(np.cos(self.phi_array/2))
        sin = np.diag(np.sin(self.phi_array/2))

        if(self.beenakker):
            self.sin_half = np.sin(self.phi_array/2)
            beenakker_pot = -self.gap*np.sqrt(1-(self.T)*self.sin_half**2)
            return np.diag(beenakker_pot)

        return -self.gap*(np.kron(sz, cos) + r*np.kron(sx, sin))
    
    def matrix_element_C(self):
        off_diag = np.ones(self.N-1) 
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        #Derivative of the Hamiltonian
        n = 1j*np.diag(off_diag,-1)-1j*np.diag(off_diag,1) #Create the first derivative
        n[0,self.N-1] = -1j#Add periodic boundary conditions
        n[self.N-1,0] = 1j

        DH = (1/(2*self.dx)*n + self.ng*np.eye(self.N))
        if(not self.beenakker):
            DH = np.kron(DH, np.eye(2))

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2

        return mel

    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        phi_matrix = np.diag(self.phi_array)
        if(not self.beenakker):
            phi_matrix = np.kron(phi_matrix, np.eye(2))
        
        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2

        return mel
    
    
    def set_resolution(self, N):
        self.N = N
        self.dx = 2.0*np.pi/self.N
        self.phi_array = np.linspace(-np.pi, np.pi, self.N)

