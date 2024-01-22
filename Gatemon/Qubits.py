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
            print("The Hamiltonian is not hermitian!!!")

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
        constant_1f_ng = 10/4*(2*np.pi)**2 *1/abs(self.eigvals[1]-self.eigvals[0])
        constant_1f_ng_new = (4*2*np.pi)**2 * 1e-8 * 1/abs(self.eigvals[1]-self.eigvals[0])
        constant_ohmic_ng = (4*2*np.pi*5.2)**2 * 1e-9 * (self.eigvals[1]-self.eigvals[0])

        mel_C = self.matrix_element_C()
        #mel_F = self.matrix_element_F()
        if(self.verbose):
            print("Charge matrix element: " + str(mel_C))
            #print("Phi matrix element: " + str(mel_F))
            print("Qubit energies: " + str(self.eigvals[0]) + " and " + str(self.eigvals[1]))
            print("Qubit frequency: " + str(self.eigvals[1]-self.eigvals[0]))
        #The decay rates are return in GHz which is 1/ns
        return np.array([constant_1f_ng_new*mel_C, constant_ohmic_ng*mel_C])


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

        #DH = np.diag(np.array([-8*self.EC*(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))
        DH = np.diag(np.array([(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2 * self.EC**2 #* self.EJ/beta**2

        return mel
    
    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]
        off_diag = np.ones(self.N - 1)
        
        phi_matrix = 1/2j * (np.diag(off_diag, 1) + np.diag(-1*off_diag, -1)) #Create the first derivative
        #This is technically a sin(phi) matrix element in charge basis
        phi_matrix[0,self.N-1] = -1/2j#Add periodic boundary conditions
        phi_matrix[self.N-1,0] = 1/2j

        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2

        return mel
    
    def dephasing_rate_ng(self):
        ng_original = self.ng
        ng_plus = ng_original + 0.002
        ng_minus = ng_original - 0.002

        self.ng = ng_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.ng = ng_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(ng_plus - ng_minus)

        self.ng = ng_original

        return np.sqrt(1e-8/4*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))

        



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

        DH = (n/(2*self.dx) - self.ng*np.eye(self.N)) #This is dH/dng

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2* self.EC**2

        return mel
    
    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        sin = np.sin(np.linspace(-np.pi, np.pi, self.N))
        sin_matrix = np.diag(sin)

        mel = np.absolute(np.conjugate(state1.T) @ sin_matrix @ state0)**2

        return mel
    
    def dephasing_rate_ng(self):
        ng_original = self.ng
        ng_plus = ng_original + 0.002
        ng_minus = ng_original - 0.002

        self.ng = ng_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.ng = ng_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(ng_plus - ng_minus)

        self.ng = ng_original

        return np.sqrt(1e-8/4*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))


    

#================================Gatemon in charge basis===========================================
class gatemon_charge(Qubit):#Averin model for the Gatemon
    def __init__(self, N, EC, gap, T, ng): #Parse all the constants
        super().__init__(N)
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T
        #There is no Beenakker model in charge basis because it would require numerical integration 
        #of every matrix element of the sqrt(1-sin(phi/2)^2) operator
        
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

        self.r = np.sqrt(1-self.T)

        self.coords = np.arange(-self.n_cut, self.n_cut+1)
        
        x, y = np.meshgrid(self.coords, self.coords)

        #The Fourier transformed cos(phi/2) and sin(phi/2)
        cos = -2*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2 - 1))
        sin = -4j*(x-y)*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2-1))

        return self.gap*(np.kron(sz, cos) + self.r*np.kron(sx, sin))

    def matrix_element_C(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        DH = np.diag(np.array([self.EC*(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))

        DH = np.kron(np.eye(2), DH)

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2 * self.EC**2

        return mel
    
    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]
        off_diag = np.ones(self.N - 1)
        
        phi_matrix = 1/2j * (np.diag(off_diag, 1) + np.diag(-1*off_diag, -1)) #Create the first derivative
        
        phi_matrix[0,self.N-1] = -1/2j#Add periodic boundary conditions
        phi_matrix[self.N-1,0] = 1/2j

        phi_matrix = np.kron(np.eye(2), phi_matrix)

        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2

        return mel
    
    def dephasing_rate_ng(self):
        ng_original = self.ng
        ng_plus = ng_original + 0.002
        ng_minus = ng_original - 0.002

        self.ng = ng_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.ng = ng_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(ng_plus - ng_minus)

        self.ng = ng_original

        return np.sqrt(1e-8/4*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))


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
        n[0,self.N-1] = 1j/(2*self.dx)#Add periodic boundary conditions
        n[self.N-1,0] = -1j/(2*self.dx)

        n2 = -1/(self.dx**2)*(-2*iden+np.diag(off_diag,-1)+np.diag(off_diag,1)) #Create the second derivative matrix
        n2[0,self.N-1] = -1/(self.dx**2) #Add periodic boundary conditions
        n2[self.N-1,0] = -1/(self.dx**2)

        n_const = self.ng*np.identity(self.N)

        hkin = 4.0*self.EC*(n2 - 2*self.ng*n + n_const@n_const) #Combine to create the kinetic energy term
        
        if(self.beenakker):
            return hkin

        return  np.kron(np.eye(2), hkin)


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

        DH = -(1/(2*self.dx)*n - self.ng*np.eye(self.N))
        if(not self.beenakker):
            DH = np.kron(np.eye(2), DH)

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2 *self.EC**2

        return mel

    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        phi_matrix = np.diag(np.sin(self.phi_array))
        if(not self.beenakker):
            phi_matrix = np.kron(np.eye(2), phi_matrix)
        
        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2

        return mel
    
    def dephasing_rate_ng(self):
        ng_original = self.ng
        ng_plus = ng_original + 0.002
        ng_minus = ng_original - 0.002

        self.ng = ng_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.ng = ng_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(ng_plus - ng_minus)

        self.ng = ng_original

        return np.sqrt(1e-8/4*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))

    
    
    def set_resolution(self, N):
        self.N = N
        self.dx = 2.0*np.pi/self.N
        self.phi_array = np.linspace(-np.pi, np.pi, self.N)

