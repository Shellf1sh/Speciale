import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import copy
from scipy import constants
from scipy import sparse
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from functions import kron_sum

kB = constants.Boltzmann
hbar = constants.hbar

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
        return Hamtot

    def solve(self):
        ham = self.Hamiltonian()
        try:
            eigenvalues, eigenvectors = eigsh(ham, min([20, ham.shape[0]-2]), which ="SA", maxiter = ham.shape[0]*100)
        except ArpackNoConvergence as e:
            print("No convergence")
            eigenvalues = e.eigenvalues
            eigenvectors = e.eigenvectors

        indices = np.argsort(eigenvalues) #Makes sure that the 

        self.eigvals = eigenvalues[indices]
        self.eigvecs = eigenvectors[:,indices]

    def matrix_element_C(self):
        return 0 #This is basis and design specific, see child classes
    
    def matrix_element_F(self):
        return 0 #This is basis and design specific, see child classes

    def T_1_gamma(self):
        #constant_1f_flux = 2*np.pi*A_flux**2/(sc.constants.hbar*abs(self.eigvals[1]-self.eigvals[0]))
        constant_1f_ng = (4*2*np.pi)**2 * 1e-8 * 1/abs(self.eigvals[1]-self.eigvals[0])
        constant_ohmic_ng = (4*2*np.pi*5.2)**2 * 1e-9 * (self.eigvals[1]-self.eigvals[0])

        mel_C = self.matrix_element_C()
        #mel_F = self.matrix_element_F()
        if(self.verbose):
            print("Charge matrix element: " + str(mel_C))
            #print("Phi matrix element: " + str(mel_F))
            print("Qubit energies: " + str(self.eigvals[0]) + " and " + str(self.eigvals[1]))
            print("Qubit frequency: " + str(self.eigvals[1]-self.eigvals[0]))
        #The decay rates are return in GHz which is 1/ns
        return np.array([constant_1f_ng*mel_C, constant_ohmic_ng*mel_C])


    def plot_wav(self, x, wavefuncs):
        if(len(self.eigvals) == 0):#If the Hamiltonian hasn't been solved yet then solve it
            self.solve()
        
        for i in range(wavefuncs):
            plt.plot(x, np.conjugate(self.eigvecs[i]) * self.eigvecs[i] + self.eigvals[i], '-o')

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
        s_hkin = sparse.dia_matrix(hkin)
        return s_hkin

    def Hampot(self):
        off_diag = np.ones(self.N - 1)     
        hpot = -self.EJ/2*(np.diag(off_diag, 1) + np.diag(off_diag, -1))
        s_hpot = sparse.dia_matrix(hpot)
        return s_hpot
    
    def plot_wav(self, number_of_wavefuncs):
        print(self.eigvals[:number_of_wavefuncs])
        #plt.plot(self.n_array, self.n_squared, '-o', color="black")
        return super().plot_wav(self.n_array, number_of_wavefuncs)
    
    def matrix_element_C(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        #DH = np.diag(np.array([-8*self.EC*(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))
        DH = sparse.diags(np.array([(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2 * self.EC**2 

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

        return np.sqrt(1e-8/2*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))



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
        s_hkin = sparse.dia_matrix(hkin)
        return s_hkin
    
    def Hampot(self):  
        cos_arr = np.cos(self.phi_array)        
        cos_matrix = -self.EJ*np.diag(cos_arr)
        s_cos = sparse.dia_matrix(cos_matrix)
        return s_cos
    

    def matrix_element_C(self):
        off_diag = np.ones(self.N-1) 
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        #Derivative of the Hamiltonian
        n = 1j*np.diag(off_diag,-1)-1j*np.diag(off_diag,1) #Create the first derivative
        n[0,self.N-1] = 1j#Add periodic boundary conditions
        n[self.N-1,0] = -1j

        DH = sparse.coo_matrix(n/(2*self.dx) - self.ng*np.eye(self.N)) #This is dH/dng

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
   
    #Calculate the dephasing time
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

        return np.sqrt(1e-8/2*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))


    

#================================Gatemon in charge basis===========================================
class gatemon_charge(Qubit):#Averin model for the Gatemon
    def __init__(self, N, EC, gap, T, ng): #Parse all the constants
        super().__init__(N)
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T
        #There is no Beenakker model in charge basis because it would require numerical integration 
        #of every matrix element of the sqrt(1-Tsin(phi/2)^2) operator
        
        if(self.N%2 == 0):#Making sure that the resolution is uneven
            self.N += 1

        self.n_cut = int((self.N-1)/2)#Setting the charge cut-off

        self.T_is_list = False
        if((type(T) == list or type(T) == np.ndarray)):
            self.T_len = len(T)
            self.T_is_list = True
        elif(type(T) == int or type(T) == float):
            self.T_len = 1

    def Hamkin(self):
        n_array = np.array([(i-self.ng)**2 for i in range(-self.n_cut, self.n_cut+1)])
        hkin = 4*self.EC*np.diag(n_array) #In charge basis the (n-ng)**2 term is a diagonal matrix
        H_kin = sparse.kron(sparse.eye(2), hkin)

        if(self.T_is_list):           
            other_channel = copy.copy(H_kin)
            for i in range(1,self.T_len):
                H_kin = kron_sum(H_kin, self.N, other_channel, self.N)
            
            s_hkin = H_kin
        else:
            s_hkin = sparse.kron(sparse.eye(2), hkin)
         
        #hkin_2level= np.kron(np.identity(2), hkin)
        #s_hkin = sparse.coo_matrix(hkin_2level)

        return s_hkin
    
    def Hampot(self):
        sx = np.array([[0, 1],[1, 0]])
        sy = np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz = np.array([[1,0],[0,-1]])

        #self.r = np.sqrt(1-self.T)

        self.coords = np.arange(-self.n_cut, self.n_cut+1)
        
        x, y = np.meshgrid(self.coords, self.coords)

        #The Fourier transformed cos(phi/2) and sin(phi/2)
        cos = sparse.coo_matrix(-2*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2 - 1)))
        sin = sparse.coo_matrix(-4j*(x-y)*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2 - 1)))

        if(not self.T_is_list):
            self.r = np.sqrt(1-self.T)
            H_pot = -self.gap*(sparse.kron(sz, cos) + self.r*sparse.kron(sx, sin))
            return sparse.coo_matrix(H_pot)
        elif(self.T_is_list):
            r = np.sqrt(1-self.T[0])
            H_pot = (sparse.kron(sz, cos) + r*sparse.kron(sx, sin))
            
            for i in range(1, self.T_len):
                r2 = np.sqrt(1-self.T[i])
                other_channel = (sparse.kron(sz, cos) + r2*sparse.kron(sx, sin))

                H_pot = kron_sum(H_pot, self.N, other_channel, self.N)

            return -self.gap*H_pot

        #return self.gap*(np.kron(sz, cos) + self.r*np.kron(sx, sin))

    def matrix_element_C(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]

        DH = sparse.diags(np.array([(i-self.ng) for i in range(-self.n_cut, self.n_cut+1)]))

        #DH_I2 = sparse.kron(sparse.eye(2), DH)

        DH_I2 = sparse.kron(sparse.eye(2**(self.T_len)), DH)
        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH_I2 @ state0)**2 * self.EC**2

        return mel
    
    def matrix_element_F(self):
        state0 = self.eigvecs[:,0]
        state1 = self.eigvecs[:,1]
        off_diag = np.ones(self.N - 1)
        
        phi_matrix = 1/2j * (np.diag(off_diag, 1) + np.diag(-1*off_diag, -1)) #Create the first derivative
        
        phi_matrix[0,self.N-1] = -1/2j#Add periodic boundary conditions
        phi_matrix[self.N-1,0] = 1/2j

        phi_matrix = np.kron(np.eye(2), phi_matrix)

        mel = np.absolute(np.conjugate(state1.T) @ phi_matrix @ state0)**2 * self.EC**2

        return mel
    
    def dephasing_rate_ng(self):
        ng_original = self.ng
        ng_plus = ng_original + 0.001
        ng_minus = ng_original - 0.001

        self.ng = ng_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.ng = ng_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(ng_plus - ng_minus)

        self.ng = ng_original

        return np.sqrt(1e-8/2*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))


    def plot_wav(self, x, wavefuncs):
        super().plot_wav(x, wavefuncs)
        plt.xlabel(r'$n$')
        plt.ylabel(r'\varphi')

    def set_resolution(self, N):
        self.N = N
        if(self.N%2 == 0):#Making sure that the resolution is uneven
            self.N += 1

        self.n_cut = int((self.N-1)/2)#Setting the charge cut-off

    def T_1_gamma_T(self): #We don't know the coupling constants A_T and B_T    
        qubit_freq = self.eigvals[1] - self.eigvals[0]

        self.coords = np.arange(-self.n_cut, self.n_cut+1)
        x, y = np.meshgrid(self.coords, self.coords)
        cos = -2*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2 - 1))
        sin = -4j*(x-y)*np.cos(np.pi*(x-y))/(np.pi*(4*(x-y)**2-1))

        operator = sparse.coo_matrix(-self.gap/(2*np.sqrt(1-self.T)) * sin)

        operator_I2 = sparse.kron(sparse.eye(2), operator)

        mel = np.abs(np.conjugate((self.eigvecs[:,1]).T) @ operator_I2 @ self.eigvecs[:,0])**2

        gamma_1f = (2*np.pi)**2 * mel * 1/np.abs(qubit_freq)
        gamma_ohmic = (2*np.pi)**2 * mel *qubit_freq

        return np.array([gamma_1f, gamma_ohmic])

    def dephasing_rate_T(self):
        T_original = self.T
        if(T_original < 0.999):
            T_plus = T_original + 0.001
        else:
            T_plus = T_original

        if(T_original > 0.001):
            T_minus = T_original - 0.001
        else:
            T_minus = T_original

        self.T = T_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.T = T_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(T_plus - T_minus)

        self.T = T_original

        return np.sqrt(first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))




#================================Gatemon in flux basis===========================================
class gatemon_flux(Qubit):#Averin and Beenakkers model for the Gatemon
    def __init__(self, N, EC, gap, T, ng): #Parse all the parameters
        super().__init__(N)
        self.N = N
        self.EC = EC
        self.gap = gap
        self.ng = ng
        self.T = T
        self.beenakker = False
        self.dx = 2.0*np.pi/self.N

        self.phi_array = np.linspace(-np.pi, np.pi, self.N)

        self.T_is_list = False
        if((type(T) == list or type(T) == np.ndarray)):
            self.T_len = len(T)
            self.T_is_list = True
        elif(type(T) == int or type(T) == float):
            self.T_len = 1

    def Hamkin(self):
        off_diag = np.ones(self.N - 1)
        iden = np.identity(self.N)
        
        n = -1j/(2*self.dx) * (np.diag(off_diag, 1) + np.diag(-1*off_diag, -1)) #Create the first derivative
        n[0,-1] = -1j/(2*self.dx)#Add periodic boundary conditions
        n[-1,0] = 1j/(2*self.dx)

        n2 = -1/(self.dx**2)*(-2*iden+np.diag(off_diag,-1)+np.diag(off_diag,1)) #Create the second derivative matrix
        n2[0,-1] = -1/(self.dx**2) #Add periodic boundary conditions
        n2[-1,0] = -1/(self.dx**2)

        n_const = self.ng*sparse.eye(self.N)

        hkin = 4.0*self.EC*(n2 - 2*self.ng*n + n_const@n_const) #Combine to create the kinetic energy term
        
        if(self.beenakker):
            return sparse.coo_matrix(hkin)
        
        #H_kin = sparse.kron(sparse.eye(2), hkin)


        H_kin = self.T_len*sparse.kron(sparse.eye(2**(self.T_len)), hkin)
        #print("Kin" + str(H_kin.shape))

        return  H_kin

    def Hampot(self):
        sx = np.array([[0, 1],[1, 0]])
        sy = np.array([[0,-1j],[1j,0]]) #Create the Pauli matrices
        sz = np.array([[1,0],[0,-1]])
        

        cos = sparse.diags(np.cos(self.phi_array/2))
        sin = sparse.diags(np.sin(self.phi_array/2))

        if(self.beenakker):
            sin_half = np.sin(self.phi_array/2)
            if(not self.T_is_list):
                self.beenakker_pot = -self.gap*np.sqrt(1-(self.T)*sin_half**2) 

            if(self.T_is_list):
                self.beenakker_pot = np.zeros_like(sin_half)
                for i in self.T:
                    self.beenakker_pot += -self.gap*np.sqrt(1-(i)*sin_half**2)

            return sparse.diags(self.beenakker_pot)

        if(not self.T_is_list):
            self.r = np.sqrt(1-self.T)
            H_pot = -self.gap*(sparse.kron(sz, cos) + self.r*sparse.kron(sx, sin))
            return sparse.coo_matrix(H_pot)
        elif(self.T_is_list):
            r = np.sqrt(1-self.T[0])
            H_pot = (sparse.kron(sz, cos) + r*sparse.kron(sx, sin))
            
            for i in range(1, self.T_len):
                r2 = np.sqrt(1-self.T[i])
                other_channel = (sparse.kron(sz, cos) + r2*sparse.kron(sx, sin))
                
                H_pot = kron_sum(H_pot, self.N, other_channel, self.N)

            return -self.gap*H_pot

        
    
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
            DH = np.kron(np.eye(2**(self.T_len)), DH)*self.T_len

        #The matrix element squared
        mel = np.absolute(np.conjugate(state1.T) @ DH @ state0)**2 * self.EC**2

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
        ng_plus = ng_original + 0.001
        ng_minus = ng_original - 0.001

        self.ng = ng_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.ng = ng_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(ng_plus - ng_minus) #Finite difference methode

        self.ng = ng_original

        return np.sqrt(1e-8/2*first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))

    def T_1_gamma_T(self): #We don't know the coupling constants A_T and B_T    
        qubit_freq = self.eigvals[1] - self.eigvals[0]

        cos = np.diag(np.cos(self.phi_array/2))
        sin = np.diag(np.sin(self.phi_array/2))

        if(self.beenakker):
            operator = self.gap*sin**2/(2*np.sqrt(1-self.T*sin**2))
        else:
            operator = -self.gap/(2*np.sqrt(1-self.T))*np.kron(np.eye(2), sin)

        mel = np.abs(np.conjugate(self.eigvecs[:,1].T) @ operator @ self.eigvecs[:,0])**2

        gamma_1f = (2*np.pi)**2 * mel * 1/np.abs(qubit_freq)
        gamma_ohmic = (2*np.pi)**2  * mel * qubit_freq

        return np.array([gamma_1f, gamma_ohmic])
    
    def dephasing_rate_T(self):
        T_original = self.T
        if(T_original < 0.999):
            T_plus = T_original + 0.001
        else:
            T_plus = T_original

        if(T_original > 0.001):
            T_minus = T_original - 0.001
        else:
            T_minus = T_original

        self.T = T_plus
        self.solve()
        omega_plus = self.eigvals[1] - self.eigvals[0]

        self.T = T_minus
        self.solve()
        omega_minus = self.eigvals[1] - self.eigvals[0]

        first_derivative = (omega_plus - omega_minus)/(T_plus - T_minus)

        self.T = T_original

        return np.sqrt(first_derivative**2*np.abs(np.log(2*np.pi*1e-6)))
    
    def set_resolution(self, N):
        self.N = N
        self.dx = 2.0*np.pi/self.N
        self.phi_array = np.linspace(-np.pi, np.pi, self.N)

