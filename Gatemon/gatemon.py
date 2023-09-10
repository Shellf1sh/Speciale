"""
Program to find eigenstates of gatemon qubit
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def mp(x):
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in x]))

#parameters
class Qubit():
       
   EC = .1
   ng = 0
   flux = 0
   n = 3
   eigvals = []   
   eigvecs = []
   t = 0.5
   model='beenakker'
   
   
   def w(self):
      return(np.sqrt(self.EC*2))
    
   def hamkin(self):
      
      vec = np.ones(self.n-1)
      iden = np.identity(self.n)
      dx = 2.*np.pi/self.n
      #d = -1j/dx/2.
      #pforward = 1j*(np.diag(vec,1)-iden)/dx       
      #pforward[self.n-1, 0] = -pforward[0, 0]
      #pbackward = 1j*(iden-np.diag(vec,-1))/dx
      #pbackward[0, self.n-1] = -pbackward[0, 0]
      #hkin = 4.*self.EC*np.matmul(pforward-self.ng*iden,pbackward-self.ng*iden)
      hkin = (2*iden-np.diag(vec,1)-np.diag(vec,-1))
      hkin[0,self.n-1] = -1
      hkin[self.n-1,0] = -1
      hkin *= 4.*self.EC/(dx**2)      
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
          sy=np.array([[0,-1j],[1j,0]])
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
      
   def eigsys(self):
      v, w =  np.linalg.eig(self.ham())
      v = v.real
      order = np.argsort(v)
      self.eigvals = np.array([v[i] for i in order])
      self.eigvecs = np.array([w[:,i] for i in order])     
      
   def plotspect(self,xmin=-0.75*np.pi,xmax=0.75*np.pi,npoints=100,neigs=10,pr=False):
       
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
       plt.title(self.model+', $\mathcal{t}$=' + str(self.t) + ', $E_C$ = ' + str(q.EC))
       plt.xlabel(r'$\varphi$')
       plt.ylabel(r'Energy')
       if pr:
           savefigure('spect.pdf')      
       plt.show()
       
   def plotwf(self,n,pr=False):
       
       dx = 2*np.pi/self.n
       x = np.array([-np.pi+dx*i for i in range(self.n)])
       if len(self.eigvecs)==0:
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
      
def savefigure(filename):
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(filename)     
    

       
###############################################################################
#
#              MAIN PROGRAM
#
###############################################################################       
       
       
q = Qubit()
q.model='averin'
q.n=200
q.t=0.91
q.ng=0
q.EC=.1
q.spect()
q.plotspect(neigs=10,xmin=-2*np.pi,xmax=2*np.pi,pr=True)
eav=q.eigvals
print('averin matrix element: ', q.dH01())
q.plotwf(5)

a=[]
t=[]
for i in range(10):
    q.t=(i+1.)/10.
    t.append(q.t)
    a.append(abs(q.dH01())**2)
plt.plot(t,a)