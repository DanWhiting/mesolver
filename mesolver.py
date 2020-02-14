from __future__ import division
import qutip as qt
import numpy as np
from scipy.linalg import eig, inv

''' 
Author - Daniel J. Whiting 
'''

kB=1.380658e-23
amu=1.6605402e-27

class mesystem():
    """ master equation system """
    def __init__(self, N, Hfunc, detunings, rabi_freqs, c_ops=[]):
        self.c_ops = c_ops
        self.N = N
        self.Hfunc = Hfunc
        self.dets = np.array(detunings)
        self.rabi_freqs = np.array(rabi_freqs)
        # Calculate Lindblad dissipator from list of collapse operators c_ops
        self.lindblad_dissipator = 0
        for c_op in c_ops:
            self.lindblad_dissipator += qt.lindblad_dissipator(c_op).full()

    def evolve(self, rho0 = None, times = np.linspace(0,1e-7,int(1e2))):
        rho = np.zeros((len(times),self.N**2),dtype='complex')
        if type(rho0) == type(None):
            rho[0,0] = 1
        else:
            rho[0,:] = rho0
        rho0 = np.mat(rho[0,:].reshape((self.N**2,1)))
        H = self.Hfunc(self.rabi_freqs,self.dets)
        Hrho = np.kron(H,np.eye(self.N))
        rhoH = np.kron(np.eye(self.N),np.transpose(H))
        L = -1j*(Hrho-rhoH)+self.lindblad_dissipator
        HV,HD = eig(L)
        for i in range(1,len(times)):
            rho[i,:] = (HD*np.mat(np.diag(np.exp(HV*times[i])))*inv(HD) * rho0).A1
        return rho
    
    def evolve_qutip(self, rho0 = None, times = np.linspace(0,1e-7,int(1e2))):
        rho = np.zeros((len(times),self.N**2),dtype='complex')
        if type(rho0) == type(None):
            rho[0,0] = 1
        else:
            rho[0,:] = rho0
        rho0 = qt.Qobj(rho[0,:].reshape((self.N,self.N)))
        H = qt.Qobj(self.Hfunc(self.rabi_freqs,self.dets))
        odedata = qt.mesolve(H,rho0,times,self.c_ops,[])
        for i in range(1,len(times)):
            rho[i,:] = odedata.states[i].full().flatten()
        return rho

    def evolve_ss(self):
        self.steady_state = 0
        H = self.Hfunc(self.rabi_freqs,self.dets)
        Hrho = np.kron(H,np.eye(self.N))
        rhoH = np.kron(np.eye(self.N),np.transpose(H))
        L = -1j*(Hrho-rhoH)+self.lindblad_dissipator
        a = np.vstack((L,np.eye(self.N).flatten()))
        b = np.zeros(self.N**2+1); b[-1] = 1
        sol = np.linalg.lstsq(a,b)[0]
        return sol
    
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	def H_int(O,D):
		O = O[0]
		D = D[0]
		H = np.mat([[0,O/2],[O/2,-D]])
		return H

	detunings = 2*np.pi * np.array([0])
	rabi_freqs = 2*np.pi * np.array([10e6])

	Gamma = 2*np.pi * 6e6
	c_ops = []
	c_ops.append(Gamma**.5*qt.basis(2,0)*qt.basis(2,1).dag())

	sys = mesystem(2, H_int, detunings, rabi_freqs, c_ops)
	
	t = np.linspace(0,2e-7,2e2)
	y = sys.evolve_qutip(times = t)[:,1].imag/rabi_freqs
	plt.figure()
	plt.ylabel('Im$[\\rho_{01}]/\\Omega$')
	plt.xlabel('Time (s)')
	plt.plot(t,y)
	
	plt.show()