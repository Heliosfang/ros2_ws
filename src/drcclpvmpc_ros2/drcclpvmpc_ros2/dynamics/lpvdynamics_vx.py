__author__ = 'Shiming Fang'
__email__ = 'sfang10@binghamton.edu'


import numpy as np
import casadi as ca

class BicycleDynamicsVx():

	def __init__(self, params):
		"""	specify model params here
		"""
		self.lf = params['lf']
		self.lr = params['lr']
		self.dr = self.lr/(self.lf+self.lr)
		self.mass = params['mass']
		self.Iz = params['Iz']

		self.Cf = params['Cf']
		self.Cr = params['Cr']

		self.Bf = params['Bf']
		self.Br = params['Br']
		self.Df = params['Df']
		self.Dr = params['Dr']

		self.Caf = params['Caf']
		self.Car = params['Car']

		self.max_vx = params['max_vx']
		self.min_vx = params['min_vx']

		self.max_vy = params['max_vy']
		self.min_vy = params['min_vy']

		self.max_omega = params['max_omega']
		self.min_omega = params['min_omega']

		self.Cm1 = params['Cm1']
		self.Cm2 = params['Cm2']
		self.Cr0 = params['Cr0']
		self.Cd = params['Cd']
		control_type = params['control_type']
		if control_type == 'pwm':
			self.approx = False
		else:
			self.approx = True

		self.n_states = 5 # x, y, phi, vy, omega
		self.n_inputs = 2 # delta, vx

	def sim_continuous(self, x0, u, t):
		"""	simulates the nonlinear continuous model with given input vector
			by numerical integration
			x0 is the initial state of size 6x1,[x,y,theta,vx,vy,omega], vx denote speed, delta denote steering angle, Delta denote distance traveled
			u is the input vector of size 2xn,[u_delta, u_acc], denote acceleration and rate of change in steering
			t is the time vector of size 1x(n+1)
		"""
		n_steps = u.shape[1]
		# print("steps :",n_steps)
		x = np.zeros([self.n_states, n_steps+1])
		dxdt = np.zeros([self.n_states, n_steps+1])
		dxdt[:,0] = self._diffequation(None, x0, u[:,0])
		x[:,0] = x0

		dt = t[1] - t[0]
		for ids in range(1, n_steps+1):
			x[:,ids] = x[:,ids-1] + dt * dxdt[:,ids-1]
			dxdt[:,ids] = self._diffequation(None, x[:,ids], u[:,ids-1])
		return x, dxdt

	def _diffequation(self, t, x, u):
		"""	write kinematics as first order ODE: dxdt = f(x(t))
			x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T, Delta denote the traveled distance
			u is a 2x1 vector: [d_steer, acc]^T
		"""
		steer = u[0]
		vx = u[1]
		psi = x[2]
		vy = x[3]
		omega = x[4]
		# Ffy, Frx, Fry = self.calc_forces(x, u)

		dxdt = np.zeros(6)
		dxdt[0] = vx*np.cos(psi) - vy*np.sin(psi)
		dxdt[1] = vx*np.sin(psi) + vy*np.cos(psi)
		dxdt[2] = omega
		dxdt[3] = 2*(self.Caf+self.Car)*vy/(self.mass*vx) + (-vx + ((2*self.Car*self.lr-2*self.Caf*self.lf)/(self.mass*vx)))*omega + 2*self.Caf*steer/self.mass
		dxdt[4] = 2*vy*(self.Car*self.lr-self.Caf*self.lf)/(self.Iz*vx) + 2*omega*(self.Caf*self.lf*self.lf+self.Car*self.lr*self.lr)/(self.Iz*vx) + 2*self.Caf*self.lf*steer/self.Iz
		return dxdt

	def sim_next_state(self,x0,u0,dt):
		steer = u0[0]
		vx = np.clip(u0[1],self.min_vx,self.max_vx)
		psi = x0[2]
		vy = x0[3]
		omega = x0[4]
		# Ffy, Frx, Fry = self.calc_forces(x0, u0)
		dxdt = np.zeros(6)
		dxdt[0] = vx*np.cos(psi) - vy*np.sin(psi)
		dxdt[1] = vx*np.sin(psi) + vy*np.cos(psi)
		dxdt[2] = omega
		dxdt[3] = -(self.Caf+self.Car)*vy/(self.mass*vx) + ((self.Car*self.lr-self.Caf*self.lf)/(self.mass*vx))*omega + self.Caf*steer/self.mass
		dxdt[4] = vy*(self.Car*self.lr-self.Caf*self.lf)/(self.Iz*vx) - omega*(self.Caf*self.lf*self.lf+self.Car*self.lr*self.lr)/(self.Iz*vx) + self.Caf*self.lf*steer/self.Iz

		return x0+dxdt*dt
	
	def sim_states(self,x0,u,dt):
		x0_ = x0.reshape((self.n_states,))
		horizon = u.shape[1]
		x_sim = np.zeros((self.n_states,horizon+1))
		x_sim[:,0] = x0_

		for i in range(horizon):
			ui = u[:,i]
			xi = self.sim_next_state(x0_,ui,dt)
			vy = np.clip(xi[3],self.min_vy,self.max_vy)
			omega = np.clip(xi[4],self.min_omega,self.max_omega)

			xi[3] = vy
			xi[4] = omega

			x_sim[:,i+1] = xi
			x0_ = xi
		
		return x_sim

	def LPV_next_state(self,x0,u0,p_vx,p_phi,dt):
		x0 = ca.DM(x0)
		u0 = ca.DM(u0)
		# print("x0 :",x0)
		# print("u0 :",u0)
		if self.approx:
			A_i = ca.DM.zeros(self.n_states,self.n_states)
			A_i[0,3] = -ca.sin(p_phi)
			A_i[1,3] = ca.cos(p_phi)
			A_i[2,4] = 1
			A_i[3,3] = self.A33(p_vx,p_phi)
			A_i[3,4] = self.A34(p_vx,p_phi)
			A_i[4,3] = self.A43(p_vx,p_phi)
			A_i[4,4] = self.A44(p_vx,p_phi)

			B_i = ca.DM.zeros(self.n_states,self.n_inputs)
		
			B_i[0,1] = ca.cos(p_phi)
			B_i[1,1] = ca.sin(p_phi)
			B_i[3,0] = self.B30(p_vx,p_phi)
			B_i[4,0] = self.B40(p_vx,p_phi)
			A_i = ca.diag(ca.DM.ones(self.n_states)) + A_i*dt
			B_i = B_i * dt
			return A_i @ x0 + B_i @ u0
	
	def LPV_states(self,x0,u,p_vx,p_phi,dt):
		x0_ = x0
		horizon = u.shape[1]
		x_sim = np.zeros((self.n_states,horizon+1))
		x_sim[:,0] = x0_

		for i in range(horizon):
			ui = u[:,i]
			xi = self.LPV_next_state(x0_,ui,p_vx[0,i],p_phi[0,i],dt)
			xi = np.array(xi).squeeze()
			# print("lpv phi :",xi[2])
			x_sim[:,i+1] = xi
			x0_ = xi

		return x_sim



	def betaf(self):
		return self.Caf/self.mass

	def gammaf(self):
		return self.Caf*self.lf/self.Iz

	def betar(self):
		return self.Car/self.mass

	def gammar(self):
		return self.lr*self.Car/self.Iz

	def A33(self,p_vx,p_phi):
		betaf = self.betaf()
		betar = self.betar()
		return -1*(betaf+betar)/p_vx

	def A34(self,p_vx,p_phi):
		betaf = self.betaf()
		betar = self.betar()
		return (betar*self.lr-betaf*self.lf)/p_vx

	def A43(self,p_vx,p_phi):
		gammaf = self.gammaf()
		gammar = self.gammar()
		return (gammar-gammaf)/p_vx

	def A44(self,p_vx,p_phi):
		gammaf = self.gammaf()
		gammar = self.gammar()
		return -1*(gammaf*self.lf+gammar*self.lr)/p_vx


	def B30(self,p_vx,p_phi):
		betaf = self.betaf()
		return betaf

	def B40(self,p_vx,p_phi):
		gammaf = self.gammaf()
		return gammaf