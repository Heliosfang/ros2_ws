__author__ = 'Shiming Fang'
__email__ = 'sfang10@binghamton.edu'


import numpy as np
import casadi as ca

class BicycleDynamics():

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

		self.n_states = 6
		self.n_inputs = 2

	def sim_continuous(self, x0, u, t):
		"""	simulates the nonlinear continuous model with given input vector
			by numerical integration
			x0 is the initial state of size 6x1,[x,y,theta,vx,vy,omega], vx denote speed, delta denote steering angle, Delta denote distance traveled
			u is the input vector of size 2xn,[u_delta, u_acc], denote acceleration and rate of change in steering
			t is the time vector of size 1x(n+1)
		"""
		n_steps = u.shape[1]
		# print("steps :",n_steps)
		x = np.zeros([6, n_steps+1])
		dxdt = np.zeros([6, n_steps+1])
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
		acc = u[1]
		psi = x[2]
		vx = x[3]
		vy = x[4]
		omega = x[5]
		Ffy, Frx, Fry = self.calc_forces(x, u)

		dxdt = np.zeros(6)
		dxdt[0] = vx*np.cos(psi) - vy*np.sin(psi)
		dxdt[1] = vx*np.sin(psi) + vy*np.cos(psi)
		dxdt[2] = omega
		dxdt[3] = (1/self.mass) * (Frx - Ffy*np.sin(steer)) + vy*omega
		# dxdt[3] = acc
		dxdt[4] = (1/self.mass) * (Fry + Ffy*np.cos(steer)) - vx*omega
		dxdt[5] = (1/self.Iz) * (Ffy*self.lf*np.cos(steer) - Fry*self.lr)
		return dxdt

	def sim_next_state(self,x0,u0,dt):
		steer = u0[0]
		acc = u0[1]
		psi = x0[2]
		vx = x0[3]
		vy = x0[4]
		omega = x0[5]
		Ffy, Frx, Fry = self.calc_forces(x0, u0)
		dxdt = np.zeros(6)
		dxdt[0] = vx*np.cos(psi) - vy*np.sin(psi)
		dxdt[1] = vx*np.sin(psi) + vy*np.cos(psi)
		dxdt[2] = omega
		dxdt[3] = (1/self.mass) * (Frx - Ffy*np.sin(steer)) + vy*omega
		dxdt[4] = (1/self.mass) * (Fry + Ffy*np.cos(steer)) - vx*omega
		dxdt[5] = (1/self.Iz) * (Ffy*self.lf*np.cos(steer) - Fry*self.lr)

		return x0+dxdt*dt
	
	def sim_states(self,x0,u,dt):
		x0_ = x0.reshape((self.n_states,))
		horizon = u.shape[1]
		x_sim = np.zeros((self.n_states,horizon+1))
		x_sim[:,0] = x0_

		for i in range(horizon):
			ui = u[:,i]
			xi = self.sim_next_state(x0_,ui,dt)
			vx = np.clip(xi[3],self.min_vx,self.max_vx)
			vy = np.clip(xi[4],self.min_vy,self.max_vy)
			omega = np.clip(xi[5],self.min_omega,self.max_omega)

			xi[3] = vx
			xi[4] = vy
			xi[5] = omega

			x_sim[:,i+1] = xi
			x0_ = xi
		
		return x_sim
	
	def calc_forces(self, x, u, return_slip=False):
		steer = u[0]
		psi = x[2]
		vx = x[3]
		vy = x[4]
		omega = x[5]
		# rolling friction and drag are ignored
		
		if self.approx:
			acc = u[1]
			Frx = self.mass*acc
		else:
			pwm = u[1]
			Frx = (self.Cm1-self.Cm2*vx)*pwm - self.Cr0 - self.Cd*(vx**2)

		alphaf = steer - np.arctan2((self.lf*omega + vy), abs(vx))
		alphar = np.arctan2((self.lr*omega - vy), abs(vx))
		Ffy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alphaf))
		Fry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alphar))
		if return_slip:
			return Ffy, Frx, Fry, alphaf, alphar
		else:
			return Ffy, Frx, Fry

	def LPV_next_state(self,x0,u0,p_vx,p_vy,p_phi,p_delta,dt):
		x0 = ca.DM(x0)
		u0 = ca.DM(u0)
		# print("x0 :",x0)
		# print("u0 :",u0)
		if self.approx:
			A_i = ca.DM.zeros(6,6)
			A_i[0,3] = ca.cos(p_phi)
			A_i[0,4] = -ca.sin(p_phi)
			A_i[1,3] = ca.sin(p_phi)
			A_i[1,4] = ca.cos(p_phi)
			A_i[2,5] = 1
			A_i[3,4] = self.A34(p_vx,p_vy,p_phi,p_delta)
			A_i[3,5] = self.A35(p_vx,p_vy,p_phi,p_delta)
			A_i[4,4] = self.A44(p_vx,p_vy,p_phi,p_delta)
			A_i[4,5] = self.A45(p_vx,p_vy,p_phi,p_delta)
			A_i[5,4] = self.A54(p_vx,p_vy,p_phi,p_delta)
			A_i[5,5] = self.A55(p_vx,p_vy,p_phi,p_delta)

			B_i = ca.DM.zeros(6,2)
		
			B_i[3,1] = 1
			B_i[3,0] = self.B30(p_vx,p_vy,p_phi,p_delta)
			B_i[4,0] = self.B40(p_vx,p_vy,p_phi,p_delta)
			B_i[5,0] = self.B50(p_vx,p_vy,p_phi,p_delta)
			A_i = ca.diag(ca.DM.ones(6)) + A_i*dt
			B_i = B_i * dt
			return A_i @ x0 + B_i @ u0
		else:
			A_i = ca.DM.zeros(6,6)
			A_i[0,3] = ca.cos(p_phi)
			A_i[0,4] = -ca.sin(p_phi)
			A_i[1,3] = ca.sin(p_phi)
			A_i[1,4] = ca.cos(p_phi)
			A_i[2,5] = 1
			A_i[3,3] = self.A33(p_vx,p_vy,p_phi,p_delta)
			A_i[3,4] = self.A34(p_vx,p_vy,p_phi,p_delta)
			A_i[3,5] = self.A35(p_vx,p_vy,p_phi,p_delta)
			A_i[4,4] = self.A44(p_vx,p_vy,p_phi,p_delta)
			A_i[4,5] = self.A45(p_vx,p_vy,p_phi,p_delta)
			A_i[5,4] = self.A54(p_vx,p_vy,p_phi,p_delta)
			A_i[5,5] = self.A55(p_vx,p_vy,p_phi,p_delta)

			B_i = ca.DM.zeros(6,2)
			B_i[3,1] = self.B31(p_vx,p_vy,p_phi,p_delta)
			B_i[3,0] = self.B30(p_vx,p_vy,p_phi,p_delta)
			B_i[4,0] = self.B40(p_vx,p_vy,p_phi,p_delta)
			B_i[5,0] = self.B50(p_vx,p_vy,p_phi,p_delta)

			C_i = ca.DM.zeros(6,1)
			C_i[3,0] = -self.Cr0/self.mass
			A_i = ca.diag(ca.DM.ones(6)) + A_i*dt
			B_i = B_i * dt
			C_i = C_i * dt
			return A_i @ x0 + B_i @ u0 + C_i
	
	def LPV_states(self,x0,u,p_vx,p_vy,p_phi,p_delta,dt):
		x0_ = x0
		horizon = u.shape[1]
		x_sim = np.zeros((self.n_states,horizon+1))
		x_sim[:,0] = x0_
		u0 = u[:,0]

		for i in range(horizon):
			ui = u[:,i]
			xi = self.LPV_next_state(x0_,ui,p_vx[0,i],p_vy[0,i],p_phi[0,i],p_delta[0,i],dt)
			xi = np.array(xi).squeeze()
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

	def A33(self,p_vx,p_vy,p_phi,p_delta):
		return -self.Cd*p_vx/self.mass

	def A34(self,p_vx,p_vy,p_phi,p_delta):
		betaf = self.betaf()
		return betaf*ca.sin(p_delta)/p_vx

	def A35(self,p_vx,p_vy,p_phi,p_delta):
		betaf = self.betaf()
		return (betaf*ca.sin(p_delta)*self.lf/p_vx) + p_vy

	def A44(self,p_vx,p_vy,p_phi,p_delta):
		betaf = self.betaf()
		betar = self.betar()
		return (-betar/p_vx) - (betaf*ca.cos(p_delta)/p_vx)

	def A45(self,p_vx,p_vy,p_phi,p_delta):
		betaf = self.betaf()
		betar = self.betar()
		return (betar* self.lr /p_vx) - (betaf * self.lf * ca.cos(p_delta)/p_vx) - p_vx


	def A54(self,p_vx,p_vy,p_phi,p_delta):
		gammaf = self.gammaf()
		gammar = self.gammar()
		return (1/p_vx) * (gammar - gammaf*ca.cos(p_delta))

	def A55(self,p_vx,p_vy,p_phi,p_delta):
		gammaf = self.gammaf()
		gammar = self.gammar()
		return (-1/p_vx) * (gammaf * self.lf * ca.cos(p_delta) + gammar * self.lr)

	def B30(self,p_vx,p_vy,p_phi,p_delta):
		betaf = self.betaf()
		return -betaf * ca.sin(p_delta)

	def B31(self,p_vx,p_vy,p_phi,p_delta):
		return (self.Cm1-self.Cm2*p_vx)/self.mass

	def B40(self,p_vx,p_vy,p_phi,p_delta):
		betaf = self.betaf()
		return betaf * ca.cos(p_delta)

	def B50(self,p_vx,p_vy,p_phi,p_delta):
		gammaf = self.gammaf()
		return gammaf*ca.cos(p_delta)