
import casadi as ca
import numpy as np

from casadi import DM
import sys
import traceback
import os
import time as tm

from drcclpvmpc_ros2.mpc.model_noise import Model_noise
from drcclpvmpc_ros2.mpc.model_noisevx import Model_noiseVx
from drcclpvmpc_ros2.obstacles.obstacle_shape import Rectangle_obs
from drcclpvmpc_ros2.racecar_path.racecar_path import RacecarPath

class STM_DRCCLPVMPC:
    def __init__(self,path_ptr:RacecarPath,
                 current_state,dt,horizon,
                 track_with,fix_noise,params) -> None:
        self.initialize = True
        self.init_vxy = True

        self.track_width_ = track_with

        script = os.path.dirname(__file__)

        self.output_log = os.path.join(script,'output','DynamicDRO_solver_output.txt')
        # params = ORCA(control=control_)
        
        
        """	specify model params here
		"""
        self.lf = params['lf']
        self.lr = params['lr']
        self.vehicle_length = self.lf + self.lr
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
        
        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Cr0 = params['Cr0']
        self.Cd = params['Cd']
        
        self.approx = params['approx']
        
        if self.approx:
            self.n_states = 5 # x, y, phi, vy, omega
        else:
            self.n_states = 6 # x, y, phi, vx, vy, omega
        self.n_inputs = 2 # steer, vx
        
        self.g = params['g'] # gravity acc

        self.max_vx = params['max_vx']
        self.min_vx = params['min_vx']
        
        self.max_vy = params['max_vy']
        self.min_vy = params['min_vy']

        self.max_omega = params['max_omega']
        self.min_omega = params['min_omega']

        self.max_steer = params['max_steer']
        self.min_steer = params['min_steer']
        
        self.max_acc = params['max_acc']
        self.min_acc = params['min_acc']

        self.path_ptr_ = path_ptr
        self.obs_map = None

        self.dt = dt

        self.horizon = horizon
        self.s_max = self.path_ptr_.get_max_length()

        ############################### initialize reference state#####################################
        self.ref_xy = []
        self.v_arr = self.max_vx*ca.DM.ones(self.horizon+1,1)
        getxy = self.get_ref_xy(current_state)
        self.current_state = current_state # x, y, phi, vx, vy, omega

        # now the ref_pre_x, ref_pre_y, ref_pre_phi are already setup
        # now the reference_x, reference_y, reference_phi are already setup

        # unwrap phi for continuous problem
        self.reference_phi = np.array(self.reference_phi)
        self.reference_phi = np.unwrap(self.reference_phi)
        self.reference_phi = ca.DM(self.reference_phi)

        self.obs_cons_atau = None
        self.obs_cons_btau = None

        self.obs_cons_an = None
        self.obs_cons_bn = None

        self.obs_cons_s0 = None
        self.obs_cons_s1 = None
        self.obs_cons_tau0 = None
        self.obs_cons_tau1 = None

        self.get_ref_xyk()
        self.ref_pre_phi = self.reference_phi

        ############################ initialize epsilon_alpha, beta #################################

        self.zeta = 0.08 # sigma
        self.epsilon = 0.10 # episilon

        ################################## initialize schedualing params####################################
        if self.approx:
            self.p_vx = self.v_arr[1:,0].T
            self.p_phi = self.ref_pre_phi[0,1:]
            
            self.past_pvx = self.p_vx
            self.past_pphi = self.p_phi
        else:
            self.p_vx = self.v_arr[1:,0].T
            self.p_vy = ca.DM.zeros(1,self.horizon)
            self.p_phi = self.ref_pre_phi[0,1:]
            self.p_delta = ca.DM.zeros(1,self.horizon)

            self.past_pvx = self.p_vx
            self.past_pvy = self.p_vy
            self.past_pphi = self.p_phi
            self.past_pdelta = self.p_delta

        ################################# initialize noise matrix ###########################################
        if self.approx:
            self.model_noise = Model_noiseVx(self.horizon,fix_noise)
        else:
            self.model_noise = Model_noise(self.horizon,fix_noise)
        self.noise_arr = self.model_noise.get_noise_arr()
        
        self.save_fixed_noise()

        self.reach_end = False

        self.start = tm.time()
        self.end = tm.time()

        self.tau0 = 0

        self.sol_gamma = 0




    def get_Updated_local_path(self,current_state,obs_map:Rectangle_obs,
                               side_avoid,safe_multiplier,usedro=True,print_=False):
        """
        return optimized_result, optimized_state, optimized_control
        """
        self.obs_map = obs_map
        ############################### Construct obstacle ###########################################
        if self.obs_map is not None:
            obs_end_ls = self.obs_map.end_ls[0]
            end_pts = np.array(obs_end_ls).transpose()
            self.obs_cons_atau = 0
            self.obs_cons_btau = 0

            self.obs_cons_an = 0
            self.obs_cons_bn = 0
                
            ab_tau = self.path_ptr_.xy_to_tau(end_pts)
            if ab_tau[0,0] < ab_tau[0,1]:
                ind = 0
                indb = 1
            else:
                ind = 1
                indb = 0
            # print("min ab :",ind)
            tau_a = ca.mmin(ab_tau)
            tau_b = ca.mmax(ab_tau)
            self.obs_cons_atau = tau_a
            self.obs_cons_btau = tau_b

            an = self.path_ptr_.f_xy_to_taun(ca.DM(end_pts[:,ind]),tau_a)
            bn = self.path_ptr_.f_xy_to_taun(ca.DM(end_pts[:,indb]),tau_b)

            self.obs_cons_an = an
            self.obs_cons_bn = bn

            self.obs_cons_sa = self.path_ptr_.tau_to_s_lookup(self.obs_cons_atau)
            self.obs_cons_sb = self.path_ptr_.tau_to_s_lookup(self.obs_cons_btau)
            
            obs_safe_dis = self.max_vx * safe_multiplier * self.horizon * self.dt
            obs_safe_disb = self.max_vx * safe_multiplier * self.horizon * self.dt
            
            obs_safe_dis = min(obs_safe_dis,float(self.obs_cons_sa)-1.0)
            obs_safe_disb = min(obs_safe_disb,float(self.s_max - self.obs_cons_sb-1.0))

            self.obs_cons_phi_fre = ca.atan2(self.obs_cons_an,obs_safe_dis)
            self.obs_cons_phi_freb = ca.atan2(self.obs_cons_bn,obs_safe_disb)

            self.obs_cons_s0 = self.obs_cons_sa - obs_safe_dis
            self.obs_cons_s1 = self.obs_cons_sb + obs_safe_disb

            self.obs_cons_tau0 = self.path_ptr_.s_to_tau_lookup(self.obs_cons_s0)
            self.obs_cons_tau1 = self.path_ptr_.s_to_tau_lookup(self.obs_cons_s1)
        else:
            self.obs_cons_atau = None
            self.obs_cons_btau = None

            self.obs_cons_an = None
            self.obs_cons_bn = None

            self.obs_cons_s0 = None
            self.obs_cons_s1 = None
            self.obs_cons_tau0 = None
            self.obs_cons_tau1 = None

        ############################### initialize weight matrix ###########################################

        self.drcc_weight = ca.DM.zeros(1,self.horizon)
        # self.drcc_weight[0,:] = -0.01*ca.linspace(0,1,self.horizon).T
        self.drcc_weight = -5.0*ca.DM.ones(1,self.horizon)
        
        self.P = ca.DM.zeros(self.n_states,self.horizon)
        ############## this used for drcc #################
        if usedro:
            # self.P[0,-1] = 20 # sepecify x
            # self.P[0,-1] = 20 # sepecify y
            self.P[0,:] = 4*ca.DM.ones(1,self.horizon).T # sepecify x
            self.P[1,:] = 4*ca.DM.ones(1,self.horizon).T # sepecify y
            # self.P[2,:] = 0.0*ca.linspace(0,1,self.horizon).T # sepecify phi
            # self.P[3,:] = 0.0*ca.DM.ones(1,self.horizon).T # sepecify ddphi
            # self.P[4,:] = 0*ca.DM.ones(1,self.horizon).T # sepecify ddelta
            
            self.Q = ca.DM.zeros(self.n_inputs,self.horizon)
            self.Q[0,:] = 0.3*ca.DM.ones(1,self.horizon)
        else:
            self.P[0,-1] = 20 # sepecify x
            self.P[0,-1] = 20 # sepecify y
            # self.P[0,:] = ca.linspace(1,5,self.horizon).T # sepecify x
            # self.P[1,:] = ca.linspace(1,5,self.horizon).T # sepecify y
            # self.P[0,:] = 1*ca.DM.ones(1,self.horizon).T # sepecify x
            # self.P[1,:] = 1*ca.DM.ones(1,self.horizon).T # sepecify y
            # self.P[2,:] = 0.0*ca.linspace(0,1,self.horizon).T # sepecify phi
            # self.P[3,:] = 0.0*ca.DM.ones(1,self.horizon).T # sepecify ddphi
            # self.P[4,:] = 0*ca.DM.ones(1,self.horizon).T # sepecify ddelta
            
            self.Q = ca.DM.zeros(2,self.horizon)
            self.Q[0,:] = 0.8*ca.DM.ones(1,self.horizon) # specify steering

        optimized_path = self.make_plan(current_state,side_avoid,usedrcc=usedro,print_=print_)
        self.end = tm.time()

        if optimized_path[0]:
            print("Solve Success, time: {:.8f}".format(self.end-self.start))

            if not self.approx:
                op_z0 = np.array(optimized_path[1][0])
                op_delta = np.array(optimized_path[1][1][0,:]).squeeze().tolist()
                op_throttle = np.array(optimized_path[1][1][1,:]).squeeze().tolist()

                ############################## update schedualing params ##########################
                self.past_pvx = self.p_vx
                self.past_pvy = self.p_vy
                self.past_pphi = self.p_phi
                self.past_pdelta = self.p_delta
                
                self.p_delta = ca.DM(op_delta).T

                return optimized_path[0],DM(op_z0),tuple((DM(op_delta),DM(op_throttle)))
            else:
                op_z0 = np.array(optimized_path[1][0])
                op_delta = np.array(optimized_path[1][1][0,:]).squeeze().tolist()
                op_vx = np.array(optimized_path[1][1][1,:]).squeeze().tolist()
                
                ############################## update schedualing params ##########################

                self.past_pvx = self.p_vx
                self.past_pphi = self.p_phi
                self.p_vx = ca.DM(op_vx).T

                return optimized_path[0],DM(op_z0),tuple((DM(op_delta),DM(op_vx)))
        else:
            if self.reach_end:
                print("################################")
            else:
                print("optimization fails")
            return optimized_path[0],DM(),tuple((DM(),DM()))


    def make_plan(self,current_state,side_avoid,usedrcc=True,print_=False):
        
        self.sol_gamma = 0

        casadi_option = {"print_time":print_}

        if self.approx:
            self.current_state = ca.DM([current_state[0],current_state[1],
            current_state[2],current_state[3], current_state[4]]) 
        else:
            self.current_state = ca.DM([current_state[0],current_state[1],
                        current_state[2],current_state[3],
                        current_state[4],current_state[5]]) 

        ############################### get reference path #################################
        getxy = self.get_ref_xy(current_state)
        self.tau0 = self.tau_arr[0,0]
        if not getxy:
            print("Reached end")
            self.reach_end = True
            return tuple((False,tuple((DM(),DM()))))
        # now self.reference_xy, self.reference_x, self.reference_y, self.reference_phi is updated

        # unwrap the reference phi for continuous concern
        # self.reference_phi = np.array(self.reference_phi)
        # self.reference_phi = np.unwrap(self.reference_phi)
        # self.reference_phi = ca.DM(self.reference_phi)

        if getxy is False:
            return False
        self.get_ref_xyk()
        # update previous reference phi, x, y
        if not self.approx:
            self.reference_vx = self.v_arr[:self.horizon+1,0].T
        # self.reference_vy = (ykpre - self.ykpre)/self.dt
        self.reference_vy = ca.DM.zeros(1,self.horizon+1)

        ################### calculate boundary function ax + by = c ###########################

        a11, b11, c11, a12, b12, c12, a13, b13, c13, a14, b14, c14 = self.get_corridor_func()

        ########################################### get rectangle tangent function ax + by = c #######################

        # ref_xy = np.array(self.ref_xy).transpose().tolist()
        # print("ref xy :",ref_xy)

        a3 = np.ones((1,self.horizon+1)) # a1,b1,c1,a2,b2,c2 * (horizon+1)
        b3 = np.ones((1,self.horizon+1)) # a1,b1,c1,a2,b2,c2 * (horizon+1)
        c3 = np.ones((1,self.horizon+1)) # a1,b1,c1,a2,b2,c2 * (horizon+1)

        a3 *= 1
        b3 *= 1
        c3 *= -1e10
        
        if self.obs_map is not None:
            for i in range(self.horizon,-1,-1):
                if self.tau_arr[0,i] <= self.obs_cons_tau0:
                    continue
                elif self.tau_arr[0,i] <= self.obs_cons_atau:
                    collid_n = self.obs_cons_an * (self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,i]) - self.obs_cons_s0)/(self.obs_cons_sa-self.obs_cons_s0)
                    shift_n = collid_n + side_avoid*0.1
                    s_a3,s_b3,s_c3 = self.get_safe_line_equation(self.tau_arr[0,i],collid_n,shift_n,self.obs_cons_phi_fre)
                    # self.update_safe_reference_track(i,collid_n,self.obs_cons_phi_fre)
                elif self.tau_arr[0,i] <= self.obs_cons_btau:
                    collid_n = self.obs_cons_bn + (self.obs_cons_an-self.obs_cons_bn)*(self.obs_cons_sb-self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,i]))/(self.obs_cons_sb-self.obs_cons_sa)
                    shift_n = collid_n + side_avoid*0.1
                    s_a3,s_b3,s_c3 = self.get_safe_line_equation(self.tau_arr[0,i],collid_n,shift_n,0)
                    # self.update_safe_reference_track(i,collid_n,0)
                elif self.tau_arr[0,i] >= self.obs_cons_btau and self.tau_arr[0,i] <= self.obs_cons_tau1:
                    collid_n = self.obs_cons_bn * (-self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,i]) + self.obs_cons_s1)/(self.obs_cons_s1-self.obs_cons_sb)
                    shift_n = collid_n + side_avoid*0.1
                    s_a3,s_b3,s_c3 = self.get_safe_line_equation(self.tau_arr[0,i],collid_n,shift_n,-self.obs_cons_phi_freb)
                    # self.update_safe_reference_track(i,collid_n,self.obs_cons_phi_freb)
                else:
                    continue


                a3[0,i] = s_a3
                b3[0,i] = s_b3
                c3[0,i] = s_c3
        
            collid_n = 0
            if (self.tau_arr[0,0] >= self.obs_cons_tau0) and (self.tau_arr[0,0] <= self.obs_cons_atau):
                collid_n = (self.obs_cons_an) * (self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,-1]) - self.obs_cons_s0)/(self.obs_cons_sa-self.obs_cons_s0)
                
            elif (self.tau_arr[0,0] >= self.obs_cons_atau) and (self.tau_arr[0,0] <= self.obs_cons_btau):
                if(self.tau_arr[0,-1] <= self.obs_cons_btau):
                    # collid_n = self.obs_cons_bn
                    collid_n = self.obs_cons_bn + (self.obs_cons_an-self.obs_cons_bn)*(self.obs_cons_sb-self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,-1]))/(self.obs_cons_sb-self.obs_cons_sa)
                else:
                    # collid_n = self.obs_cons_bn
                    collid_n = self.obs_cons_bn + (self.obs_cons_an-self.obs_cons_bn)*(self.obs_cons_sb-self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,-1]))/(self.obs_cons_sb-self.obs_cons_sa)

            elif self.tau_arr[0,0] >= (self.obs_cons_btau) and self.tau_arr[0,-1] <= self.obs_cons_tau1:
                collid_n = (self.obs_cons_bn) * (-self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,-1]) + self.obs_cons_s1)/(self.obs_cons_s1-self.obs_cons_sb)
                
            if collid_n != 0:
                phi_fre = ca.atan2(self.n_arr[0,0] - collid_n, self.s_arr[0,-1]-self.s_arr[0,0])
                collid_n = ca.linspace(self.n_arr[0,0],collid_n,self.horizon+1).T
                
                new_xy = self.path_ptr_.f_taun_to_xy(self.tau_arr,collid_n)
                self.reference_x = new_xy[0,:]
                self.reference_y = new_xy[1,:]
                self.reference_phi += phi_fre
                
                # for i in range(0,self.horizon+1):
                #     self.update_safe_reference_track(i,collid_n[0,i],phi_fre)
        ############################### unwrap the reference phi #################################
        # if self.reference_phi[0,0] - self.ref_pre_phi[0,1] >= 2*ca.pi-0.5:
        #     self.reference_phi -= 2*ca.pi
        # elif self.reference_phi[0,0] - self.ref_pre_phi[0,1] <= -2*ca.pi+0.5:
        #     self.reference_phi += 2*ca.pi
        
        self.ref_pre_phi = self.reference_phi
        self.ref_pre_x = self.reference_x
        self.ref_pre_y = self.reference_y

        ################################################ get schedual matrix ######################################################

        # alphaf = self.p_delta - (self.p_vy+self.lf*self.p_omega)/self.p_vx
        # alphar = (self.lr*self.p_omega-self.p_vy)/self.p_vx
        if self.approx:
            A33 = self.A33()
            A34 = self.A34()
            A43 = self.A43()
            A44 = self.A44()

            B30 = self.B30()
            B40 = self.B40()
        else:
            A33 = self.A33()
            A34 = self.A34()
            A35 = self.A35()
            A44 = self.A44()
            A45 = self.A45()
            A54 = self.A54()
            A55 = self.A55()

            B30 = self.B30()
            B31 = self.B31()
            B40 = self.B40()
            B50 = self.B50()
        ##################################### define objective function ################################

        opti = ca.Opti('conic')

        z = opti.variable(self.n_states + self.n_inputs*self.horizon,1)

        ################### calculate the list of A and B #########################
        if self.approx:
            A_ls = []
            B_ls = []
        else:
            A_ls = []
            B_ls = []
            C_ls = []
        
            C_k = ca.DM.zeros(self.n_states,1)
            C_k[3,0] = -self.Cr0/self.mass
            C_k = C_k * self.dt

        for i in range(self.horizon):
            if self.approx:
                A_i = ca.DM.zeros(self.n_states,self.n_states)
                A_i[0,3] = -ca.sin(self.p_phi[0,i])
                A_i[1,3] = ca.cos(self.p_phi[0,i])
                A_i[2,4] = 1
                A_i[3,3] = A33[0,i]
                A_i[3,4] = A34[0,i]
                A_i[4,3] = A43[0,i]
                A_i[4,4] = A44[0,i]

                B_i = ca.DM.zeros(self.n_states,self.n_inputs)
                B_i[0,1] = ca.cos(self.p_phi[0,i])
                B_i[1,1] = ca.sin(self.p_phi[0,i])
                B_i[3,0] = B30
                B_i[4,0] = B40

                A_i = ca.diag(ca.DM.ones(self.n_states)) + A_i*self.dt
                B_i = B_i * self.dt
                
                A_ls.append(A_i) # A0, A1, ...
                B_ls.append(B_i) # B0, B1, ...
            else:
                A_i = ca.DM.zeros(self.n_states,self.n_states)
                A_i[0,3] = ca.cos(self.p_phi[0,i])
                A_i[0,4] = -ca.sin(self.p_phi[0,i])
                A_i[1,3] = ca.sin(self.p_phi[0,i])
                A_i[1,4] = ca.cos(self.p_phi[0,i])
                A_i[2,5] = 1
                A_i[3,3] = A33[0,i]
                A_i[3,4] = A34[0,i]
                A_i[3,5] = A35[0,i]
                A_i[4,4] = A44[0,i]
                A_i[4,5] = A45[0,i]
                A_i[5,4] = A54[0,i]
                A_i[5,5] = A55[0,i]

                B_i = ca.DM.zeros(self.n_states,self.n_inputs)
                B_i[3,1] = B31[0,i]
                B_i[3,0] = B30[0,i]
                B_i[4,0] = B40[0,i]
                B_i[5,0] = B50[0,i]

                A_i = ca.diag(ca.DM.ones(self.n_states)) + A_i*self.dt
                B_i = B_i * self.dt

                A_ls.append(A_i) # A0, A1, ...
                B_ls.append(B_i) # B0, B1, ...
                C_ls.append(C_k)

        E_ls = []
        for i in range(self.horizon): # this for the steps
            A_init = ca.DM.eye(self.n_states)
            E_i = []
            for j in range(i,-1,-1): # inside each steps
                A_init = A_init@A_ls[j]
                E_i.append(A_init)
            E_i.reverse()
            E_ls.append(E_i) # [[E00],[E01,E11],[E02,E12,E22],...]

        ############################ Construct the Ln ###############################

        Ln = ca.DM.zeros(self.n_states*self.horizon,self.n_states+self.n_inputs*self.horizon)

        for i in range(self.horizon):
            Ln[i*self.n_states:(i+1)*self.n_states,0:self.n_states] = E_ls[i][0]
            for j in range(1,len(E_ls[i])):
                Ln[i*self.n_states:(i+1)*self.n_states,self.n_states+(j-1)*self.n_inputs:self.n_states+j*self.n_inputs] = E_ls[i][j] @ B_ls[j-1]
            Ln[i*self.n_states:(i+1)*self.n_states,self.n_states+j*self.n_inputs:self.n_states+(j+1)*self.n_inputs] = B_ls[j]
        
        # print("Ln ls shape:",len(Ln_ls))

        ############################### Construct the Hn ###############################

        Hn = ca.DM.zeros((self.n_states*self.horizon,self.n_states*self.horizon))


        for i in range(self.horizon):
            for j in range(0,i):
                Hn[i*self.n_states:(i+1)*self.n_states,j*self.n_states:(j+1)*self.n_states] = E_ls[i][j+1]
            Hn[i*self.n_states:(i+1)*self.n_states,i*self.n_states:(i+1)*self.n_states] = np.eye(self.n_states)

        # print("Hn ls :",Hn_ls)
        if not self.approx:
            C_N = ca.DM.zeros(self.n_states*self.horizon,1)
            for i in range(self.horizon):
                C_N[i*self.n_states:(i+1)*self.n_states,0] = C_ls[i]
        ############################ update objective function ############################
        if self.approx:
            ref_state = np.array(ca.vertcat(self.reference_x,self.reference_y,self.reference_phi,
                    self.reference_vy,self.reference_omega))
        else:
            ref_state = np.array(ca.vertcat(self.reference_x,self.reference_y,self.reference_phi,
                        self.reference_vx,self.reference_vy,self.reference_omega))
        
        self.ref_xy = ca.DM(ref_state[:3,:])

        ref_state = np.reshape(ref_state[:,1:],(-1,1),order='F')
        ref_state = ca.DM(ref_state)

        

        Dn = ca.DM.zeros(self.n_states*self.horizon,1)
        for i in range(self.horizon):
            Dn[i*self.n_states,0] = self.P[0,i]
            Dn[i*self.n_states+1,0] = self.P[1,i]
            Dn[i*self.n_states+2,0] = self.P[2,i]

        Fn = ca.DM.zeros(self.n_states+self.n_inputs*self.horizon,1)
        for i in range(self.horizon):
            Fn[self.n_states+self.n_inputs*i,0] = self.Q[0,i]
            Fn[self.n_states+self.n_inputs*i+1,0] = self.Q[1,i]

        # xn_obj = ca.dot(Dn*(Ln@z-ref_state),Dn*(Ln@z-ref_state)) + ca.dot(Fn*z,Fn*z)
        if self.approx:
            xn_obj = ca.dot(Dn*(Ln@z-ref_state),Dn*(Ln@z-ref_state)) + ca.dot(Fn*z,Fn*z)
        else:

            xn_obj = ca.dot(Dn*(Ln@z+Hn@C_N-ref_state),Dn*(Ln@z+Hn@C_N-ref_state)) + ca.dot(Fn*z,Fn*z)
            

        use_chance_covx = False # specify whether or not to use CvaR to convert the state bounds of vx
        use_chance_box = False
        CvaR_corrid = usedrcc

        ############################## calculate d_k_vx, f_k_vx, define assist variable nu, q, gamma and add to the constraints ###################################

        cvar_cost = 0

        # if use_chance_covx:
        #     Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
        #     for i in range(self.horizon):
        #         Dn[i,i*self.n_states+3] = 1
        #     for i in range(self.horizon):
        #         dn = ca.DM.zeros(1,self.horizon)

        #         dn[0,i] = 1

        #         q = opti.variable(self.noise_arr.shape[0],1)
        #         nu = opti.variable(1,1)
        #         gamma = opti.variable(1,1)

        #         q2 = opti.variable(self.noise_arr.shape[0],1)
        #         nu2 = opti.variable(1,1)
        #         gamma2 = opti.variable(1,1)
        #         # qm = opti.variable(1,1) # min vx

        #         dDLnZn = dn@Dn@Ln@z
                
        #         if not self.approx:
        #             dDLnZn += dn@Dn@Hn@C_N
        #         dDHn = dn@Dn@Hn

        #         casj_samples = ca.DM.zeros(self.n_states*self.horizon,self.noise_arr.shape[0])
        #         for j in range(self.noise_arr.shape[0]):
        #             casj = ca.DM(self.noise_arr[j])
        #             casj_samples[:,j] = casj
        #         dDHn_casj = dDHn@casj_samples

        #         opti.subject_to(self.zeta*nu - self.epsilon*gamma - ca.sum1(q)/self.noise_arr.shape[0] >= 0)
        #         opti.subject_to(q >= 0)
        #         opti.subject_to(nu <= 1)
        #         opti.subject_to(gamma <= 0)

        #         opti.subject_to(self.zeta*nu2 - self.epsilon*gamma2 - ca.sum1(q2)/self.noise_arr.shape[0] >= 0)
        #         opti.subject_to(q2 >= 0)
        #         opti.subject_to(nu2 <= 1)
        #         opti.subject_to(gamma2 <= 0)

        #         for j in range(self.noise_arr.shape[0]):
        #             dq = ca.DM.zeros(1,self.noise_arr.shape[0])
        #             dq[0,j] = 1
        #             opti.subject_to(dq@q + gamma >= -self.max_vx + dDLnZn + dDHn_casj[0,j])

        #             opti.subject_to(dq@q2 + gamma2 >= self.min_vx - dDLnZn - dDHn_casj[0,j])

        #             cvar_cost += (dq@q)/ (1 - self.epsilon)
        #             cvar_cost += (dq@q2)/ (1 - self.epsilon)



        # ########################### below gives the boundary where no chance constraints used ##################
        # else:
        #     Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
        #     for i in range(self.horizon):
        #         Dn[i,i*self.n_states+3] = 1

        #     DLnZn = Dn@Ln@z
        #     if not self.approx:
        #         DLnZn += Dn@Hn@C_N

        #     opti.subject_to(opti.bounded(self.min_vx,DLnZn,self.max_vx))

        # ##################################### corridor constraints functions a11, b11, c11 ####################################
        
        Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
        for i in range(self.horizon):
            Dn[i,i*self.n_states] = a11[i+1,0]
            Dn[i,i*self.n_states+1] = b11[i+1,0]

        # print("c3 :",c3)

        for i in range(self.horizon):
            dn = ca.DM.zeros(1,self.horizon)

            dn[0,i] = 1

            # q = opti.variable(1,1)

            dDLnZn = dn@Dn@Ln@z
            if not self.approx:
                dDLnZn += dn@Dn@Hn@C_N
            # dDHn = dn@Dn@Hn

            opti.subject_to(dDLnZn - c11[i+1] <= 0)

        # ##################################### corridor constraints functions a12, b12, c12 ####################################
        Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
        for i in range(self.horizon):
            Dn[i,i*self.n_states] = a12[i+1,0]
            Dn[i,i*self.n_states+1] = b12[i+1,0]

        # print("c3 :",c3)

        for i in range(self.horizon):
            dn = ca.DM.zeros(1,self.horizon)

            dn[0,i] = 1

            # q = opti.variable(1,1)

            dDLnZn = dn@Dn@Ln@z
            if not self.approx:
                dDLnZn += dn@Dn@Hn@C_N
            # dDHn = dn@Dn@Hn

            opti.subject_to(dDLnZn - c12[i+1] <= 0)

        # ##################################### corridor constraints functions a13, b13, c13 ####################################
        Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
        for i in range(self.horizon):
            Dn[i,i*self.n_states] = a13[i+1,0]
            Dn[i,i*self.n_states+1] = b13[i+1,0]

        # print("c3 :",c3)

        for i in range(self.horizon):
            dn = ca.DM.zeros(1,self.horizon)

            dn[0,i] = 1

            # q = opti.variable(1,1)

            dDLnZn = dn@Dn@Ln@z
            if not self.approx:
                dDLnZn += dn@Dn@Hn@C_N
            # dDHn = dn@Dn@Hn

            opti.subject_to(dDLnZn - c13[i+1] <= 0)

        # ##################################### corridor constraints functions a14, b14, c14 ####################################
        Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
        for i in range(self.horizon):
            Dn[i,i*self.n_states] = a14[i+1,0]
            Dn[i,i*self.n_states+1] = b14[i+1,0]

        # print("c3 :",c3)

        for i in range(self.horizon):
            dn = ca.DM.zeros(1,self.horizon)

            dn[0,i] = 1

            # q = opti.variable(1,1)

            dDLnZn = dn@Dn@Ln@z
            if not self.approx:
                dDLnZn += dn@Dn@Hn@C_N
            # dDHn = dn@Dn@Hn

            opti.subject_to(dDLnZn - c14[i+1] <= 0)

        # ##################################### obstacles constraints functions a3, b3, c3 ####################################
        if CvaR_corrid:
            Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
            for i in range(self.horizon):
                Dn[i,i*self.n_states] = a3[0,i+1]
                Dn[i,i*self.n_states+1] = b3[0,i+1]

            gamma_next = 0
            for i in range(self.horizon):
                if c3[0,i+1] == -1e10:
                    continue
                dn = ca.DM.zeros(1,self.horizon)

                dn[0,i] = 1

                Mj = 1000

                q = opti.variable(self.noise_arr.shape[0],1)
                gamma = opti.variable(1,1)

                if i == 0:
                    gamma_next = gamma

                dDLnZn = dn@Dn@Ln@z
                if not self.approx:
                    dDLnZn += dn@Dn@Hn@C_N
                dDHn = dn@Dn@Hn

                casj_samples = ca.DM.zeros(self.n_states*self.horizon,self.noise_arr.shape[0])
                for j in range(self.noise_arr.shape[0]):
                    casj = ca.DM(self.noise_arr[j])
                    casj_samples[:,j] = casj
                dDHn_casj = dDHn@casj_samples

                opti.subject_to(gamma >= 0)
                opti.subject_to(self.zeta - self.epsilon*gamma - ca.sum1(q)/self.noise_arr.shape[0] <= 0)

                for j in range(self.noise_arr.shape[0]):
                    dq = ca.DM.zeros(1,self.noise_arr.shape[0])
                    dq[0,j] = 1
                    sj = opti.variable(1,1)
                    yj = opti.variable(1,1)
                    opti.subject_to(dq@q + gamma <= sj)
                    opti.subject_to(dDLnZn + dDHn_casj[0,j] - c3[0,i+1] + Mj*(1 - yj) - sj >= 0)
                    opti.subject_to(sj <= Mj*yj)
                    opti.subject_to(dq@q <= 0)
                    opti.subject_to(sj >= 0)
                    opti.subject_to(opti.bounded(0,yj,1))

                    cvar_cost += self.drcc_weight[0,i]*(dq@q + gamma)


        ################ back to the hard constraint ####################
        else:
            Dn = ca.DM.zeros(self.horizon,self.n_states*self.horizon)
            for i in range(self.horizon):
                Dn[i,i*self.n_states] = -a3[0,i+1]
                Dn[i,i*self.n_states+1] = -b3[0,i+1]

            # print("c3 :",c3)

            for i in range(self.horizon):
                dn = ca.DM.zeros(1,self.horizon)

                dn[0,i] = 1

                # q = opti.variable(1,1)

                dDLnZn = dn@Dn@Ln@z
                if not self.approx:
                    dDLnZn += dn@Dn@Hn@C_N
                # dDHn = dn@Dn@Hn

                opti.subject_to(dDLnZn + c3[0,i+1] <= 0)

        

        

        # ######################################## specify constraints #########################################
        A = ca.DM.zeros(self.n_states, self.n_states+self.n_inputs*self.horizon)
        b = ca.DM.zeros(self.n_states,1)
        for i in range(self.n_states):
            A[i,i] = 1
            b[i,0] = self.current_state[i,0]

        # A = A.sparsity()
        opti.subject_to(A@z - b == 0)

        ####################################### control boundary ########################################
        Adu = ca.DM.zeros(self.horizon, self.n_states+self.n_inputs*self.horizon)
        Aau = ca.DM.zeros(self.horizon, self.n_states+self.n_inputs*self.horizon)

        for i in range(self.horizon):
            Adu[i,self.n_states+i*self.n_inputs] = 1
            Aau[i,self.n_states+i*self.n_inputs+1] = 1
        
        # Adu = Adu.sparsity()
        # Aau = Aau.sparsity()
        if self.approx:
            opti.subject_to(opti.bounded(self.min_vx, Aau@z, self.max_vx))
        else:
            opti.subject_to(opti.bounded(0.38, Aau@z, 1.0))
        opti.subject_to(opti.bounded(self.min_steer, Adu@z , self.max_steer))

        ###################################### minimize the objective function ##########################################

        opti.minimize(xn_obj + cvar_cost)
        
        #########################################################################################################################################

        opti.solver("proxqp",casadi_option)
        # opti.solver("qpoases",casadi_option)

        output_to_file = True
        self.start = tm.time()

        if output_to_file:

            with open(self.output_log,'w') as output_file:
                stdout_old = sys.stdout
                sys.stdout = output_file

                try:
                    sol = opti.solve()

                    sol_z = sol.value(z)
                    sol_z = sol_z.reshape((-1,1))
                    sol_z0 = sol_z[0:self.n_states,0]
                    z_delta = sol_z[self.n_states::self.n_inputs].T
                    z_acc = sol_z[self.n_states+1::self.n_inputs].T
                    sol_zu = ca.vertcat(z_delta,z_acc)

                    if CvaR_corrid:
                        self.sol_gamma = sol.value(gamma_next)
                    # sol_g = sol.value(gamma)
                    # sol_q = sol.value(q)


                    # sol_qm = sol.value(qimean)
                    # sol_nu = sol.value(nuvxmax)
                    # sol_gamma = sol.value(gammavxmax)
                    # print("sol q:",sol_qm)
                    # print("sol nv:",sol_nu)
                    # print("sol gamma:",sol_gamma)


                    print("sol z0:",sol_z0)
                    print("sol acc:",z_acc)
                    print("sol delta:",z_delta)
                    # print("sol gamma:",sol_g)
                    # print("sol q:",sol_q)



                    return tuple((True,tuple((sol_z0,sol_zu))))
                except:
                    # debug_z = opti.debug.value(z)
                    # debug_z = debug_z.reshape((-1,1))
                    # sol_z0 = debug_z[0:self.n_states,0]
                    # print("debug z0 : ",sol_z0)
                    # z_delta = debug_z[self.n_states::self.n_inputs].T
                    # z_acc = debug_z[self.n_states+1::self.n_inputs].T
                    # sol_zu = ca.vertcat(z_delta,z_acc)
                    # print("debug zu : ",sol_zu)

                    # sol_qm = opti.debug.value(qimean)
                    # sol_nu = opti.debug.value(nuvxmax)
                    # sol_gamma = opti.debug.value(gammavxmax)
                    # print("sol q:",sol_qm)
                    # print("sol nv:",sol_nu)
                    # print("sol gamma:",sol_gamma)

                    print("OPTIMIZED SOLVER FAILED")
                    traceback.print_exc(file=stdout_old)
                    print("Solve optimal problem fails")
                    return tuple((False,tuple((DM(),DM()))))
                finally:
                    sys.stdout = stdout_old
        else:
            try:
                # start = tm.time()
                sol = opti.solve()
                # end = tm.time()
                # print("Solve Success, time: {:.2f}".format(end-start))
                
                sol_z = sol.value(z)
                sol_z = sol_z.reshape((-1,1))
                print("sol z:",sol_z.shape)
                sol_z0 = sol_z[0:self.n_states,0]
                z_delta = sol_z[self.n_states::self.n_inputs].T
                z_acc = sol_z[self.n_states+1::self.n_inputs].T
                sol_zu = ca.vertcat(z_delta,z_acc)


                # sol_qm = sol.value(qimean)
                # sol_nu = sol.value(nuvxmax)
                # sol_gamma = sol.value(gammavxmax)
                # print("sol q:",sol_qm)
                # print("sol nv:",sol_nu)
                # print("sol gamma:",sol_gamma)


                print("sol z0:",sol_z0)
                print("sol u:",sol_zu)

                return tuple((True,tuple((sol_z0,sol_zu))))
            except:
                debug_z = opti.debug.value(z)
                debug_z = debug_z.reshape((-1,1))
                sol_z0 = debug_z[0:self.n_states,0]
                print("debug z0 : ",sol_z0)
                z_delta = debug_z[self.n_states::self.n_inputs].T
                z_acc = debug_z[self.n_states+1::self.n_inputs].T
                sol_zu = ca.vertcat(z_delta,z_acc)
                print("debug zu : ",sol_zu)

                # sol_qm = opti.debug.value(qimean)
                # sol_nu = opti.debug.value(nuvxmax)
                # sol_gamma = opti.debug.value(gammavxmax)
                # print("sol q:",sol_qm)
                # print("sol nv:",sol_nu)
                # print("sol gamma:",sol_gamma)

                print("OPTIMIZED SOLVER FAILED")
                traceback.print_exc(file=stdout_old)
                print("Solve optimal problem fails")
                return tuple((False,tuple((DM(),DM()))))
    

    def get_ref_xy(self,current_state):
        """
        update self.ref_xy based on given current state
        """
        s0 = 0.0
        st = 0.0
        if self.approx:
            x0 = ca.DM([current_state[0],current_state[1],
            current_state[2],current_state[3], current_state[4]]) 
        else:
            x0 = ca.DM([current_state[0],current_state[1],
                        current_state[2],current_state[3],
                        current_state[4],current_state[5]]) 
        # x, y, phi, vx, vy, omega
        tau0 = self.path_ptr_.xy_to_tau(x0[:2])
        # print("tau 0 :",tau0)
        if tau0<0.00001:
            tau0 = 0.1

        s0 = self.path_ptr_.tau_to_s_lookup(tau0)
        
        st = s0 + self.max_vx*self.dt*self.horizon
        
        if st>=self.s_max:
            st = self.s_max - 0.1
            
        if st - s0 < 0.5:
            return False
        
        self.s_arr = ca.linspace(s0,st,self.horizon+1).T

        self.tau_arr = self.path_ptr_.s_to_tau_lookup(self.s_arr)        

        n0 = self.path_ptr_.f_xy_to_taun(x0[:2],tau0)

        self.n_arr = ca.linspace(n0,0,self.horizon+1).T
        
        phi_fre = ca.atan2(n0,st-s0)

        self.ref_xy = self.path_ptr_.f_taun_to_xy(self.tau_arr,self.n_arr)

        track_phi = self.path_ptr_.f_phi(self.tau_arr)
        track_phi[0,0] = x0[2]
        for i in range(1,self.horizon+1):
            delta = ca.arctan2(ca.sin(track_phi[0,i]-track_phi[0,i-1]),ca.cos(track_phi[0,i]-track_phi[0,i-1]))
            track_phi[0,i] = track_phi[0,i-1] + delta
        self.ref_pre_phi = track_phi
        self.reference_phi = track_phi + phi_fre
        self.reference_x = self.ref_xy[0,:]
        self.reference_y = self.ref_xy[1,:]

        return True

    
    def get_abc12(self,up_a,up_b,up_c,low_a,low_b,low_c):
        lhs = (np.array(up_a*self.ref_xy[0,:]+up_b*self.ref_xy[1,:]-up_c)<=0).astype(int)

        a10 = up_a*lhs
        b10 = up_b*lhs
        c10 = up_c*lhs
        a20 = low_a*lhs
        b20 = low_b*lhs
        c20 = low_c*lhs

        lhs = (np.array(up_a*self.ref_xy[0,:]+up_b*self.ref_xy[1,:]-up_c)>0).astype(int)

        a11 = low_a*lhs
        b11 = low_b*lhs
        c11 = low_c*lhs
        a21 = up_a*lhs
        b21 = up_b*lhs
        c21 = up_c*lhs

        a1 = a10 + a11
        b1 = b10 + b11
        c1 = c10 + c11

        a2 = a20 + a21
        b2 = b20 + b21
        c2 = c20 + c21

        return a1,b1,c1,a2,b2,c2
    
    def get_ref_xyk(self):
        """
        calculate reference states x_ref, y_ref, phi_ref, vx_ref, vy_ref, omega_ref, alphaf_ref, alphar_ref
        """
        self.reference_omega = (self.reference_phi-self.ref_pre_phi[0,:self.horizon+1])/self.dt
    
    def betaf(self):
        # Caf = self.fp1*ca.power(alphaf,3)+self.fp2*ca.power(alphaf,2)+self.fp3*alphaf+self.fp4+self.fp5/(alphaf+0.0075)
        # Caf = ca.DM(self.Bf * self.Cf * self.Df)
        return self.Caf/self.mass
    
    def gammaf(self):
        # Caf = self.fp1*ca.power(alphaf,3)+self.fp2*ca.power(alphaf,2)+self.fp3*alphaf+self.fp4+self.fp5/(alphaf+0.0075)
        # Caf = ca.DM(self.Bf * self.Cf * self.Df)

        return self.Caf*self.lf/self.Iz
    
    def betar(self):
        # Car = self.rp1*ca.power(alphar,3)+self.rp2*ca.power(alphar,2)+self.rp3*alphar+self.rp4+self.rp5/(alphar+0.0075)
        # Car = ca.DM(self.Br * self.Cr * self.Dr)
        return self.Car/self.mass
    
    def gammar(self):
        # Car = self.rp1*ca.power(alphar,3)+self.rp2*ca.power(alphar,2)+self.rp3*alphar+self.rp4+self.rp5/(alphar+0.0075)
        # Car = ca.DM(self.Br * self.Cr * self.Dr)

        return self.lr*self.Car/self.Iz
    
    def A33(self):
        betaf = self.betaf()
        betar = self.betar()
        return -self.Cd*self.p_vx/self.mass if not self.approx else -(betaf+betar)/self.p_vx
    
    def A34(self):
        betaf = self.betaf()
        betar = self.betar()
        return betaf*ca.sin(self.p_delta)/self.p_vx if not self.approx else (betar*self.lr-betaf*self.lf)/self.p_vx
    
    def A35(self):
        betaf = self.betaf()
        return (betaf*ca.sin(self.p_delta)*self.lf/self.p_vx) + self.p_vy
    
    def A43(self):
        gammaf = self.gammaf()
        gammar = self.gammar()
        return (gammar-gammaf)/self.p_vx
    
    def A44(self):
        betaf = self.betaf()
        betar = self.betar()
        gammaf = self.gammaf()
        gammar = self.gammar()
        # return -betaf * ca.cos(self.p_delta) * (1/self.p_vx) - betar * (1/self.p_vx)
        return ((-betar/self.p_vx) - (betaf*ca.cos(self.p_delta)/self.p_vx) if 
                not self.approx else -(gammaf*self.lf+gammar*self.lr)/self.p_vx)
    
    def A45(self):
        betaf = self.betaf()
        betar = self.betar()
        # return -self.p_vx - betaf * ca.cos(self.p_delta) * (1/self.p_vx) * self.lf + betar * (1/self.p_vx) * self.lr
        return (betar* self.lr /self.p_vx) - (betaf * self.lf * ca.cos(self.p_delta)/self.p_vx) - self.p_vx
    
    
    def A54(self):
        gammaf = self.gammaf()
        gammar = self.gammar()
        return (1/self.p_vx) * (gammar - gammaf*ca.cos(self.p_delta))
    
    def A55(self):
        gammaf = self.gammaf()
        gammar = self.gammar()
        return (-1/self.p_vx) * (gammaf * self.lf * ca.cos(self.p_delta) + gammar * self.lr)
    
    def B30(self):
        betaf = self.betaf()
        return -betaf * ca.sin(self.p_delta) if not self.approx else betaf
    
    def B31(self):
        return (self.Cm1-self.Cm2*self.p_vx)/self.mass
    
    def B40(self):
        betaf = self.betaf()
        gammaf = self.gammaf()
        return betaf * ca.cos(self.p_delta) if not self.approx else gammaf
    
    def B50(self):
        gammaf = self.gammaf()
        return gammaf*ca.cos(self.p_delta)
    
    def get_reference_path(self):
        return self.ref_xy
    
    def get_reference_phi(self):
        return self.reference_phi
    
    def get_obs_atau(self):
        return self.obs_cons_atau
    
    def get_obs_an(self):
        return self.obs_cons_an
    
    def get_obs_btau(self):
        return self.obs_cons_btau
    
    def get_obs_bn(self):
        return self.obs_cons_bn
    
    def get_obs_tau0(self):
        return self.obs_cons_tau0
    
    def get_obs_tau1(self):
        return self.obs_cons_tau1
    
    def casadi_unwrap(self,op_phi):
        
        op_phi_np = np.unwrap(np.array(op_phi))

        # print("phi diff:",op_phi_np)

        return op_phi_np
    
    def get_safe_line_equation(self,tau0,n0,n1,phi_ref):

        xy1 = self.path_ptr_.f_taun_to_xy(tau0,n0)
        xy2 = self.path_ptr_.f_taun_to_xy(tau0,n1)

        phi = self.path_ptr_.f_phi(tau0)

        phi_cart = phi + phi_ref
        # print("phi cart :",phi_cart)
        x1 = xy1[0,:]
        y1 = xy1[1,:]
        x2 = xy2[0,:]
        y2 = xy2[1,:]
        # Calculate coefficients
        a = ca.sin(phi_cart)
        b = -ca.cos(phi_cart)
        c = ca.sin(phi_cart)*x1 - ca.cos(phi_cart)*y1

        # the line will show as ax + by - c = 0

        test_func_val = a * x2 + b * y2 - c

        test_func_val = np.array(test_func_val)
        # print("test function value:",test_func_val)

        negative_index = np.where(test_func_val < 0)[1]
        # print("negative value :",negative_index)
        
        if negative_index.shape[0] != 0: 
            a[negative_index] = -1*a[negative_index]
            b[negative_index] = -1*b[negative_index]
            c[negative_index] = -1*c[negative_index]

        return a, b, c
    
    def update_safe_reference_track(self, current_idx, new_n, phi_ref):
        current_tau = self.tau_arr[0,current_idx]
        new_xy = self.path_ptr_.f_taun_to_xy(current_tau,new_n)
        # print("new xy :",new_xy)
        self.reference_x[0,current_idx] = new_xy[0,0]
        self.reference_y[0,current_idx] = new_xy[1,0]
        self.reference_phi[0,current_idx] += phi_ref
    
    def get_corridor_func(self):
        taus = self.path_ptr_.xy_to_tau(self.ref_xy)

        ns = ca.DM.zeros(1,taus.columns())
        xy = self.path_ptr_.f_taun_to_xy(taus,ns)

        x0 = xy[0,:] - self.track_width_
        x1 = xy[0,:] + self.track_width_

        y0 = xy[1,:] - self.track_width_
        y1 = xy[1,:] + self.track_width_

        # all should satisfies ax + by - c <= 0

        a11 = -1 * ca.DM.ones(1,x0.shape[1])
        b11 = ca.DM.zeros(1,x0.shape[1])
        c11 = -x0

        a12 = ca.DM.ones(1,x0.shape[1])
        b12 = ca.DM.zeros(1,x0.shape[1])
        c12 = x1

        a13 = ca.DM.zeros(1,x0.shape[1])
        b13 = -1 * ca.DM.ones(1,x0.shape[1])
        c13 = -y0

        a14 = ca.DM.zeros(1,x0.shape[1])
        b14 = ca.DM.ones(1,x0.shape[1])
        c14 = y1

        return a11.T, b11.T, c11.T, a12.T, b12.T, c12.T, a13.T, b13.T, c13.T, a14.T, b14.T, c14.T
    
    def get_old_p_param(self):
        return self.past_pvx,self.past_pvy,self.past_pphi,self.past_pdelta
    
    def get_old_p_paramVx(self):
        return self.past_pvx,self.past_pphi
    
    def update_model_noise(self,noise):
        self.noise_arr = self.model_noise.update_noise_matrix(noise)

    def update_new_p_param(self,new_pvx, new_pvy, new_pphi):
        self.p_vx = ca.DM(new_pvx).T
        self.p_vy = ca.DM(new_pvy).T
        self.p_phi = ca.DM(new_pphi).T
        # self.p_delta = ca.DM(new_pdelta).T

        # print(self.p_vx,self.p_vy,self.p_phi,self.p_delta)
        
    def update_new_p_paramVx(self, new_pphi):
        self.p_phi = ca.DM(new_pphi).T

    def save_fixed_noise(self):
        self.model_noise.save_noise_arr()

    def get_tau0_value(self):
        return self.tau0
    
    def get_LB(self):
        return self.zeta + (1-self.epsilon)*self.sol_gamma
    
    def get_reach_end(self):
        return self.reach_end
