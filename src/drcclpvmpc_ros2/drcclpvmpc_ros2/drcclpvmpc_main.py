#!/usr/bin/env python3
import math
import numpy as np
import casadi as ca
from casadi import DM

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from nav_msgs.srv import GetPlan
from std_msgs.msg import Float32MultiArray

from drcclpvmpc_ros2.vehicle.carSpec import carSpec
import threading, queue
import matplotlib.pyplot as plt

from drcclpvmpc_ros2.racecar_path.racecar_path import RacecarPath
from drcclpvmpc_ros2.obstacles.obstacle_shape import Rectangle_obs
from drcclpvmpc_ros2.mpc.drcclpvmpc_core import STM_DRCCLPVMPC
from drcclpvmpc_ros2.dynamics.lpvdynamics import BicycleDynamics
from drcclpvmpc_ros2.dynamics.lpvdynamics_vx import BicycleDynamicsVx
from drcclpvmpc_ros2.racecar_path.utils import plot_path

import open3d as o3d
import os

class DRCCLPVMPCRos2Main(Node):
    def __init__(self):
        super().__init__("drcclpvmpc_main", automatically_declare_parameters_from_overrides=True)

        # Load all params from YAML
        all_params = {name: p.value for name, p in self.get_parameters_by_prefix('').items()}
        self.dt = all_params['dt_']
        self.wheel_base = all_params['wheel_base']
        self.horizon = all_params.get('horizon_', 6)
        self.track_width = all_params.get('track_width_', 4.0)
        
        self.use_drcc = all_params.get('use_drcc', True)
        self.safe_multiplier = all_params.get('safe_multiplier', 2.0)

        self.carParams = carSpec(all_params)
        
        self.approx = self.carParams['approx'] # True for 'acc' or 'vx', False for 'pwm'

        if self.approx:
            self.model = BicycleDynamicsVx(all_params)
        else:
            self.model = BicycleDynamics(all_params)
            
        # Service name and goal topic
        self.services_name = all_params.get('service_name', 'get_path_from_txt')
        self.goal_topic    = all_params.get('goal_topic', '/move_base_simple/goal')

        # State
        if self.approx:
            self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0] # [x, y, phi, vy, omega]
        else:
            self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [x, y, phi, vx, vy, omega]
        self.track_ptr = None
        self.lock = threading.Lock()
        self.initialized = False
        self.rec_obs = None
        self.controller = None
        self.test_bug = False
        self.s_max = np.inf
        self.drcc_success = None
        self.drcc_z0 = None
        self.drcc_control = None
        self.control_step = 0
        self.reach_end = False
        self.prev_control = None
        
        ################ load pcd environment map ###############
        self.env_map = all_params.get('environment_map', 'Town05')
        
        ################ Ensure continuous phi ##################
        self.prev_odom_phi = None
        self.prev_rec_phi = None
        #########################################################

        self.track_queue = queue.Queue()
        self.obs_queue = queue.Queue()
        self.lpvx_queue = queue.Queue()
        self.lpvy_queue = queue.Queue()
        self.refx_queue = queue.Queue()
        self.refy_queue = queue.Queue()
        self.P_queue = queue.Queue()
        
        self.atau_queue = queue.Queue()
        self.btau_queue = queue.Queue()
        self.an_queue = queue.Queue()
        self.bn_queue = queue.Queue()
        self.tau0_queue = queue.Queue()
        self.tau1_queue = queue.Queue()
        
        ###################################################################################
        ###################################################################################
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)
        self.obs_sub  = self.create_subscription(Float32MultiArray, "/obs_box", self.obs_callback, 10)

        # Publisher (combined [steering, throttle, brake])
        self.cmd_pub = self.create_publisher(Float32MultiArray, "/control_cmd", 10)
        
        # Publisher (combined [steering, vx, brake])
        self.vel_cmd_pub = self.create_publisher(Float32MultiArray, "/velocity_cmd", 10)

        # Service client
        self.plan_client = self.create_client(GetPlan, self.services_name)

        self.get_logger().info(
            f"DRCC-LPVMPC node initialized. Subscribed to /odom, {self.goal_topic}, /obs_box. "
            f"Publishing [steer, throttle] on /control_cmd. "
            f"Using service: {self.services_name}"
        )

    # ------------------- Subscribers -------------------
    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        phi = math.atan2(siny_cosp, cosy_cosp)
        
        if self.prev_odom_phi is not None:
            delta = np.arctan2(np.sin(phi - self.prev_odom_phi), np.cos(phi - self.prev_odom_phi))
            phi = self.prev_odom_phi + delta
        self.prev_odom_phi = phi
        # print("odom phi:", phi)
        vx = msg.twist.twist.linear.x
        # print("vx:", vx)
        vy = msg.twist.twist.linear.y
        # print("vy:", vy)
        omega = msg.twist.twist.angular.z
        # print("omega:", omega)
        with self.lock:
            if self.approx:
                self.current_state = [x, y, phi, vy, omega]
            else:
                self.current_state = [x, y, phi, vx, vy, omega]

    def goal_callback(self, goal: PoseStamped):
        if not self.plan_client.service_is_ready():
            ready = self.plan_client.wait_for_service(timeout_sec=2.0)
            if not ready:
                self.get_logger().warn(f"Service '{self.services_name}' not available; skipping goal.")
                return
        req = GetPlan.Request()
        req.goal = goal
        req.tolerance = 0.0
        with self.lock:
            x, y = self.current_state[0], self.current_state[1]
        start = PoseStamped()
        start.header.frame_id = goal.header.frame_id or "map"
        start.pose.position.x = float(x)
        start.pose.position.y = float(y)
        start.pose.position.z = 0.0
        start.pose.orientation.w = 1.0
        req.start = start
        future = self.plan_client.call_async(req)
        future.add_done_callback(self._on_plan_response)
        gx, gy = goal.pose.position.x, goal.pose.position.y
        self.get_logger().info(f"Goal received at ({gx:.2f}, {gy:.2f}); requested plan from service.")

    # ------------------- /obs_box handler -------------------
    def obs_callback(self, msg: Float32MultiArray):
        if not self.initialized or self.reach_end:
            return

        data = list(msg.data)
        if not data:
            return
        if len(data) % 2 != 0:
            data = data[:-1]

        global_min_tau = np.inf

        with self.lock:
            current_state = self.current_state
            self.P_queue.put(current_state[0:3])
        current_tau = self.track_ptr.xy_to_tau(current_state[:2])
        current_s = float(self.track_ptr.tau_to_s_lookup(current_tau))
        # current_n = self.track_ptr.f_xy_to_taun(current_state[:2],current_tau)
        
        # self.get_logger().debug("current tau is:{current_tau}")

        try:
            arr = np.asarray(data, dtype=float).reshape(-1, 2) # N*2
            obs_dm = ca.DM(arr.T) # 2*N
        except Exception as e:
            self.get_logger().warn(f"Failed to parse /obs_box: {e}")
            return

        max_tau = -np.inf
        min_tau = np.inf
        max_n = -np.inf
        min_n = np.inf
        s1 = np.inf
        s0 = -np.inf
        min_s0 = np.inf
        box_num = obs_dm.shape[1] // 4
            
        for i in range(box_num):
            box_i = obs_dm[:,i*4:(i*4)+4]
            min_tau_val = np.inf
            max_tau_val = -np.inf
            for j in range(4):
                tau_j = self.track_ptr.xy_to_tau(box_i[:,j])
                min_tau_val = min(min_tau_val, tau_j)
                max_tau_val = max(max_tau_val, tau_j)
            
            max_s_val = self.track_ptr.tau_to_s_lookup(max_tau_val)
            min_s_val = self.track_ptr.tau_to_s_lookup(min_tau_val)
            
            s1 = max_s_val + self.carParams['max_vx'] * self.safe_multiplier * self.horizon * self.dt
            s1 = min(s1, self.s_max)
            s0 = min_s_val - self.carParams['max_vx'] * self.safe_multiplier * self.horizon * self.dt
            s0 = max(s0, 0.1)
            print("s0:", s0, "s1:", s1)
            
            max_tau_val = self.track_ptr.s_to_tau_lookup(s1)
            if max_tau_val < current_tau:
                continue
            min_s0 = min(min_s0, s0)
            if min_tau_val < global_min_tau:
                global_min_tau = min_tau_val
                FL, FR, RR, RL = box_i[:,0], box_i[:,1], box_i[:,2], box_i[:,3]
                FL_tau = self.track_ptr.xy_to_tau(FL)
                FR_tau = self.track_ptr.xy_to_tau(FR)
                RR_tau = self.track_ptr.xy_to_tau(RR)
                RL_tau = self.track_ptr.xy_to_tau(RL)
                FL_n = self.track_ptr.f_xy_to_taun(FL,FL_tau)
                FR_n = self.track_ptr.f_xy_to_taun(FR,FR_tau)
                RR_n = self.track_ptr.f_xy_to_taun(RR,RR_tau)
                RL_n = self.track_ptr.f_xy_to_taun(RL,RL_tau)
                max_n = max(FL_n, FR_n, RR_n, RL_n)
                min_n = min(FL_n, FR_n, RR_n, RL_n)
                max_tau = max(FL_tau, FR_tau, RR_tau, RL_tau)
                min_tau = min(FL_tau, FR_tau, RR_tau, RL_tau)
                if max_n < 0 or min_n > 0:
                    global_min_tau = np.inf
                    max_tau, min_tau, max_n, min_n = -np.inf, np.inf, -np.inf, np.inf
                    continue
        side_avoid = 1 if (max_n + min_n < 0) else -1
        if (side_avoid == 1 and max_n >= self.track_width) or (side_avoid == -1 and min_n <= -self.track_width):
            self.get_logger().warn("The obstacle ahead is blocking the way, I will stop and wait.")
            cmd_msg = Float32MultiArray()
            cmd_msg.data = [0.0, 0.0, 1.0]
            self.cmd_pub.publish(cmd_msg)
        
        elif global_min_tau < np.inf and current_s > min_s0 - 5.0:
            refined_FL = self.track_ptr.f_taun_to_xy(max_tau, max_n)
            refined_FR = self.track_ptr.f_taun_to_xy(max_tau, min_n)
            # refined_RR = self.track_ptr.f_taun_to_xy(min_tau, min_n)
            refined_RL = self.track_ptr.f_taun_to_xy(min_tau, max_n)
            
            # print("refined FL:", refined_FL)
            # print("refined FR:", refined_FR)
            # print("refined RL:", refined_RL)
            length = np.hypot(refined_FL[0]-refined_RL[0], refined_FL[1]-refined_RL[1])
            width = np.hypot(refined_FL[0]-refined_FR[0], refined_FL[1]-refined_FR[1])
            
            # print("obs length:", length, "obs width:", width)
            
            center_tau = (max_tau + min_tau) / 2.0
            center_n = (max_n + min_n) / 2.0
            center_xy = np.array(self.track_ptr.f_taun_to_xy(center_tau, center_n))
            
            center_phi = self.track_ptr.f_phi(center_tau)
            if self.prev_rec_phi is not None:
                delta = np.arctan2(np.sin(center_phi - self.prev_rec_phi), np.cos(center_phi - self.prev_rec_phi))
                center_phi = self.prev_rec_phi + delta
            self.prev_rec_phi = center_phi
            self.rec_obs = Rectangle_obs(center_xy, float(width), float(length), center_phi*180/ca.pi, side_avoid)
            self.obs_queue.put(self.rec_obs)
            # self.get_logger().info(f"Received obstacle at tau range [{min_tau}, {max_tau}] on side {side_avoid}.")
            

        with self.lock:
            if self.test_bug:
                self.drcc_success, self.drcc_z0, self.drcc_control = self.controller.get_Updated_local_path(
                    current_state, self.rec_obs, side_avoid, self.safe_multiplier, self.use_drcc
                )
                if self.drcc_success:
                    steer = float(self.drcc_control[0,self.control_step])
                    if self.approx:
                        vx = float(self.drcc_control[1,self.control_step])
                        brake = 0.0
                        vel_msg = Float32MultiArray()
                        vel_msg.data = [steer, vx, brake]
                        self.vel_cmd_pub.publish(vel_msg)
                        self.get_logger().info(f"Publishing control: steer={steer:.3f}, vx={vx:.3f}")
                    else:
                        throttle = float(self.drcc_control[1,self.control_step])
                        brake = 0.0
                        cmd_msg = Float32MultiArray()
                        cmd_msg.data = [steer, throttle, brake]
                        self.cmd_pub.publish(cmd_msg)
                        self.get_logger().info(f"Publishing control: steer={steer:.3f}, throttle={throttle:.3f}")
                self.test_bug = False
            else:
                self.drcc_success, self.drcc_z0, self.drcc_control = self.controller.get_Updated_local_path(
                    current_state, self.rec_obs, side_avoid, self.safe_multiplier, self.use_drcc
                )
                reference_xyz = self.controller.get_reference_path()
                reference_x = reference_xyz[0,:]
                reference_y = reference_xyz[1,:]
                
                self.atau_queue.put(self.controller.get_obs_atau())
                self.btau_queue.put(self.controller.get_obs_btau())
                self.an_queue.put(self.controller.get_obs_an())
                self.bn_queue.put(self.controller.get_obs_bn())
                self.tau0_queue.put(self.controller.get_obs_tau0())
                self.tau1_queue.put(self.controller.get_obs_tau1())

                ################### store the local reference path for plot ###########
                ref_x = np.array(reference_x).transpose()
                ref_y = np.array(reference_y).transpose()
                self.refx_queue.put(ref_x)
                self.refy_queue.put(ref_y)
                
                if not self.drcc_success:
                    self.reach_end = self.controller.get_reach_end()
                    if not self.reach_end:
                        # try to stop the car by giving a zero signal
                        self.control_step += 1
                        if self.prev_control is not None and self.control_step < self.horizon:
                            self.get_logger().warn("Fail to solve the optimization problem, use previous feasible solution")
                            current_control = Float32MultiArray()
                            current_control.data = [self.prev_control[0,self.control_step],self.prev_control[1,self.control_step],0.0]
                            if self.approx:
                                self.vel_cmd_pub.publish(current_control)
                            else:
                                self.cmd_pub.publish(current_control)
                        else:
                            self.get_logger().warn("Fail to solve the optimization problem and use feasible solution")
                            current_control = Float32MultiArray()
                            current_control.data = [0,0,1]
                            if self.approx:
                                self.vel_cmd_pub.publish(current_control)
                            else:
                                self.cmd_pub.publish(current_control)
                    else:
                        self.get_logger().info("Planner reached to the end")
                        # try to stop the car by giving a zero signal
                        current_control = Float32MultiArray()
                        current_control.data = [0,0,1]
                        if self.approx:
                            self.vel_cmd_pub.publish(current_control)
                        else:
                            self.cmd_pub.publish(current_control)
                else:
                    self.control_step = 0
                    self.prev_control = self.drcc_control
                    drcc_z0 = np.array(self.drcc_z0).squeeze()
                    drcc_control = np.array(self.drcc_control)
                    if self.approx:
                        lpv_pvx, lpv_pphi = self.controller.get_old_p_paramVx()
                        lpv_pred_x = self.model.LPV_states(drcc_z0,drcc_control,lpv_pvx,lpv_pphi,self.dt)
                    else:
                        lpv_pvx, lpv_pvy, lpv_pphi, lpv_pdelta = self.controller.get_old_p_param()
                        lpv_pred_x = self.model.LPV_states(drcc_z0,drcc_control,lpv_pvx,lpv_pvy,lpv_pphi,lpv_pdelta,self.dt)
                    
                    ################ store the lpv predicted trajectory for plot ###########
                    lpv_x = lpv_pred_x[0,:]
                    lpv_y = lpv_pred_x[1,:]
                    
                    self.lpvx_queue.put(lpv_x)
                    self.lpvy_queue.put(lpv_y)
                    
                    if self.approx:
                        new_pphi = lpv_pred_x[2,1:]
                        self.controller.update_new_p_paramVx(new_pphi)
                    else:
                        new_pvx = lpv_pred_x[3,1:]
                        new_pvy = lpv_pred_x[4,1:]
                        new_pphi = lpv_pred_x[2,1:]
                        self.controller.update_new_p_param(new_pvx,new_pvy,new_pphi)
                    
                    current_control = Float32MultiArray()
                    if not self.approx:
                        current_control.data = [drcc_control[0,self.control_step],drcc_control[1,self.control_step],0.0]
                    else:
                        current_control.data = [drcc_control[0,self.control_step],0.42,0.0]
                    # if self.approx:
                    #     print("steer:", drcc_control[0,self.control_step], "speed:", drcc_control[1,self.control_step])
                    # else:
                    #     print("steer:", drcc_control[0,self.control_step], "throttle:", drcc_control[1,self.control_step])
                    if not self.approx:
                        self.vel_cmd_pub.publish(current_control)
                    else:
                        self.cmd_pub.publish(current_control)

    # ------------------- Service response handler -------------------
    def _on_plan_response(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"GetPlan call failed: {e}")
            return
        path: Path = resp.plan
        N = len(path.poses)
        if N == 0:
            with self.lock:
                self.track_ptr = None
            return
        xs = [ps.pose.position.x for ps in path.poses]
        ys = [ps.pose.position.y for ps in path.poses]
        mat = np.vstack([xs, ys])
        with self.lock:
            latest_path_dm = ca.DM(mat)
            self.track_ptr = RacecarPath(latest_path_dm, DM())
            # plot_path(self.track_ptr,type=1,labels="reference track")
            self.track_queue.put(self.track_ptr)
            self.s_max = self.track_ptr.get_max_length()
            start_tau = 1
            start_pt = self.track_ptr.f_taun_to_xy(start_tau, 0.0)
            start_phi = self.track_ptr.f_phi(start_tau)[0]
            self.current_state = DM([start_pt[0], start_pt[1], float(start_phi), 0.0, 0.0, 0.0])
            self.controller = STM_DRCCLPVMPC(
                self.track_ptr, self.current_state, self.dt,
                self.horizon, self.track_width,
                True, self.carParams
            )
            self.reach_end = False
            self.initialized = True
        self.get_logger().info(f"Received plan with {N} poses; DM shape = {latest_path_dm.shape}.")
        
def main(args=None):
    rclpy.init(args=args)
    
    ###################################################################################
    ############################ initialize the live plot #############################
    ax = plt.gca()
    ax.invert_yaxis()
    plt.ion()
    
    LnS, = ax.plot([], [], 'r',alpha=1,lw=2,label="Trajectory")
    LnR, = ax.plot([], [], '-b', marker='o', markersize=1, lw=1,label="Local Reference")
    LnP, = ax.plot([], [], 'g', marker='o', alpha=0.5, markersize=5,label="current position")
    LnO, = ax.plot([],[],lw =2, color = "tab:blue",label = "Orientation")
    LnH, = ax.plot([], [], '-g', marker='o', markersize=1, lw=0.5)
    Lna0, = ax.plot([], [], lw = 1, ls='--', color='black')
    Lnb1, = ax.plot([], [], lw = 1, ls='--', color='black')
    
    

    xylabel_fontsize = 26
    legend_fontsize = 26
    xytick_size = 26
    ax.set_xlabel('x [m]',fontsize = xylabel_fontsize)
    ax.set_ylabel('y [m]',fontsize = xylabel_fontsize)
    ax.legend(fontsize=legend_fontsize,borderpad=0.1,labelspacing=0.2, handlelength=1.4, handletextpad=0.37,loc='lower right')
    ax.tick_params(axis='both',which='major',labelsize = xytick_size)
    ax.figure.set_size_inches(10, 10)
    ###################################################################################
    ###################################################################################
    node = DRCCLPVMPCRos2Main()
    Px_data = []
    Py_data = []
    arrowLength = 0.5
    patch_obs = None
    prev_patch = None
    
    if node.env_map:
        home_dir = os.path.expanduser("~")
        pcd_file = os.path.join(home_dir, "Carla-0916", "HDMaps", "Town05.pcd")
        pcd = o3d.io.read_point_cloud(pcd_file)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-285, -100, -1),   # x_min, y_min, z_min
        max_bound=(-150, 70,  1)    # x_max, y_max, z_max
        )       
        pcd = pcd.crop(bbox)
        # --- Downsample to reduce number of points ---
        voxel_size = 0.8  # meters, adjust as needed
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd.points)
        ax.scatter(pts[:,0], pts[:,1], s=0.1, c='gray', alpha=0.5)
    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            
            refx = None
            refy = None
            lpv_x = None
            lpv_y = None
            rec_obs = None
            current_x = None
            current_y = None
            current_phi = None
            
            atau = None
            btau = None
            an = None
            bn = None
            tau0 = None
            tau1 = None

            while not node.atau_queue.empty():
                atau = node.atau_queue.get()
                btau = node.btau_queue.get()
                an = node.an_queue.get()
                bn = node.bn_queue.get()
                tau0 = node.tau0_queue.get()
                tau1 = node.tau1_queue.get()
            
            while not node.track_queue.empty():
                track_ptr = node.track_queue.get()
                plot_path(track_ptr,type=1,labels="reference track")
                
            while not node.obs_queue.empty():
                rec_obs = node.obs_queue.get()
                
            while not node.P_queue.empty():
                current_P = node.P_queue.get()
                Px_data.append(current_P[0])
                Py_data.append(current_P[1])
                current_x = current_P[0]
                current_y = current_P[1]
                current_phi = current_P[2]
                
            while not node.refx_queue.empty():
                refx = node.refx_queue.get()
                refy = node.refy_queue.get()
                
            while not node.lpvx_queue.empty():
                lpv_x = node.lpvx_queue.get()
                lpv_y = node.lpvy_queue.get()
                
            if rec_obs is not None:
                
                x,y = rec_obs.get_rectanglexy()
                patch_obs = ax.fill(x,y,color='black',zorder=1, alpha=1)
                
                if prev_patch is not None:
                    prev_patch[0].remove()
                    prev_patch = patch_obs
            else:
                if patch_obs is not None:
                    patch_obs[0].remove()
                    patch_obs = None

            if lpv_x is not None:
                LnH.set_xdata(lpv_x)
                LnH.set_ydata(lpv_y)
                
            if current_x is not None and current_y is not None:
                LnP.set_xdata([current_x])
                LnP.set_ydata([current_y])
                LnO.set_data([current_x, current_x+arrowLength*np.cos(current_phi)],[current_y, current_y+arrowLength*np.sin(current_phi)])
                
            if refx is not None:
                LnR.set_xdata(refx)
                LnR.set_ydata(refy)
            LnS.set_xdata(Px_data)
            LnS.set_ydata(Py_data)
            
            if patch_obs is not None:
                if atau is not None:
                    if an > 0:
                        safe_ataus = ca.linspace(tau0, atau, 10).T

                        safe_ans = ca.linspace(0, an, 10).T

                        safe_axys = node.track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

                        safe_btaus = ca.linspace(btau, tau1, 10).T

                        safe_bns = ca.linspace(bn, 0, 10).T

                        safe_bxys = node.track_ptr.f_taun_to_xy(safe_btaus,safe_bns)

                    else:
                        safe_ataus = ca.linspace(atau, tau0, 10).T

                        safe_ans = ca.linspace(an, 0, 10).T

                        safe_axys = node.track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

                        safe_btaus = ca.linspace(tau1, btau, 10).T

                        safe_bns = ca.linspace(0, bn, 10).T

                        safe_bxys = node.track_ptr.f_taun_to_xy(safe_btaus,safe_bns)
                    a0 = np.array(safe_axys)
                    b1 = np.array(safe_bxys)
                    
                    Lna0.set_xdata(a0[0,:])
                    Lna0.set_ydata(a0[1,:])
                    Lnb1.set_xdata(b1[0,:])
                    Lnb1.set_ydata(b1[1,:])
            else:
                Lna0.set_xdata([])
                Lna0.set_ydata([])
                Lnb1.set_xdata([])
                Lnb1.set_ydata([])
            # ax.relim()
            # ax.autoscale_view()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
