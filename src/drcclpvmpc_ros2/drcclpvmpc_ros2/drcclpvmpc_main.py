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
import threading

from drcclpvmpc_ros2.racecar_path.racecar_path import RacecarPath
from drcclpvmpc_ros2.obstacles.obstacle_shape import Rectangle_obs
from drcclpvmpc_ros2.mpc.drcclpvmpc_core import STM_DRCCLPVMPC
from drcclpvmpc_ros2.dynamics.lpvdynamics import BicycleDynamics


class DRCCLPVMPCRos2Main(Node):
    def __init__(self):
        super().__init__("drcclpvmpc_main", automatically_declare_parameters_from_overrides=True)

        # Load all params from YAML
        all_params = {name: p.value for name, p in self.get_parameters_by_prefix('').items()}
        self.dt = all_params['dt_']
        self.wheel_base = all_params['wheel_base']
        self.horizon = all_params.get('horizon_', 6)
        self.track_width = all_params.get('track_width_', 1.0)
        self.use_drcc = all_params.get('use_drcc', True)
        self.safe_multiplier = all_params.get('safe_multiplier', 2.0)

        self.carParams = carSpec(all_params)
        self.model = BicycleDynamics(all_params)

        # Service name and goal topic
        self.services_name = all_params.get('service_name', 'get_path_from_txt')
        self.goal_topic    = all_params.get('goal_topic', '/move_base_simple/goal')

        # State
        self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)
        self.obs_sub  = self.create_subscription(Float32MultiArray, "/obs_box", self.obs_callback, 10)

        # Publisher (combined [steering, throttle])
        self.cmd_pub = self.create_publisher(Float32MultiArray, "/control_cmd", 10)

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
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z
        with self.lock:
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
        current_tau = self.track_ptr.xy_to_tau(current_state[:2])
        current_n = self.track_ptr.f_xy_to_taun(current_state[:2],current_tau)
        
        self.get_logger().debug("current tau is:{current_tau}")

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
            
        for i in range(obs_dm.shape[1]):
            box_i = obs_dm[:,i*4:(i*4)+4]
            min_tau_val = np.inf
            max_tau_val = -np.inf
            for j in range(4):
                tau_j = self.track_ptr.xy_to_tau(box_i[:,j])
                min_tau_val = min(min_tau_val, tau_j)
                max_tau_val = max(max_tau_val, tau_j)
            
            max_s_val = self.track_ptr.tau_to_s_lookup(max_tau_val)
            min_s_val = self.track_ptr.tau_to_s_lookup(min_tau_val)
            
            s1 = max_s_val + self.carParams['max_vx'] * self.safe_multiplier
            s1 = min(s1, self.s_max)
            s0 = min_s_val - self.carParams['max_vx'] * self.safe_multiplier
            s0 = max(s0, 0.1)
            
            max_tau_val = self.track_ptr.s_to_tau_lookup(s1)
            if max_tau_val < current_tau:
                continue
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
        else:
            refined_FL = self.track_ptr.f_taun_to_xy(max_tau, max_n)
            refined_FR = self.track_ptr.f_taun_to_xy(max_tau, min_n)
            # refined_RR = self.track_ptr.f_taun_to_xy(min_tau, min_n)
            refined_RL = self.track_ptr.f_taun_to_xy(min_tau, max_n)

            length = ca.hypot(refined_FL[0]-refined_RL[0], refined_FL[1]-refined_RL[1])
            width = ca.hypot(refined_FL[0]-refined_FR[0], refined_FL[1]-refined_FR[1])
            
            center_tau = (max_tau + min_tau) / 2.0
            center_n = (max_n + min_n) / 2.0
            center_xy = self.track_ptr.f_taun_to_xy(center_tau, center_n)
            
            center_phi = self.track_ptr.f_phi(center_tau) * 180 / ca.pi
            self.rec_obs = Rectangle_obs(center_xy, width, length, center_phi, side_avoid)
            self.get_logger().info(f"Received obstacle at tau range [{min_tau:.2f}, {max_tau:.2f}] on side {side_avoid}.")
            


        if self.test_bug:
            self.drcc_success, self.drcc_z0, self.drcc_control = self.controller.get_Updated_local_path(
                current_state, self.rec_obs, side_avoid, self.safe_multiplier, self.use_drcc
            )
            if self.drcc_success:
                steer = float(self.drcc_control[0,self.control_step])
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
            if not self.drcc_success:
                self.reach_end = self.controller.get_reach_end()
                if not self.reach_end:
                    # try to stop the car by giving a zero signal
                    self.control_step += 1
                    if self.drcc_control is not None and self.control_step < self.horizon:
                        self.get_logger().warn("Fail to solve the optimization problem, use previous feasible solution")
                        current_control = Float32MultiArray()
                        current_control.data = [self.drcc_control[0,self.control_step],self.drcc_control[1,self.control_step],0.0]
                        self.cmd_pub.publish(current_control)
                    else:
                        self.get_logger().warn("Fail to solve the optimization problem and use feasible solution")
                        current_control = Float32MultiArray()
                        current_control.data = [0,0,1]
                        self.cmd_pub.publish(current_control)
                else:
                    self.get_logger().info("Planner reached to the end")
                    # try to stop the car by giving a zero signal
                    current_control = Float32MultiArray()
                    current_control.data = [0,0,1]
                    self.cmd_pub.publish(current_control)
            else:
                drcc_z0 = np.array(drcc_z0).squeeze()
                drcc_control = np.array(drcc_control)
                lpv_pvx, lpv_pvy, lpv_pphi, lpv_pdelta = self.controller.get_old_p_param()
                lpv_pred_x = self.model.LPV_states(drcc_z0,drcc_control,lpv_pvx,lpv_pvy,lpv_pphi,lpv_pdelta,self.dt)
                new_pvx = lpv_pred_x[3,1:]
                new_pvy = lpv_pred_x[4,1:]
                new_pphi = lpv_pred_x[2,1:]
                self.controller.update_new_p_param(new_pvx,new_pvy,new_pphi)
                
                current_control = Float32MultiArray()
                current_control.data = [drcc_control[0,0],drcc_control[1,0],0.0]
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
    node = DRCCLPVMPCRos2Main()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
