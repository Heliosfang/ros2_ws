#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile
import math
import time

try:
    from ackermann_msgs.msg import AckermannDrive
    HAVE_ACKERMANN = True
except Exception:
    HAVE_ACKERMANN = False

import carla


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class CarlaVehicleController(Node):
    def __init__(self):
        super().__init__('carla_vehicle_controller')

        # ---------- Parameters ----------
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 2000)

        # Which vehicle to control (same precedence used in odom node)
        self.declare_parameter('vehicle_actor_id', 0)          # >0 to force a specific actor id
        self.declare_parameter('vehicle_role_name', 'ego')     # recommended: tag in spawner
        self.declare_parameter('vehicle_blueprint', 'vehicle.lincoln.mkz')
        self.declare_parameter('vehicle_blueprint_index', 0)

        # Mapping from accel (m/s^2) -> throttle/brake
        self.declare_parameter('accel_to_throttle_gain', 0.25)   # throttle ≈ gain * accel (clip to [0,1])
        self.declare_parameter('accel_to_brake_gain',   0.30)    # brake    ≈ gain * |accel|
        self.declare_parameter('deadband_accel',        0.05)    # |accel| below this → neither throttle nor brake

        # Steering normalization and rate limit
        self.declare_parameter('steer_saturation', 1.0)          # expected steer_norm range
        self.declare_parameter('steer_rate_limit',  2.0)         # max change per second in steer command (abs), in normalized units

        # How to interpret /control_cmd data=[steer_rad, throttle_or_accel]
        #  - "steer_throttle": throttle in [0,1] (no brake)
        #  - "steer_accel":    accel in m/s^2  (mapped to throttle/brake)
        self.declare_parameter('control_cmd_mode', 'steer_throttle')
        self.declare_parameter('max_steer_angle_rad', 0.6)       # fallback if CARLA physics not available

        # Retry behavior
        self.declare_parameter('resolve_period_sec', 0.5)
        self.declare_parameter('log_every_n_resolves', 10)

        # Read params
        host = self.get_parameter('host').value
        port = int(self.get_parameter('port').value)

        self.req_actor_id = int(self.get_parameter('vehicle_actor_id').value)
        self.req_role_name = self.get_parameter('vehicle_role_name').value or ""
        self.req_blueprint = self.get_parameter('vehicle_blueprint').value or ""
        self.req_bp_index = int(self.get_parameter('vehicle_blueprint_index').value)

        self.accel_to_throttle = float(self.get_parameter('accel_to_throttle_gain').value)
        self.accel_to_brake    = float(self.get_parameter('accel_to_brake_gain').value)
        self.deadband_accel    = float(self.get_parameter('deadband_accel').value)

        self.steer_sat         = float(self.get_parameter('steer_saturation').value)
        self.steer_rate_limit  = float(self.get_parameter('steer_rate_limit').value)

        self.control_cmd_mode  = str(self.get_parameter('control_cmd_mode').value).lower()
        # Start with fallback; will be overwritten by CARLA physics if available
        self.max_steer_angle   = float(self.get_parameter('max_steer_angle_rad').value)

        self.resolve_period    = float(self.get_parameter('resolve_period_sec').value)
        self._resolve_attempts = 0
        self._vehicle_found    = False

        # State
        self.vehicle = None
        self.prev_time = None
        self.prev_steer = 0.0
        self.reverse = False  # can be extended with a service to toggle

        # ---------- Connect to CARLA ----------
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)

        self.world = None
        for _ in range(60):  # wait up to 60s for server
            try:
                self.world = self.client.get_world()
                break
            except RuntimeError:
                time.sleep(1.0)
        if self.world is None:
            raise RuntimeError(f"Could not connect to CARLA at {host}:{port}")

        # ---------- Subscribers ----------
        qos = QoSProfile(depth=10)

        # Subscribe to your combined control topic: [steer_rad, throttle_or_accel]
        self.ctrl_sub = self.create_subscription(
            Float32MultiArray, '/control_cmd', self._on_control_cmd, qos
        )

        # Back-compat: accel+steer_norm cmd
        self.accel_steer_sub = self.create_subscription(
            Float32MultiArray, 'carla/accel_steer_cmd', self._on_accel_steer, qos
        )

        if HAVE_ACKERMANN:
            self.ackermann_sub = self.create_subscription(
                AckermannDrive, 'carla/ackermann_cmd', self._on_ackermann, qos
            )
            self.get_logger().info("Ackermann subscriber enabled (ackermann_msgs found).")
        else:
            self.get_logger().info("ackermann_msgs not available; skipping Ackermann subscriber.")

        # ---------- Timers ----------
        self.resolve_timer = self.create_timer(self.resolve_period, self._try_resolve_vehicle)

        self.get_logger().info(
            f"Listening for /control_cmd as [steer_rad, {'throttle(0..1)' if self.control_cmd_mode=='steer_throttle' else 'accel(m/s^2)'}]; "
            f"(initial) max_steer_angle_rad={self.max_steer_angle:.3f}, mode='{self.control_cmd_mode}'"
        )

    # ---------------- Vehicle resolution ----------------
    def _try_resolve_vehicle(self):
        if self.vehicle is not None and self._is_actor_alive(self.vehicle):
            return
        self._resolve_attempts += 1
        if self._resolve_attempts % int(self.get_parameter('log_every_n_resolves').value) == 1:
            self.get_logger().info("Trying to resolve vehicle to control...")

        actors = self.world.get_actors().filter('vehicle.*')

        # 1) actor id
        if self.req_actor_id > 0:
            v = actors.find(self.req_actor_id)
            if v is not None:
                self._set_vehicle(v)
                return

        # 2) role name
        if self.req_role_name:
            matches = [a for a in actors if a.attributes.get('role_name','') == self.req_role_name]
            if matches:
                self._set_vehicle(matches[0])
                return

        # 3) blueprint
        if self.req_blueprint:
            matches = actors.filter(self.req_blueprint)
            if not matches:
                all_vs = actors.filter('vehicle.*')
                matches = [a for a in all_vs if a.type_id == self.req_blueprint]
            if matches:
                idx = max(0, min(self.req_bp_index, len(matches)-1))
                self._set_vehicle(matches[idx])
                return

    def _set_vehicle(self, v: carla.Actor):
        self.vehicle = v
        self._vehicle_found = True
        self.prev_time = time.time()
        self.prev_steer = 0.0

        self.get_logger().info(
            f"Controlling vehicle id={v.id}, type={v.type_id}, role_name='{v.attributes.get('role_name','')}'"
        )
        # NEW: read max steer angle from physics (deg → rad)
        self._refresh_max_steer_from_vehicle()

    def _refresh_max_steer_from_vehicle(self):
        """Read maximum steering angle from CARLA physics and set self.max_steer_angle (radians).
        Falls back to current value if anything fails."""
        if self.vehicle is None:
            return
        try:
            phys = self.vehicle.get_physics_control()
            wheels = getattr(phys, 'wheels', [])
            if not wheels:
                self.get_logger().warn("Vehicle physics has no wheels array; keeping configured max_steer_angle_rad.")
                return
            # Take the max across all wheels; front wheels usually have non-zero steer limits.
            max_deg = max(float(w.max_steer_angle) for w in wheels if hasattr(w, 'max_steer_angle'))
            if max_deg > 1e-6:
                self.max_steer_angle = math.radians(max_deg)
                self.get_logger().info(f"Detected max steering angle from physics: {max_deg:.2f} deg = {self.max_steer_angle:.3f} rad")
            else:
                self.get_logger().warn("max_steer_angle from physics is ~0; keeping configured fallback.")
        except Exception as e:
            self.get_logger().warn(f"Could not read max steer angle from physics: {e}; keeping configured fallback.")

    def _is_actor_alive(self, actor: carla.Actor) -> bool:
        try:
            _ = actor.id
            return True
        except Exception:
            return False

    # ---------------- Command handlers ----------------
    def _on_control_cmd(self, msg: Float32MultiArray):
        """
        Expect data = [steer_rad, throttle_or_accel]
        - steer_rad: steering angle in radians (positive = left)
        - throttle_or_accel: behavior depends on 'control_cmd_mode':
            * 'steer_throttle': throttle in [0,1], no brake
            * 'steer_accel':    accel in m/s^2 (mapped to throttle/brake)
        """
        if self.vehicle is None or not self._is_actor_alive(self.vehicle):
            return
        if not msg.data or len(msg.data) < 3:
            return

        steer_rad = float(msg.data[0])
        second    = float(msg.data[1])
        brake    = float(msg.data[2])

        # Convert rad -> normalized steer
        steer_norm = 0.0
        if self.max_steer_angle > 1e-6:
            steer_norm = clamp(steer_rad / self.max_steer_angle, -1.0, 1.0)
        steer = self._limit_steer_rate(self._normalize_steer(steer_norm))

        throttle = 0.0
        # brake = 0.0
        if self.control_cmd_mode == 'steer_accel':
            throttle, brake = self._accel_to_tb(second)  # treat as accel m/s^2
        else:
            # 'steer_throttle' → treat as normalized throttle [0,1]
            throttle = clamp(second, 0.0, 1.0)
            brake = clamp(brake, 0.0, 1.0)

        ctrl = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            reverse=self.reverse,
            hand_brake=False,
            manual_gear_shift=False
        )
        try:
            self.vehicle.apply_control(ctrl)
        except Exception as e:
            self.get_logger().warn(f"apply_control failed: {e}")

    def _on_accel_steer(self, msg: Float32MultiArray):
        """Back-compat: Expect data = [accel_mps2, steer_norm]."""
        if self.vehicle is None or not self._is_actor_alive(self.vehicle):
            return
        if not msg.data or len(msg.data) < 2:
            return

        accel = float(msg.data[0])          # m/s^2 (positive forward, negative brake)
        steer_norm = float(msg.data[1])     # normalized [-1, 1]

        throttle, brake = self._accel_to_tb(accel)
        steer = self._limit_steer_rate(self._normalize_steer(steer_norm))

        ctrl = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            reverse=self.reverse,
            hand_brake=False,
            manual_gear_shift=False
        )
        try:
            self.vehicle.apply_control(ctrl)
        except Exception as e:
            self.get_logger().warn(f"apply_control failed: {e}")

    def _on_ackermann(self, msg):
        """Map AckermannDrive to CARLA control."""
        if self.vehicle is None or not self._is_actor_alive(self.vehicle):
            return

        accel = float(msg.acceleration) if hasattr(msg, 'acceleration') else 0.0
        steer_angle = float(msg.steering_angle) if hasattr(msg, 'steering_angle') else 0.0

        # Use dynamically detected max steer angle (fallback to 0.6 rad if unknown)
        assumed_max_rad = self.max_steer_angle if self.max_steer_angle > 1e-6 else 0.6
        steer_norm = clamp(steer_angle / assumed_max_rad, -1.0, 1.0)

        throttle, brake = self._accel_to_tb(accel)
        steer = self._limit_steer_rate(self._normalize_steer(steer_norm))

        ctrl = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            reverse=self.reverse,
            hand_brake=False,
            manual_gear_shift=False
        )
        try:
            self.vehicle.apply_control(ctrl)
        except Exception as e:
            self.get_logger().warn(f"apply_control failed: {e}")

    # ---------------- Helpers: mapping & limits ----------------
    def _accel_to_tb(self, accel_mps2: float):
        """Map desired longitudinal acceleration to throttle / brake."""
        a = accel_mps2
        dead = self.deadband_accel
        if abs(a) < dead:
            return 0.0, 0.0
        if a > 0.0:
            th = clamp(self.accel_to_throttle * a, 0.0, 1.0)
            return th, 0.0
        else:
            br = clamp(self.accel_to_brake * (-a), 0.0, 1.0)
            return 0.0, br

    def _normalize_steer(self, steer_norm: float):
        """Clamp normalized steer to CARLA range [-1,1], honoring steer_saturation param."""
        sat = max(1e-3, self.steer_sat)
        s = clamp(steer_norm / sat, -1.0, 1.0)
        return s

    def _limit_steer_rate(self, target: float):
        """Limit rate of change of steering to avoid unrealistically sharp commands."""
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            self.prev_steer = target
            return target
        dt = max(1e-3, now - self.prev_time)
        max_delta = self.steer_rate_limit * dt
        new_val = clamp(target, self.prev_steer - max_delta, self.prev_steer + max_delta)
        self.prev_time = now
        self.prev_steer = new_val
        return new_val


def main(args=None):
    rclpy.init(args=args)
    node = CarlaVehicleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
