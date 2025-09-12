#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import carla
import time
import threading
import sys
import termios
import tty
import select
import math

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class CarlaVehicleKeyboard(Node):
    def __init__(self):
        super().__init__('carla_vehicle_keyboard')

        # ---------- Parameters ----------
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 2000)

        # Which vehicle to control
        self.declare_parameter('vehicle_actor_id', 0)
        self.declare_parameter('vehicle_role_name', 'ego')   # recommend tagging in spawner
        self.declare_parameter('vehicle_blueprint', 'vehicle.lincoln.mkz')  # fallback
        self.declare_parameter('vehicle_blueprint_index', 0)

        # Control rate and steps
        self.declare_parameter('control_rate_hz', 20.0)      # how often we send controls
        self.declare_parameter('duty_step', 0.05)            # W/S step per key press
        self.declare_parameter('steer_step', 0.05)           # A/D step per key press
        self.declare_parameter('steer_rate_limit', 2.5)      # max |Δsteer| per second (normalized units)
        self.declare_parameter('resolve_period_sec', 0.5)    # how often to search for the car if not found

        # Read params
        host = self.get_parameter('host').value
        port = int(self.get_parameter('port').value)
        self.req_actor_id   = int(self.get_parameter('vehicle_actor_id').value)
        self.req_role_name  = self.get_parameter('vehicle_role_name').value or ""
        self.req_blueprint  = self.get_parameter('vehicle_blueprint').value or ""
        self.req_bp_index   = int(self.get_parameter('vehicle_blueprint_index').value)

        self.ctrl_dt        = 1.0 / float(self.get_parameter('control_rate_hz').value)
        self.duty_step      = float(self.get_parameter('duty_step').value)
        self.steer_step     = float(self.get_parameter('steer_step').value)
        self.steer_rate_lim = float(self.get_parameter('steer_rate_limit').value)
        self.resolve_period = float(self.get_parameter('resolve_period_sec').value)

        # State
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)

        self.world = None
        for _ in range(60):
            try:
                self.world = self.client.get_world()
                break
            except RuntimeError:
                time.sleep(1.0)
        if self.world is None:
            raise RuntimeError(f"Could not connect to CARLA at {host}:{port}")

        self.vehicle = None
        self._resolve_attempts = 0

        # Current command state
        self.duty = 0.0          # [-1, 1] ; >0 throttle, <0 brake
        self._saved_duty = 0.0   # for resume after panic brake
        self.steer = 0.0         # [-1, 1]
        self.reverse = False
        self._last_apply_time = time.time()
        self._last_steer = 0.0

        # Timers
        self.resolve_timer = self.create_timer(self.resolve_period, self._try_resolve_vehicle)
        self.control_timer = self.create_timer(self.ctrl_dt, self._apply_control)

        # Keyboard thread (raw mode)
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._kb_thread.start()
        self.get_logger().info("Keyboard control: W/S throttle & brake, A/D steer, SPACE brake, X center, R reverse, C clear brake, Q/ESC quit")

    # ------------------- Vehicle resolution -------------------
    def _try_resolve_vehicle(self):
        if self.vehicle is not None and self._is_actor_alive(self.vehicle):
            return

        self._resolve_attempts += 1
        if self._resolve_attempts % 10 == 1:
            self.get_logger().info("Searching for vehicle to control...")

        actors = self.world.get_actors().filter('vehicle.*')

        # 1) actor id
        if self.req_actor_id > 0:
            v = actors.find(self.req_actor_id)
            if v is not None:
                self._set_vehicle(v)
                return

        # 2) role_name
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
        self.get_logger().info(
            f"Controlling vehicle id={v.id}, type={v.type_id}, role_name='{v.attributes.get('role_name','')}'"
        )

    def _is_actor_alive(self, actor: carla.Actor) -> bool:
        try:
            _ = actor.id
            return True
        except Exception:
            return False

    # ------------------- Control application -------------------
    def _apply_control(self):
        v = self.vehicle
        if v is None or not self._is_actor_alive(v):
            return

        now = time.time()
        dt = max(1e-3, now - self._last_apply_time)
        self._last_apply_time = now

        # Rate limit steering change
        max_d = self.steer_rate_lim * dt
        steer_cmd = clamp(self.steer, self._last_steer - max_d, self._last_steer + max_d)
        self._last_steer = steer_cmd

        # Map duty to throttle/brake
        duty = clamp(self.duty, -1.0, 1.0)
        throttle = clamp(duty, 0.0, 1.0)
        brake    = clamp(-duty, 0.0, 1.0)

        ctrl = carla.VehicleControl(
            throttle=throttle,
            steer=clamp(steer_cmd, -1.0, 1.0),
            brake=brake,
            reverse=self.reverse,
            hand_brake=False,
            manual_gear_shift=False
        )
        try:
            v.apply_control(ctrl)
        except Exception as e:
            self.get_logger().warn(f"apply_control failed: {e}")

    # ------------------- Keyboard handler -------------------
    def _keyboard_loop(self):
        # Use raw terminal to capture single key presses
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # noncanonical, no echo
            while rclpy.ok():
                if self._key_available():
                    ch = sys.stdin.read(1)
                    if not ch:
                        continue
                    code = ord(ch)
                    key = ch.lower()

                    if key == 'w':
                        self.duty = clamp(self.duty + self.duty_step, -1.0, 1.0)
                    elif key == 's':
                        self.duty = clamp(self.duty - self.duty_step, -1.0, 1.0)
                    elif key == 'a':
                        self.steer = clamp(self.steer - self.steer_step, -1.0, 1.0)
                    elif key == 'd':
                        self.steer = clamp(self.steer + self.steer_step, -1.0, 1.0)
                    elif key == 'x':
                        self.steer = 0.0
                    elif key == 'r':
                        self.reverse = not self.reverse
                        self.get_logger().info(f"Reverse toggled: {self.reverse}")
                    elif key == 'c':
                        # clear brake and resume saved duty
                        self.duty = self._saved_duty
                    elif key == ' ':
                        # panic brake: save duty, then full brake
                        self._saved_duty = self.duty
                        self.duty = -1.0
                        self.steer = 0.0
                    elif key == 'q' or code == 27:  # 'q' or ESC
                        self.get_logger().info("Exit requested from keyboard.")
                        rclpy.shutdown()
                        break

                    # Optional on-screen HUD
                    sys.stdout.write(f"\rW/S duty={self.duty:+.2f}   A/D steer={self.steer:+.2f}   reverse={self.reverse}      ")
                    sys.stdout.flush()
                else:
                    time.sleep(0.01)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\n")
            sys.stdout.flush()

    @staticmethod
    def _key_available():
        return select.select([sys.stdin], [], [], 0.0)[0] != []

def main(args=None):
    rclpy.init(args=args)
    node = CarlaVehicleKeyboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
