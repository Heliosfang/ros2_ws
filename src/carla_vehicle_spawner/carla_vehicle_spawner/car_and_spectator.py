#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import carla
import time
import random
import math
import numpy as np

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class CarlaCarAndSpectator(Node):
    def __init__(self):
        super().__init__('carla_car_and_spectator')

        # ---- Connection & world ----
        self.declare_parameter('carla_host', '127.0.0.1')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 5.0)
        self.declare_parameter('carla_map', 'Town01')

        # ---- Vehicle ----
        self.declare_parameter('vehicle_blueprint', 'vehicle.lincoln.mkz_2017')
        self.declare_parameter('role_name', 'ego')
        self.declare_parameter('enable_autopilot', False)

        # ---- Spawn configuration ----
        self.declare_parameter('spawn_mode', 'index')  # 'index' or 'pose'
        self.declare_parameter('spawn_index', 0)
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('spawn_z', 0.5)
        self.declare_parameter('spawn_roll', 0.0)      # deg
        self.declare_parameter('spawn_pitch', 0.0)     # deg
        self.declare_parameter('spawn_yaw', 0.0)       # deg

        # ---- Spectator (vehicle-frame offsets & smoothing) ----
        self.declare_parameter('spectator_offset_x', -8.0)  # behind (+ ahead, - behind)
        self.declare_parameter('spectator_offset_y', 0.0)   # lateral (left +, right -)
        self.declare_parameter('spectator_offset_z', 3.0)   # above

        self.declare_parameter('spectator_smooth_tau_pos', 0.15)  # s
        self.declare_parameter('spectator_smooth_tau_yaw', 0.20)  # s
        self.declare_parameter('spectator_follow_lag_sec', 0.10)  # s (position lag)
        self.declare_parameter('spectator_yaw_lookahead_sec', 0.15)  # s (heading look-ahead)

        # ---- Connect to CARLA ----
        host = self.get_parameter('carla_host').value
        port = int(self.get_parameter('carla_port').value)
        timeout = float(self.get_parameter('carla_timeout').value)
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

        world = None
        for _ in range(30):
            try:
                world = self.client.get_world()
                break
            except RuntimeError:
                self.get_logger().warn('CARLA not ready, retrying...')
                time.sleep(1.0)
        if world is None:
            raise RuntimeError(f'Could not connect to CARLA at {host}:{port}')
        self.world = world

        # ---- Load map if needed ----
        target_map = self.get_parameter('carla_map').value
        try:
            current_tail = self.world.get_map().name.split('/')[-1]
        except Exception:
            current_tail = ''
        if current_tail != target_map:
            self.get_logger().info(f'Loading map: {target_map}')
            self.world = self.client.load_world(target_map)
            self.world.wait_for_tick()
        else:
            self.get_logger().info(f'Already on requested map: {target_map}')

        # ---- Blueprint ----
        bp_lib = self.world.get_blueprint_library()
        veh_name = self.get_parameter('vehicle_blueprint').value
        try:
            vehicle_bp = bp_lib.find(veh_name)
        except IndexError:
            self.get_logger().warn(f"Blueprint '{veh_name}' not found; choosing a random vehicle.")
            candidates = bp_lib.filter('vehicle.*')
            if not candidates:
                raise RuntimeError('No vehicle blueprints available!')
            vehicle_bp = random.choice(candidates)
        # role_name
        try:
            vehicle_bp.set_attribute('role_name', str(self.get_parameter('role_name').value))
        except Exception:
            pass

        # ---- Spawn transform ----
        mode = self.get_parameter('spawn_mode').value
        if mode == 'index':
            sps = self.world.get_map().get_spawn_points()
            if not sps:
                raise RuntimeError('No spawn points on this map.')
            idx = int(self.get_parameter('spawn_index').value)
            if idx < 0 or idx >= len(sps):
                self.get_logger().warn(f'spawn_index {idx} out of range [0,{len(sps)-1}], clamping.')
                idx = max(0, min(idx, len(sps)-1))
            spawn_tf = sps[idx]
            self.get_logger().info(f'Spawning by index {idx} at {spawn_tf.location}')
        elif mode == 'pose':
            x = float(self.get_parameter('spawn_x').value)
            y = float(self.get_parameter('spawn_y').value)
            z = float(self.get_parameter('spawn_z').value)
            roll  = float(self.get_parameter('spawn_roll').value)
            pitch = float(self.get_parameter('spawn_pitch').value)
            yaw   = float(self.get_parameter('spawn_yaw').value)
            spawn_tf = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
            )
            self.get_logger().info(f'Spawning by pose at (x={x:.2f}, y={y:.2f}, z={z:.2f}, rpy=({roll},{pitch},{yaw}))')
        else:
            raise ValueError("spawn_mode must be 'index' or 'pose'")

        # ---- Spawn vehicle ----
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)
        if self.vehicle is None:
            raise RuntimeError('Failed to spawn vehicle at the requested transform.')
        self.get_logger().info(f"Spawned vehicle: {self.vehicle.type_id} (id={self.vehicle.id})")

        # ---- Autopilot (optional) ----
        if bool(self.get_parameter('enable_autopilot').value):
            try:
                tm = self.client.get_trafficmanager()
                self.vehicle.set_autopilot(True, tm.get_port())
                self.get_logger().info('Autopilot enabled (Traffic Manager).')
            except Exception as e:
                self.get_logger().warn(f'Failed TM autopilot, trying legacy: {e}')
                try:
                    self.vehicle.set_autopilot(True)
                except Exception:
                    self.get_logger().error('Legacy autopilot failed.')

        # ---- Spectator control ----
        self.spectator = self.world.get_spectator()
        self._spec_loc_filt = None  # np.array([x,y,z])
        self._spec_yaw_filt = None  # radians
        self.world.on_tick(self._on_world_tick)

        # cleanup tracking
        self.spawned_actor = self.vehicle

    # ---- Spectator follow (smooth chase-cam) ----
    def _on_world_tick(self, snapshot: carla.WorldSnapshot):
        if self.vehicle is None:
            return

        T = self.vehicle.get_transform()
        yaw_deg = float(T.rotation.yaw)
        yaw = math.radians(yaw_deg)

        # Vehicle-frame offset (x forward, y left, z up), then rotated to world
        off_x = float(self.get_parameter('spectator_offset_x').value)
        off_y = float(self.get_parameter('spectator_offset_y').value)
        off_z = float(self.get_parameter('spectator_offset_z').value)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        wx = T.location.x + off_x * cos_y - off_y * sin_y
        wy = T.location.y + off_x * sin_y + off_y * cos_y
        wz = T.location.z + off_z
        desired_loc = carla.Location(x=wx, y=wy, z=wz)

        # Yaw + look-ahead
        look_sec = max(0.0, float(self.get_parameter('spectator_yaw_lookahead_sec').value))
        omega_rad = math.radians(float(self.vehicle.get_angular_velocity().z))  # deg/s -> rad/s
        desired_yaw = yaw + omega_rad * look_sec

        # Position lag by velocity (in world)
        lag_sec = max(0.0, float(self.get_parameter('spectator_follow_lag_sec').value))
        if lag_sec > 0.0:
            v = self.vehicle.get_velocity()
            desired_loc = carla.Location(
                x=desired_loc.x - v.x * lag_sec,
                y=desired_loc.y - v.y * lag_sec,
                z=desired_loc.z
            )

        # Init filters
        if self._spec_loc_filt is None:
            self._spec_loc_filt = np.array([desired_loc.x, desired_loc.y, desired_loc.z], dtype=np.float32)
            self._spec_yaw_filt = desired_yaw

        # EMA smoothing
        dt = snapshot.timestamp.delta_seconds or 0.05
        tau_pos = max(1e-3, float(self.get_parameter('spectator_smooth_tau_pos').value))
        tau_yaw = max(1e-3, float(self.get_parameter('spectator_smooth_tau_yaw').value))
        a_pos = max(0.0, min(1.0, 1.0 - math.exp(-dt / tau_pos)))
        a_yaw = max(0.0, min(1.0, 1.0 - math.exp(-dt / tau_yaw)))

        target = np.array([desired_loc.x, desired_loc.y, desired_loc.z], dtype=np.float32)
        self._spec_loc_filt += (target - self._spec_loc_filt) * a_pos

        dyaw = _wrap_pi(desired_yaw - self._spec_yaw_filt)
        self._spec_yaw_filt = _wrap_pi(self._spec_yaw_filt + a_yaw * dyaw)

        sm_loc = carla.Location(float(self._spec_loc_filt[0]),
                                float(self._spec_loc_filt[1]),
                                float(self._spec_loc_filt[2]))
        sm_rot = carla.Rotation(pitch=T.rotation.pitch,
                                yaw=math.degrees(self._spec_yaw_filt),
                                roll=T.rotation.roll)
        self.spectator.set_transform(carla.Transform(sm_loc, sm_rot))

    # ---- Cleanup ----
    def destroy_node(self):
        try:
            if hasattr(self, 'spawned_actor') and self.spawned_actor is not None:
                self.get_logger().info(f'Destroying vehicle (id={self.spawned_actor.id})')
                self.spawned_actor.destroy()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarlaCarAndSpectator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
