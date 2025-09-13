#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, LaserScan, Image
import carla
import numpy as np
import random
import time
import math


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wrap_angle_pi(a):
    """Wrap angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def yaw_deg_to_rad(yaw_deg):
    return math.radians(float(yaw_deg))

def lerp(a, b, t):
    return a + (b - a) * t


class CarlaVehicleSpawner(Node):
    def __init__(self):
        super().__init__('carla_vehicle_spawner')

        # ---- Map & Vehicle (CARLA 0.9.16-friendly) ----
        self.declare_parameter("carla_map", "Town01")
        self.declare_parameter("vehicle_blueprint", "vehicle.lincoln.mkz_2017")

        # ---- Fixed spawn configuration ----
        self.declare_parameter("spawn_mode", "index")  # "index" or "pose"
        self.declare_parameter("spawn_index", 0)
        self.declare_parameter("spawn_x", 0.0)
        self.declare_parameter("spawn_y", 0.0)
        self.declare_parameter("spawn_z", 0.5)
        self.declare_parameter("spawn_roll", 0.0)
        self.declare_parameter("spawn_pitch", 0.0)
        self.declare_parameter("spawn_yaw", 0.0)

        # ---- 2D LiDAR configuration ----
        self.declare_parameter("lidar_range", 30.0)
        self.declare_parameter("lidar_upper_fov", 0.0)
        self.declare_parameter("lidar_lower_fov", 0.0)
        self.declare_parameter("lidar_points_per_second", 200000)
        self.declare_parameter("lidar_rotation_frequency", 10.0)
        self.declare_parameter("lidar_dropoff_general_rate", 0.0)
        self.declare_parameter("lidar_dropoff_intensity_limit", 1.0)
        self.declare_parameter("lidar_dropoff_zero_intensity", 0.0)

        # ---- Camera (optional) ----
        self.declare_parameter("enable_camera", True)
        self.declare_parameter("camera_width", 800)
        self.declare_parameter("camera_height", 600)
        self.declare_parameter("camera_fov", 90.0)
        self.declare_parameter("camera_sensor_tick", 0.0)
        
        # ---- Spectator follow (vehicle frame) ----
        self.declare_parameter("spectator_offset_x", 6.0)   # in front
        self.declare_parameter("spectator_offset_z", 3.0)   # above

        # Smoothing (EMA time constants, seconds)
        self.declare_parameter("spectator_smooth_tau_pos", 0.15)
        self.declare_parameter("spectator_smooth_tau_yaw", 0.20)

        # NEW: chase-cam feel
        self.declare_parameter("spectator_follow_lag_sec", 0.10)     # position lags behind velocity (s)
        self.declare_parameter("spectator_yaw_lookahead_sec", 0.15)  # look slightly ahead in heading (s)

        # Read params
        map_name     = self.get_parameter("carla_map").value
        vehicle_pref = self.get_parameter("vehicle_blueprint").value

        # ---- Connect to CARLA ----
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(5.0)

        world = None
        for _ in range(20):
            try:
                world = self.client.get_world()
                break
            except RuntimeError:
                self.get_logger().warn("CARLA not ready, retrying...")
                time.sleep(2)
        if world is None:
            raise RuntimeError("Could not connect to CARLA at 127.0.0.1:2000")

        # ---- Load requested map (simple) ----
        try:
            current_tail = world.get_map().name.split('/')[-1]
        except Exception:
            current_tail = ""
        if current_tail != map_name:
            self.get_logger().info(f"Loading map: {map_name}")
            world = self.client.load_world(map_name)
            world.wait_for_tick()
        else:
            self.get_logger().info(f"Already on requested map: {map_name}")

        self.world = world
        self.spectator = self.world.get_spectator()
        blueprint_library = self.world.get_blueprint_library()

        # ---- Vehicle blueprint ----
        try:
            vehicle_bp = blueprint_library.find(vehicle_pref)
            # tag so other nodes can find it deterministically
            vehicle_bp.set_attribute('role_name', 'ego')
        except Exception:
            self.get_logger().warn(f"'{vehicle_pref}' not found; choosing a random vehicle.")
            candidates = blueprint_library.filter('vehicle.*')
            if not candidates:
                raise RuntimeError("No vehicle blueprints available!")
            vehicle_bp = random.choice(candidates)

        # ---- Compute fixed spawn Transform ----
        spawn_mode = self.get_parameter("spawn_mode").value
        if spawn_mode == "index":
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points found on the map!")
            idx = int(self.get_parameter("spawn_index").value)
            if idx < 0 or idx >= len(spawn_points):
                self.get_logger().warn(
                    f"spawn_index {idx} out of range [0,{len(spawn_points)-1}], clamping."
                )
                idx = max(0, min(idx, len(spawn_points)-1))
            spawn_tf = spawn_points[idx]
            self.get_logger().info(f"Spawning by index {idx} at {spawn_tf.location}")
        elif spawn_mode == "pose":
            x = float(self.get_parameter("spawn_x").value)
            y = float(self.get_parameter("spawn_y").value)
            z = float(self.get_parameter("spawn_z").value)
            roll  = float(self.get_parameter("spawn_roll").value)
            pitch = float(self.get_parameter("spawn_pitch").value)
            yaw   = float(self.get_parameter("spawn_yaw").value)
            spawn_tf = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)  # degrees
            )
            self.get_logger().info(
                f"Spawning by pose at (x={x:.2f}, y={y:.2f}, z={z:.2f}, rpy=({roll},{pitch},{yaw}))"
            )
        else:
            raise ValueError("spawn_mode must be 'index' or 'pose'")

        # ---- Spawn vehicle (no randomness) ----
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at the requested fixed transform.")
        self.get_logger().info(f"Spawned vehicle: {self.vehicle.type_id}")

        # ---- Publishers BEFORE sensor callbacks ----
        self.lidar_pub = self.create_publisher(PointCloud2, 'carla/lidar', 10)
        self.scan_pub  = self.create_publisher(LaserScan,  'carla/scan',  10)
        self.cam_pub   = self.create_publisher(Image,      'carla/camera', 10) \
                         if self.get_parameter("enable_camera").value else None

        # ---- 2D LiDAR (single channel) ----
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(self.get_parameter("lidar_range").value))
        lidar_bp.set_attribute('channels', '1')
        lidar_bp.set_attribute('upper_fov', str(self.get_parameter("lidar_upper_fov").value))
        lidar_bp.set_attribute('lower_fov', str(self.get_parameter("lidar_lower_fov").value))
        lidar_bp.set_attribute('points_per_second', str(self.get_parameter("lidar_points_per_second").value))
        lidar_bp.set_attribute('rotation_frequency', str(self.get_parameter("lidar_rotation_frequency").value))
        lidar_bp.set_attribute('dropoff_general_rate', str(self.get_parameter("lidar_dropoff_general_rate").value))
        lidar_bp.set_attribute('dropoff_intensity_limit', str(self.get_parameter("lidar_dropoff_intensity_limit").value))
        lidar_bp.set_attribute('dropoff_zero_intensity', str(self.get_parameter("lidar_dropoff_zero_intensity").value))

        lidar_tf = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.vehicle)
        # self.lidar.listen(self.publish_lidar)

        # ---- Camera (optional) ----
        self.camera = None
        if self.get_parameter("enable_camera").value:
            cam_bp = blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(self.get_parameter("camera_width").value))
            cam_bp.set_attribute('image_size_y', str(self.get_parameter("camera_height").value))
            cam_bp.set_attribute('fov',          str(self.get_parameter("camera_fov").value))
            cam_bp.set_attribute('sensor_tick',  str(self.get_parameter("camera_sensor_tick").value))
            cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
            # self.camera.listen(self.publish_camera)

        # ---- Spectator follow (smooth & synced) ----
        self._spec_loc_filt = None  # np.array([x,y,z])
        self._spec_yaw_filt = None  # radians
        self.world.on_tick(self._on_world_tick)  # sync to sim

        self.spawned_actors = [self.vehicle, self.lidar, self.camera]

    # ---- Smooth spectator following, synced to CARLA tick ----
    def _on_world_tick(self, snapshot: carla.WorldSnapshot):
        if self.vehicle is None:
            return

        # Desired pose from vehicle transform + vehicle-frame offset
        T = self.vehicle.get_transform()
        ox = float(self.get_parameter("spectator_offset_x").value)
        oz = float(self.get_parameter("spectator_offset_z").value)
        desired_loc = T.transform(carla.Location(x=ox, y=0.0, z=oz))
        desired_yaw_rad = math.radians(float(T.rotation.yaw))

        # --- NEW: chase-cam lag & yaw look-ahead ---
        lag_sec   = max(0.0, float(self.get_parameter("spectator_follow_lag_sec").value))
        look_sec  = max(0.0, float(self.get_parameter("spectator_yaw_lookahead_sec").value))

        # World-frame linear velocity (m/s)
        v = self.vehicle.get_velocity()  # carla.Vector3D (x,y,z)
        # Apply positional lag: move desired cam point *backwards* by v * lag_sec
        if lag_sec > 0.0:
            desired_loc = carla.Location(
                x=desired_loc.x - v.x * lag_sec,
                y=desired_loc.y - v.y * lag_sec,
                z=desired_loc.z
            )

        # Angular velocity about Z (deg/s -> rad/s)
        omega_rad = math.radians(float(self.vehicle.get_angular_velocity().z))
        if look_sec > 0.0:
            desired_yaw_rad = desired_yaw_rad + omega_rad * look_sec

        # Init filters first tick
        if self._spec_loc_filt is None:
            self._spec_loc_filt = np.array([desired_loc.x, desired_loc.y, desired_loc.z], dtype=np.float32)
            self._spec_yaw_filt = desired_yaw_rad

        # Exponential smoothing
        dt = snapshot.timestamp.delta_seconds or 0.05
        tau_pos = max(1e-3, float(self.get_parameter("spectator_smooth_tau_pos").value))
        tau_yaw = max(1e-3, float(self.get_parameter("spectator_smooth_tau_yaw").value))
        alpha_pos = max(0.0, min(1.0, 1.0 - math.exp(-dt / tau_pos)))
        alpha_yaw = max(0.0, min(1.0, 1.0 - math.exp(-dt / tau_yaw)))

        # Position filter
        target = np.array([desired_loc.x, desired_loc.y, desired_loc.z], dtype=np.float32)
        self._spec_loc_filt = self._spec_loc_filt + (target - self._spec_loc_filt) * alpha_pos

        # Yaw filter with wrap-around handling
        def wrap(a): return (a + math.pi) % (2.0 * math.pi) - math.pi
        dyaw = wrap(desired_yaw_rad - self._spec_yaw_filt)
        self._spec_yaw_filt = wrap(self._spec_yaw_filt + alpha_yaw * dyaw)

        # Apply smoothed transform
        sm_loc = carla.Location(float(self._spec_loc_filt[0]),
                                float(self._spec_loc_filt[1]),
                                float(self._spec_loc_filt[2]))
        sm_rot = carla.Rotation(pitch=T.rotation.pitch,
                                yaw=math.degrees(self._spec_yaw_filt),
                                roll=T.rotation.roll)
        self.spectator.set_transform(carla.Transform(sm_loc, sm_rot))


    # ---- LiDAR publishers ----
    def publish_lidar(self, carla_lidar_data: carla.LidarMeasurement):
        pc = PointCloud2()
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.header.frame_id = "lidar_link"

        buf = carla_lidar_data.raw_data
        pts = np.frombuffer(buf, dtype=np.float32)   # [x,y,z,intensity]*N
        n = int(pts.size / 4)

        pc.height = 1
        pc.width = n
        pc.is_bigendian = False
        pc.is_dense = True
        pc.point_step = 16
        pc.row_step = pc.point_step * n
        pc.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc.data = buf
        self.lidar_pub.publish(pc)

        scan = LaserScan()
        scan.header = pc.header
        scan.angle_min = -math.pi
        scan.angle_max =  math.pi
        scan.range_min = 0.1
        scan.range_max = float(self.get_parameter("lidar_range").value)

        if n > 0:
            scan.angle_increment = (scan.angle_max - scan.angle_min) / n
            ranges, intens = [], []
            for i in range(n):
                x = pts[4*i + 0]
                y = pts[4*i + 1]
                ranges.append(math.hypot(x, y))
                intens.append(float(pts[4*i + 3]))
            scan.ranges = ranges
            scan.intensities = intens
        else:
            scan.angle_increment = 0.0
            scan.ranges = []
            scan.intensities = []

        self.scan_pub.publish(scan)

    # ---- Camera publisher ----
    def publish_camera(self, carla_image: carla.Image):
        if self.cam_pub is None:
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.height = carla_image.height
        msg.width  = carla_image.width
        msg.encoding = "bgra8"
        msg.step = msg.width * 4
        msg.data = carla_image.raw_data.tobytes()
        self.cam_pub.publish(msg)

    # ---- Cleanup ----
    def destroy_node(self):
        try:
            for a in self.spawned_actors:
                if a is not None:
                    a.destroy()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarlaVehicleSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()