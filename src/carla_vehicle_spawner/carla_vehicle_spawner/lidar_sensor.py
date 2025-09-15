#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import tf2_ros
import carla
import numpy as np
import time
import math
import random

class CarlaLidarAttacher(Node):
    def __init__(self):
        super().__init__('carla_lidar_attacher')

        # ---- Connection ----
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 2000)
        self.declare_parameter('timeout', 5.0)

        # ---- Vehicle selection (priority: actor_id > role_name > blueprint) ----
        self.declare_parameter('vehicle_actor_id', 0)                 # exact actor id
        self.declare_parameter('vehicle_role_name', 'ego')            # role_name used by your spawner
        self.declare_parameter('vehicle_blueprint', 'vehicle.*')      # fallback filter
        self.declare_parameter('vehicle_blueprint_index', 0)

        # ---- LiDAR mounting (relative to base_link, CARLA vehicle frame: x fwd, y right, z up) ----
        self.declare_parameter('lidar_frame_id', 'lidar_link')
        self.declare_parameter('lidar_mount_x', 0.0)                  # meters
        self.declare_parameter('lidar_mount_y', 0.0)
        self.declare_parameter('lidar_mount_z', 2.4)
        self.declare_parameter('lidar_mount_roll', 0.0)               # degrees
        self.declare_parameter('lidar_mount_pitch', 0.0)
        self.declare_parameter('lidar_mount_yaw', 0.0)

        # ---- LiDAR sensor parameters (CARLA 0.9.16 ray_cast) ----
        self.declare_parameter('range', 100.0)
        self.declare_parameter('channels', 32)
        self.declare_parameter('upper_fov', 10.0)
        self.declare_parameter('lower_fov', -30.0)
        self.declare_parameter('points_per_second', 1200000)
        self.declare_parameter('rotation_frequency', 20.0)
        self.declare_parameter('noise_stddev', 0.0)
        self.declare_parameter('dropoff_general_rate', 0.0)
        self.declare_parameter('dropoff_intensity_limit', 1.0)
        self.declare_parameter('dropoff_zero_intensity', 0.0)
        self.declare_parameter('atmosphere_attenuation_rate', 0.004)
        self.declare_parameter('sensor_tick', 0.0)  # 0 => every sim tick

        # ---- Other ----
        self.declare_parameter('resolve_period_sec', 0.5)

        host = self.get_parameter('host').value
        port = int(self.get_parameter('port').value)
        timeout = float(self.get_parameter('timeout').value)

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

        # Wait for world
        self.world = None
        for _ in range(60):
            try:
                self.world = self.client.get_world()
                break
            except RuntimeError:
                time.sleep(1.0)
        if self.world is None:
            raise RuntimeError(f'Could not connect to CARLA at {host}:{port}')

        # Pub/Sub/TF
        self.pc_pub = self.create_publisher(PointCloud2, 'carla/lidar', 10)
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)

        # State
        self.vehicle = None
        self.lidar = None
        self.lidar_frame = self.get_parameter('lidar_frame_id').value
        self._spawned_static_tf = False

        # Periodic resolver (find vehicle & attach lidar once)
        period = float(self.get_parameter('resolve_period_sec').value)
        self.create_timer(period, self._resolve_and_attach)

    # ---------- Resolve the target vehicle & attach LiDAR ----------
    def _resolve_and_attach(self):
        if self.lidar is not None:
            # already attached; ensure actor still alive
            if not self._is_actor_alive(self.lidar) or not self._is_actor_alive(self.vehicle):
                self.get_logger().warn('Attached LiDAR or vehicle disappeared; will reattach.')
                self._destroy_lidar_only()
            else:
                return  # all good

        if self.vehicle is None or not self._is_actor_alive(self.vehicle):
            self.vehicle = self._find_vehicle()
            if self.vehicle is None:
                return

        # Attach LiDAR
        try:
            bp_lib = self.world.get_blueprint_library()
            lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
            # Attributes
            def set_attr(key, val):
                try:
                    lidar_bp.set_attribute(key, str(val))
                except Exception:
                    pass

            set_attr('range', self.get_parameter('range').value)
            set_attr('channels', self.get_parameter('channels').value)
            set_attr('upper_fov', self.get_parameter('upper_fov').value)
            set_attr('lower_fov', self.get_parameter('lower_fov').value)
            set_attr('points_per_second', self.get_parameter('points_per_second').value)
            set_attr('rotation_frequency', self.get_parameter('rotation_frequency').value)
            set_attr('noise_stddev', self.get_parameter('noise_stddev').value)
            set_attr('dropoff_general_rate', self.get_parameter('dropoff_general_rate').value)
            set_attr('dropoff_intensity_limit', self.get_parameter('dropoff_intensity_limit').value)
            set_attr('dropoff_zero_intensity', self.get_parameter('dropoff_zero_intensity').value)
            set_attr('atmosphere_attenuation_rate', self.get_parameter('atmosphere_attenuation_rate').value)
            set_attr('sensor_tick', self.get_parameter('sensor_tick').value)

            # Mount transform (relative to vehicle)
            lx = float(self.get_parameter('lidar_mount_x').value)
            ly = float(self.get_parameter('lidar_mount_y').value)
            lz = float(self.get_parameter('lidar_mount_z').value)
            lroll = float(self.get_parameter('lidar_mount_roll').value)
            lpitch = float(self.get_parameter('lidar_mount_pitch').value)
            lyaw = float(self.get_parameter('lidar_mount_yaw').value)
            mount_tf = carla.Transform(
                carla.Location(x=lx, y=ly, z=lz),
                carla.Rotation(roll=lroll, pitch=lpitch, yaw=lyaw)
            )

            self.lidar = self.world.spawn_actor(lidar_bp, mount_tf, attach_to=self.vehicle)
            if self.lidar is None:
                self.get_logger().error('Failed to spawn LiDAR actor.')
                return

            self.get_logger().info(f'LiDAR attached to vehicle id={self.vehicle.id}, type={self.vehicle.type_id}')
            self.lidar.listen(self._on_lidar)
            self._publish_static_tf_base_to_lidar()

        except Exception as e:
            self.get_logger().error(f'Error attaching LiDAR: {e}')
            self._destroy_lidar_only()

    # ---------- LiDAR callback ----------
    def _on_lidar(self, meas: carla.LidarMeasurement):
        # Build PointCloud2: fields x,y,z,intensity (float32), 16 bytes/point
        buf = meas.raw_data  # already bytes: [x,y,z,intensity] float32 repeated
        pts = np.frombuffer(buf, dtype=np.float32)
        n = pts.size // 4

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.lidar_frame
        msg.height = 1
        msg.width = int(n)
        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = buf
        self.pc_pub.publish(msg)

    # ---------- Helpers ----------
    def _publish_static_tf_base_to_lidar(self):
        if self._spawned_static_tf:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = self.lidar_frame
        t.transform.translation.x = float(self.get_parameter('lidar_mount_x').value)
        t.transform.translation.y = float(self.get_parameter('lidar_mount_y').value)
        t.transform.translation.z = float(self.get_parameter('lidar_mount_z').value)

        # RPY (degrees) → quaternion; roll/pitch/yaw are about X/Y/Z in vehicle frame
        roll = math.radians(float(self.get_parameter('lidar_mount_roll').value))
        pitch = math.radians(float(self.get_parameter('lidar_mount_pitch').value))
        yaw = math.radians(float(self.get_parameter('lidar_mount_yaw').value))
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        # ZYX intrinsic
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_static.sendTransform(t)
        self._spawned_static_tf = True
        self.get_logger().info(f'Published static TF base_link → {self.lidar_frame}')

    def _find_vehicle(self):
        try:
            actors = self.world.get_actors().filter('vehicle.*')
        except Exception:
            return None

        # 1) actor_id
        aid = int(self.get_parameter('vehicle_actor_id').value)
        if aid > 0:
            v = actors.find(aid)
            if v is not None:
                self.get_logger().info(f'Found vehicle by actor_id={aid}')
                return v

        # 2) role_name
        role = self.get_parameter('vehicle_role_name').value or ''
        if role:
            matches = [a for a in actors if a.attributes.get('role_name', '') == role]
            if matches:
                self.get_logger().info(f"Found vehicle by role_name='{role}'")
                return matches[0]

        # 3) blueprint filter
        flt = self.get_parameter('vehicle_blueprint').value or 'vehicle.*'
        matches = actors.filter(flt)
        if matches:
            idx = max(0, min(int(self.get_parameter('vehicle_blueprint_index').value), len(matches) - 1))
            self.get_logger().info(f"Found vehicle by blueprint filter '{flt}' (idx={idx})")
            return matches[idx]

        self.get_logger().info('No target vehicle found yet (will retry)...')
        return None

    def _is_actor_alive(self, actor: carla.Actor) -> bool:
        try:
            _ = actor.id
            return True
        except Exception:
            return False

    def _destroy_lidar_only(self):
        try:
            if self.lidar is not None:
                self.get_logger().info('Destroying LiDAR actor')
                self.lidar.destroy()
        except Exception:
            pass
        finally:
            self.lidar = None

    def destroy_node(self):
        self._destroy_lidar_only()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarlaLidarAttacher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
