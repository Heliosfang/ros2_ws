#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros
import carla
import numpy as np
import time
import math

class CarlaCameraAttacher(Node):
    def __init__(self):
        super().__init__('carla_camera_attacher')

        # ---- Connection ----
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 2000)
        self.declare_parameter('timeout', 5.0)

        # ---- Vehicle selection (priority: actor_id > role_name > blueprint) ----
        self.declare_parameter('vehicle_actor_id', 0)
        self.declare_parameter('vehicle_role_name', 'ego')
        self.declare_parameter('vehicle_blueprint', 'vehicle.*')
        self.declare_parameter('vehicle_blueprint_index', 0)

        # ---- Camera mounting (relative to base_link; CARLA: x fwd, y right, z up) ----
        self.declare_parameter('camera_frame_id', 'camera_link')
        self.declare_parameter('mount_x', 1.5)
        self.declare_parameter('mount_y', 0.0)
        self.declare_parameter('mount_z', 2.0)
        self.declare_parameter('mount_roll', 0.0)   # deg
        self.declare_parameter('mount_pitch', 0.0)  # deg
        self.declare_parameter('mount_yaw', 0.0)    # deg

        # ---- Camera sensor params (CARLA RGB camera) ----
        self.declare_parameter('image_width', 800)
        self.declare_parameter('image_height', 600)
        self.declare_parameter('fov', 90.0)               # degrees
        self.declare_parameter('sensor_tick', 0.0)        # 0 => every sim tick
        # Encoding options: "bgr8" (default, converts) or "bgra8" (raw)
        self.declare_parameter('ros_image_encoding', 'bgr8')

        # Optional CameraInfo publishing
        self.declare_parameter('publish_camera_info', True)
        self.declare_parameter('camera_info_frame_id', 'camera_link')
        self.declare_parameter('camera_info_topic', 'carla/camera_info')

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

        # Publishers & TF
        self.img_pub = self.create_publisher(Image, 'carla/camera/image', 10)
        self.cinfo_pub = self.create_publisher(CameraInfo,
                                               self.get_parameter('camera_info_topic').value,
                                               10) if bool(self.get_parameter('publish_camera_info').value) else None
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)

        # State
        self.vehicle = None
        self.camera = None
        self.cam_frame = self.get_parameter('camera_frame_id').value
        self._static_tf_sent = False

        # Precompute intrinsics if we’ll publish CameraInfo
        self._camera_info = None
        if self.cinfo_pub is not None:
            self._camera_info = self._make_camera_info_msg()

        # Periodic resolver (find vehicle & attach camera once)
        period = float(self.get_parameter('resolve_period_sec').value)
        self.create_timer(period, self._resolve_and_attach)

    # ---------- Resolve the target vehicle & attach camera ----------
    def _resolve_and_attach(self):
        if self.camera is not None:
            if not self._is_actor_alive(self.camera) or not self._is_actor_alive(self.vehicle):
                self.get_logger().warn('Attached camera or vehicle disappeared; will reattach.')
                self._destroy_camera_only()
            else:
                return

        if self.vehicle is None or not self._is_actor_alive(self.vehicle):
            self.vehicle = self._find_vehicle()
            if self.vehicle is None:
                return

        try:
            bp_lib = self.world.get_blueprint_library()
            cam_bp = bp_lib.find('sensor.camera.rgb')

            # Attributes
            def set_attr(k, v):
                try:
                    cam_bp.set_attribute(k, str(v))
                except Exception:
                    pass

            w = int(self.get_parameter('image_width').value)
            h = int(self.get_parameter('image_height').value)
            fov = float(self.get_parameter('fov').value)
            tick = float(self.get_parameter('sensor_tick').value)

            set_attr('image_size_x', w)
            set_attr('image_size_y', h)
            set_attr('fov', fov)
            set_attr('sensor_tick', tick)

            # Mount transform
            lx = float(self.get_parameter('mount_x').value)
            ly = float(self.get_parameter('mount_y').value)
            lz = float(self.get_parameter('mount_z').value)
            lroll = float(self.get_parameter('mount_roll').value)
            lpitch = float(self.get_parameter('mount_pitch').value)
            lyaw = float(self.get_parameter('mount_yaw').value)
            mount_tf = carla.Transform(
                carla.Location(x=lx, y=ly, z=lz),
                carla.Rotation(roll=lroll, pitch=lpitch, yaw=lyaw)
            )

            self.camera = self.world.spawn_actor(cam_bp, mount_tf, attach_to=self.vehicle)
            if self.camera is None:
                self.get_logger().error('Failed to spawn camera actor.')
                return

            self.get_logger().info(f'Camera attached to vehicle id={self.vehicle.id}, type={self.vehicle.type_id}')
            self.camera.listen(self._on_camera)
            self._publish_static_tf_base_to_camera()

        except Exception as e:
            self.get_logger().error(f'Error attaching camera: {e}')
            self._destroy_camera_only()

    # ---------- Camera callback ----------
    def _on_camera(self, carla_image: carla.Image):
        # CARLA provides BGRA raw bytes (4 bytes per pixel)
        width = carla_image.width
        height = carla_image.height

        # Build ROS Image
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cam_frame

        desired_enc = self.get_parameter('ros_image_encoding').value.lower()
        if desired_enc == 'bgra8':
            # no conversion: keep 4 channels
            msg.height = height
            msg.width = width
            msg.encoding = 'bgra8'
            msg.step = width * 4
            msg.data = bytes(carla_image.raw_data)
        else:
            # default: convert BGRA -> BGR (drop alpha)
            arr = np.frombuffer(carla_image.raw_data, dtype=np.uint8).reshape((height, width, 4))
            bgr = arr[:, :, :3]  # drop alpha
            msg.height = height
            msg.width = width
            msg.encoding = 'bgr8'
            msg.step = width * 3
            msg.data = bgr.tobytes()

        self.img_pub.publish(msg)

        if self.cinfo_pub is not None and self._camera_info is not None:
            cinfo = self._camera_info
            cinfo.header.stamp = msg.header.stamp
            cinfo.header.frame_id = self.get_parameter('camera_info_frame_id').value
            self.cinfo_pub.publish(cinfo)

    # ---------- Helpers ----------
    def _publish_static_tf_base_to_camera(self):
        if self._static_tf_sent:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = self.cam_frame
        t.transform.translation.x = float(self.get_parameter('mount_x').value)
        t.transform.translation.y = float(self.get_parameter('mount_y').value)
        t.transform.translation.z = float(self.get_parameter('mount_z').value)

        # RPY (deg) -> quaternion (intrinsic ZYX)
        roll = math.radians(float(self.get_parameter('mount_roll').value))
        pitch = math.radians(float(self.get_parameter('mount_pitch').value))
        yaw = math.radians(float(self.get_parameter('mount_yaw').value))
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_static.sendTransform(t)
        self._static_tf_sent = True
        self.get_logger().info(f'Published static TF base_link → {self.cam_frame}')

    def _make_camera_info_msg(self) -> CameraInfo:
        w = int(self.get_parameter('image_width').value)
        h = int(self.get_parameter('image_height').value)
        fov_deg = float(self.get_parameter('fov').value)
        f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)  # simple pinhole, square pixels

        cinfo = CameraInfo()
        cinfo.header.frame_id = self.get_parameter('camera_info_frame_id').value
        cinfo.width = w
        cinfo.height = h
        cinfo.k = [f, 0, w/2.0,
                   0, f, h/2.0,
                   0, 0, 1]
        cinfo.p = [f, 0, w/2.0, 0,
                   0, f, h/2.0, 0,
                   0, 0, 1, 0]
        cinfo.d = []           # no distortion (CARLA pinhole approx)
        cinfo.r = [1,0,0, 0,1,0, 0,0,1]
        cinfo.distortion_model = 'plumb_bob'
        return cinfo

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

    def _destroy_camera_only(self):
        try:
            if self.camera is not None:
                self.get_logger().info('Destroying camera actor')
                self.camera.destroy()
        except Exception:
            pass
        finally:
            self.camera = None

    def destroy_node(self):
        self._destroy_camera_only()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarlaCameraAttacher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
