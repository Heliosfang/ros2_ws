#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import carla
import math
import time
import tf2_ros


def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    q = Quaternion()
    half = 0.5 * yaw_rad
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


class CarlaOdomPublisher(Node):
    def __init__(self):
        super().__init__('carla_odom_publisher')

        # ---- Parameters ----
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 2000)
        self.declare_parameter('rate_hz', 20.0)

        self.declare_parameter('vehicle_actor_id', 0)
        self.declare_parameter('vehicle_role_name', 'ego')
        self.declare_parameter('vehicle_blueprint', 'vehicle.lincoln.mkz')
        self.declare_parameter('vehicle_blueprint_index', 0)

        self.declare_parameter('resolve_period_sec', 0.5)
        self.declare_parameter('log_every_n_resolves', 10)

        host = self.get_parameter('host').value
        port = int(self.get_parameter('port').value)
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)

        self.req_actor_id = int(self.get_parameter('vehicle_actor_id').value)
        self.req_role_name = self.get_parameter('vehicle_role_name').value or ""
        self.req_blueprint = self.get_parameter('vehicle_blueprint').value or ""
        self.req_bp_index = int(self.get_parameter('vehicle_blueprint_index').value)

        self.resolve_period = float(self.get_parameter('resolve_period_sec').value)
        self._resolve_attempts = 0
        self._announced_found = False

        # ---- Connect to CARLA ----
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

        # ---- Publishers ----
        self.odom_pub = self.create_publisher(Odometry, 'carla/odom', 10)
        self.state_pub = self.create_publisher(Float64MultiArray, 'carla/odom_xyphi_vxvyomega', 10)

        # ---- TF Broadcaster ----
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- State ----
        self.vehicle = None

        # ---- Timers ----
        self.resolve_timer = self.create_timer(self.resolve_period, self._try_resolve_vehicle)
        self.publish_timer = self.create_timer(self.dt, self._tick)

    def _try_resolve_vehicle(self):
        if self.vehicle is not None and self._is_actor_alive(self.vehicle):
            return

        self._resolve_attempts += 1
        if self._resolve_attempts % int(self.get_parameter('log_every_n_resolves').value) == 1:
            self.get_logger().info("Trying to resolve target vehicle...")

        actors = self.world.get_actors().filter('vehicle.*')

        if self.req_actor_id > 0:
            v = actors.find(self.req_actor_id)
            if v is not None:
                self._set_vehicle(v)
                return

        if self.req_role_name:
            matches = [a for a in actors if a.attributes.get('role_name', '') == self.req_role_name]
            if matches:
                self._set_vehicle(matches[0])
                return

        if self.req_blueprint:
            matches = actors.filter(self.req_blueprint)
            if len(matches) == 0:
                all_vs = actors.filter('vehicle.*')
                matches = [a for a in all_vs if a.type_id == self.req_blueprint]
            if matches:
                idx = max(0, min(self.req_bp_index, len(matches) - 1))
                self._set_vehicle(matches[idx])
                return

        if self._announced_found:
            self.get_logger().warn("Vehicle lost; will keep searching...")
            self._announced_found = False

    def _set_vehicle(self, v: carla.Actor):
        self.vehicle = v
        self._announced_found = True
        self.get_logger().info(
            f"Resolved vehicle: id={v.id}, type={v.type_id}, role_name='{v.attributes.get('role_name','')}'"
        )

    def _is_actor_alive(self, actor: carla.Actor) -> bool:
        try:
            _ = actor.id
            return True
        except Exception:
            return False

    def _tick(self):
        v = self.vehicle
        if v is None or not self._is_actor_alive(v):
            return

        try:
            tf = v.get_transform()
            loc = tf.location
            rot = tf.rotation

            x = float(loc.x)
            y = float(loc.y)
            phi = math.radians(float(rot.yaw))

            vw = v.get_velocity()
            vx_w = float(vw.x)
            vy_w = float(vw.y)

            c, s = math.cos(phi), math.sin(phi)
            vx_b =  c * vx_w + s * vy_w
            vy_b = -s * vx_w + c * vy_w

            omega = math.radians(float(v.get_angular_velocity().z))

            # ---- Odometry ----
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = 'map'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = float(loc.z)
            odom.pose.pose.orientation = yaw_to_quaternion(phi)
            odom.twist.twist.linear.x  = vx_b
            odom.twist.twist.linear.y  = vy_b
            odom.twist.twist.angular.z = omega
            self.odom_pub.publish(odom)

            # ---- TF (map -> base_link) ----
            t = TransformStamped()
            t.header.stamp = odom.header.stamp
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = float(loc.z)
            t.transform.rotation = yaw_to_quaternion(phi)
            self.tf_broadcaster.sendTransform(t)

            # ---- State array ----
            arr = Float64MultiArray()
            arr.layout.dim = [MultiArrayDimension(label='state', size=6, stride=6)]
            arr.data = [x, y, phi, vx_b, vy_b, omega]
            self.state_pub.publish(arr)

        except Exception as e:
            self.get_logger().warn(f"Odom tick failed: {e}")
            if not self._is_actor_alive(v):
                self.vehicle = None
                self._announced_found = False


def main(args=None):
    rclpy.init(args=args)
    node = CarlaOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
