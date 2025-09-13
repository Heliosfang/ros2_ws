#!/usr/bin/env python3
import time
import random
from typing import List, Tuple, Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

import carla


def bbox_corners_world_2d(actor: carla.Actor, inflation_radius: float = 0.0) -> List[Tuple[float, float]]:
    """Return 4 world-frame (x,y) corners of the actor’s (optionally inflated) bounding box:
       order: FL, FR, RR, RL."""
    T = actor.get_transform()
    bb = actor.bounding_box

    # Half-extents in local frame (+ inflation)
    r = max(0.0, float(inflation_radius))
    ex = float(bb.extent.x) + r   # forward half-length
    ey = float(bb.extent.y) + r   # half-width

    cx = float(bb.location.x)
    cy = float(bb.location.y)
    cz = float(bb.location.z)

    local = [
        carla.Location(x=cx + ex, y=cy + ey, z=cz),   # FL
        carla.Location(x=cx + ex, y=cy - ey, z=cz),   # FR
        carla.Location(x=cx - ex, y=cy - ey, z=cz),   # RR
        carla.Location(x=cx - ex, y=cy + ey, z=cz),   # RL
    ]
    return [(float(p.x), float(p.y)) for p in (T.transform(p) for p in local)]


class ObstacleSpawner(Node):
    def __init__(self):
        # Auto-declare any parameters provided via YAML/CLI
        super().__init__('obstacle_spawner', automatically_declare_parameters_from_overrides=True)

        # -------- Read scalar params with safe defaults (no re-declare) --------
        host = self._get_param_or('carla_host', '127.0.0.1')
        port = int(self._get_param_or('carla_port', 2000))
        rate_hz = float(self._get_param_or('publish_rate_hz', 10.0))
        self.inflation_radius = float(self._get_param_or('inflation_radius', 0.0))

        # -------- Read vehicles.* via prefix --------
        # YAML structure:
        # obstacle_spawner:
        #   ros__parameters:
        #     vehicles:
        #       car1: { blueprint: "...", x: ..., y: ..., z: ..., yaw: ... }
        params_by_prefix: Dict[str, rclpy.parameter.Parameter] = self.get_parameters_by_prefix('vehicles')
        vehicles_map: Dict[str, Dict[str, float]] = {}
        for key, param in params_by_prefix.items():
            parts = key.split('.', 1)  # "car1.blueprint" -> ["car1", "blueprint"]
            if len(parts) != 2:
                continue
            car_name, attr = parts[0], parts[1]
            vehicles_map.setdefault(car_name, {})[attr] = param.value

        vehicles_cfg = []
        for car_name in sorted(vehicles_map.keys()):
            cfg = vehicles_map[car_name]
            vehicles_cfg.append({
                'blueprint': str(cfg.get('blueprint', 'vehicle.audi.tt')),
                'x': float(cfg.get('x', 0.0)),
                'y': float(cfg.get('y', 0.0)),
                'z': float(cfg.get('z', 0.1)),
                'yaw': float(cfg.get('yaw', 0.0)),
            })

        # -------- Publisher & timer (always created) --------
        self.pub = self.create_publisher(Float32MultiArray, 'car_box', 10)
        period = 1.0 / max(1e-3, rate_hz)
        self.timer = self.create_timer(period, self.publish_boxes)

        # -------- If NO vehicles are defined: run in "empty publisher" mode --------
        if not vehicles_cfg:
            self.get_logger().warn(
                "No vehicles defined in parameters. Will publish an empty car_box list."
            )
            self.world = None
            self.actors: List[carla.Actor] = []
            # if self.inflation_radius > 0.0:
            #     self.get_logger().info(
            #         f"(Note) inflation_radius={self.inflation_radius:.3f} m has no effect with zero actors."
            #     )
            return

        # -------- Connect to CARLA (only if we actually need to spawn) --------
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)

        self.world = None
        for _ in range(30):
            try:
                self.world = self.client.get_world()
                break
            except RuntimeError:
                self.get_logger().warn('CARLA not ready, retrying...')
                time.sleep(1.0)
        if self.world is None:
            raise RuntimeError(f'Could not connect to CARLA at {host}:{port}')

        bp_lib = self.world.get_blueprint_library()

        # -------- Spawn all vehicles --------
        self.actors = []
        for i, cfg in enumerate(vehicles_cfg):
            bp_name = cfg['blueprint']
            try:
                bp = bp_lib.find(bp_name)
            except Exception:
                cand = bp_lib.filter('vehicle.*')
                if not cand:
                    raise RuntimeError('No vehicle blueprints available!')
                self.get_logger().warn(f"Blueprint '{bp_name}' not found; using a random vehicle.")
                bp = random.choice(cand)
            bp.set_attribute('role_name', f'obstacle_{i+1}')

            x, y, z, yaw = cfg['x'], cfg['y'], cfg['z'], cfg['yaw']
            tf = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
            )
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                # Cleanup any already-spawned actors
                for a in self.actors:
                    try:
                        a.destroy()
                    except Exception:
                        pass
                raise RuntimeError(f"Failed to spawn a vehicle at x={x}, y={y}, yaw={yaw}")
            self.actors.append(actor)
            self.get_logger().info(f"Spawned [{i+1}]: {actor.type_id} at ({x:.2f},{y:.2f}) yaw={yaw:.1f}")

        # if self.inflation_radius > 0.0:
        #     self.get_logger().info(f"Inflating bounding boxes by {self.inflation_radius:.3f} m.")
        # else:
        #     self.get_logger().info("Publishing original (non-inflated) bounding boxes.")

    # Helper: fetch parameter value if present, else default (no re-declare)
    def _get_param_or(self, name: str, default):
        if self.has_parameter(name):
            return self.get_parameter(name).value
        return default

    def publish_boxes(self):
        N = len(getattr(self, 'actors', []))
        flat: List[float] = []

        for actor in getattr(self, 'actors', []):
            try:
                corners = bbox_corners_world_2d(actor, self.inflation_radius)  # FL, FR, RR, RL
                flat.extend([
                    corners[0][0], corners[0][1],
                    corners[1][0], corners[1][1],
                    corners[2][0], corners[2][1],
                    corners[3][0], corners[3][1],
                ])
            except Exception as e:
                self.get_logger().warn(f'BBox calc failed: {e}')
                flat.extend([0.0] * 8)  # keep layout

        msg = Float32MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='cars', size=N, stride=N * 8),
                MultiArrayDimension(label='coords', size=8, stride=8),
            ],
            data_offset=0
        )
        msg.data = flat  # [] when N == 0
        self.pub.publish(msg)

    def destroy_node(self):
        # Best-effort cleanup of spawned actors (if any)
        try:
            for a in getattr(self, 'actors', []):
                if a is not None:
                    a.destroy()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
