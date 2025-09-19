#!/usr/bin/env python3
import math, time, random
from typing import List, Tuple, Dict
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import carla

def rot2d(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x*c - y*s, x*s + y*c

def rect_corners(center, yaw, L, W):
    # local corners (forward +x, left +y): FL, FR, RR, RL
    L2, W2 = 0.5*L, 0.5*W
    pts_local = [(+L2,+W2),(+L2,-W2),(-L2,-W2),(-L2,+W2)]
    cx, cy = center
    return [(cx+rot2d(x,y,yaw)[0], cy+rot2d(x,y,yaw)[1]) for (x,y) in pts_local]

def side_midpoints(corners):
    """Return midpoints of the 4 edges in order: front(FL-FR), right(FR-RR), rear(RR-RL), left(RL-FL)."""
    FL, FR, RR, RL = corners
    mid_front = ((FL[0]+FR[0])/2.0, (FL[1]+FR[1])/2.0)
    mid_right = ((FR[0]+RR[0])/2.0, (FR[1]+RR[1])/2.0)
    mid_rear  = ((RR[0]+RL[0])/2.0, (RR[1]+RL[1])/2.0)
    mid_left  = ((RL[0]+FL[0])/2.0, (RL[1]+FL[1])/2.0)
    return [mid_front, mid_right, mid_rear, mid_left]


class ObstacleSpawner(Node):
    def __init__(self):
        super().__init__('static_obstacle_spawner', automatically_declare_parameters_from_overrides=True)

        # ---- Params ----
        host = self._get('carla_host','127.0.0.1'); port = int(self._get('carla_port',2000))
        rate = float(self._get('publish_rate_hz',10.0))
        self.infl = float(self._get('inflation_radius',0.0))
        cone_bp_name = 'static.prop.trafficcone01'
        # Read rectangles
        rect_params: Dict[str, rclpy.parameter.Parameter] = self.get_parameters_by_prefix('rects')
        rects_map: Dict[str, Dict[str, float]] = {}
        for k,p in rect_params.items():
            nm, attr = (k.split('.',1)+[''])[:2]
            rects_map.setdefault(nm, {})[attr] = p.value
        self.rects = []  # [(cx,cy,z,yaw_rad,L,W)]
        for nm in sorted(rects_map):
            r = rects_map[nm]
            self.rects.append((
                float(r.get('x',0.0)),
                float(r.get('y',0.0)),
                float(r.get('z',0.1)),
                math.radians(float(r.get('yaw',0.0))),
                float(r.get('length',4.5)),
                float(r.get('width',2.0)),
            ))
        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, 'car_box', 10)
        self.create_timer(1.0/max(1e-3,rate), self.publish_rect_boxes)

        if not self.rects:
            self.get_logger().warn("No rects.* provided; publishing empty car_box.")
            self.world=None; self.cones=[]; return

        # ---- CARLA connect ----
        client = carla.Client(host, port); client.set_timeout(6.0)
        for _ in range(30):
            try:
                self.world = client.get_world(); break
            except RuntimeError:
                self.get_logger().warn("CARLA not ready, retrying..."); time.sleep(1.0)
        if not hasattr(self,'world') or self.world is None:
            raise RuntimeError(f"Could not connect to CARLA at {host}:{port}")
        bp_lib = self.world.get_blueprint_library()
        try:
            cone_bp = bp_lib.find(cone_bp_name)
        except Exception:
            cand = bp_lib.filter('static.prop.trafficcone*') or bp_lib.filter('static.prop.*cone*')
            if not cand: raise RuntimeError("No cone blueprint found.")
            cone_bp = random.choice(cand)

        # ---- Spawn cones at 4 corners + 2 long-side mids per rect ----
        # ---- Spawn cones at 4 corners + 4 side mids per rect ----
        self.cones: List[carla.Actor] = []
        for (cx,cy,cz,yaw,L,W) in self.rects:
            corners = rect_corners((cx,cy), yaw, L, W)  # FL, FR, RR, RL
            mids = side_midpoints(corners)              # front, right, rear, left
            pts = corners + mids
            for (px,py) in pts:
                tf = carla.Transform(carla.Location(x=px, y=py, z=cz),
                                    carla.Rotation(pitch=0.0, yaw=math.degrees(yaw), roll=0.0))
                actor = self.world.try_spawn_actor(cone_bp, tf)
                if actor: self.cones.append(actor)
                else: self.get_logger().warn(f"Failed to spawn cone at ({px:.2f},{py:.2f},{cz:.2f})")


    def _get(self, name, default):  # tiny param helper
        return self.get_parameter(name).value if self.has_parameter(name) else default

    def publish_rect_boxes(self):
        # Publish inflated rectangle corners, NOT cone bboxes
        flat: List[float] = []
        for (cx,cy,_cz,yaw,L,W) in self.rects:
            corners = rect_corners((cx,cy), yaw, L+2*self.infl, W+2*self.infl)
            # order FL, FR, RR, RL -> 8 floats
            flat += [corners[0][0],corners[0][1], corners[1][0],corners[1][1],
                     corners[2][0],corners[2][1], corners[3][0],corners[3][1]]
        N = len(self.rects)
        msg = Float32MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[MultiArrayDimension(label='cars', size=N, stride=N*8),
                 MultiArrayDimension(label='coords', size=8, stride=8)],
            data_offset=0)
        msg.data = flat
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            for a in getattr(self,'cones',[]): a.destroy()
        except Exception: pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    n = ObstacleSpawner()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        n.destroy_node(); rclpy.shutdown()
