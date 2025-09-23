#!/usr/bin/env python3
import math
from collections import deque
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from visualization_msgs.msg import Marker, MarkerArray

# ----------------- helpers -----------------
def idx_to_rc(index: int, width: int) -> Tuple[int, int]:
    return index // width, index % width

def rc_to_xy(row: int, col: int, res: float, ox: float, oy: float) -> Tuple[float, float]:
    # cell center coordinates
    return ox + (col + 0.5) * res, oy + (row + 0.5) * res

def monotonic_chain_convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotonic chain. Returns CCW hull without repeating first point."""
    if len(points) <= 1:
        return points[:]
    pts = sorted(points)
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def rotate_point(px: float, py: float, c: float, s: float) -> Tuple[float, float]:
    # rotate by angle whose cos=c, sin=s
    return (px * c + py * s, -px * s + py * c)

def inv_rotate_point(px: float, py: float, c: float, s: float) -> Tuple[float, float]:
    # rotate back by -theta
    return (px * c - py * s, px * s + py * c)

def min_area_rect(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Minimum-area bounding rectangle of a set of 2D points.
    Returns 4 points (CCW). If degenerate, returns a small box.
    """
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        x, y = points[0]
        eps = 1e-3
        return [(x-eps,y-eps),(x+eps,y-eps),(x+eps,y+eps),(x-eps,y+eps)]

    hull = monotonic_chain_convex_hull(points)
    if len(hull) == 2:
        # line segment → skinny rectangle aligned with the segment
        (x1,y1),(x2,y2) = hull
        dx, dy = x2-x1, y2-y1
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy/L, dx/L
        w = 1e-3
        return [(x1+nx*w, y1+ny*w), (x2+nx*w, y2+ny*w),
                (x2-nx*w, y2-ny*w), (x1-nx*w, y1-ny*w)]

    best_area = float('inf')
    best_rect = None

    # try each edge orientation
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i+1) % len(hull)]
        ex, ey = x2 - x1, y2 - y1
        elen = math.hypot(ex, ey)
        if elen < 1e-9:
            continue
        c, s = ex/elen, ey/elen  # cos, sin of edge angle

        # rotate all hull points to this frame
        xs, ys = [], []
        for (px, py) in hull:
            rx, ry = rotate_point(px, py, c, s)
            xs.append(rx); ys.append(ry)

        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        area = (maxx - minx) * (maxy - miny)

        if area < best_area:
            # rectangle corners in rotated frame (CCW)
            rect_r = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
            # rotate back
            rect_w = [inv_rotate_point(rx, ry, c, s) for (rx, ry) in rect_r]
            best_area = area
            best_rect = rect_w

    if best_rect is None:
        return []
    # reorder to CCW starting from corner with smallest (x+y) for consistency
    idx0 = min(range(4), key=lambda i: (best_rect[i][0] + best_rect[i][1]))
    rect = [best_rect[(idx0 + k) % 4] for k in range(4)]
    # ensure CCW (positive area)
    def signed_area(poly):
        A = 0.0
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % 4]
            A += x1*y2 - x2*y1
        return 0.5*A
    if signed_area(rect) < 0:
        rect = [rect[0], rect[3], rect[2], rect[1]]
    return rect

# ----------------- node -----------------
class CostmapToRectangles(Node):
    def __init__(self):
        super().__init__('costmap_to_rectangles')

        # Parameters
        self.declare_parameter('costmap_topic', '/local_costmap/costmap')
        self.declare_parameter('occupied_threshold', 50)     # 0..100
        self.declare_parameter('min_cluster_cells', 6)
        self.declare_parameter('max_rectangles', 200)
        self.declare_parameter('publish_rate', 5.0)          # Hz
        self.declare_parameter('frame_id', '')

        self.costmap_topic = self.get_parameter('costmap_topic').get_parameter_value().string_value
        self.occ_thr = int(self.get_parameter('occupied_threshold').get_parameter_value().integer_value)
        self.min_cluster_cells = int(self.get_parameter('min_cluster_cells').get_parameter_value().integer_value)
        self.max_rectangles = int(self.get_parameter('max_rectangles').get_parameter_value().integer_value)
        self.publish_rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)
        self.forced_frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.last_publish_time = self.get_clock().now()

        self.sub = self.create_subscription(OccupancyGrid, self.costmap_topic, self.cb_costmap, 10)
        self.pub_rects = self.create_publisher(Float32MultiArray, 'rectangles', 10)
        self.pub_markers = self.create_publisher(MarkerArray, 'rectangle_markers', 10)

        self.get_logger().info(f"Subscribing to {self.costmap_topic}")

    def cb_costmap(self, msg: OccupancyGrid):
        now = self.get_clock().now()
        if (now - self.last_publish_time).nanoseconds < (1.0 / max(self.publish_rate, 1e-3)) * 1e9:
            return

        width, height = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        frame = self.forced_frame_id if self.forced_frame_id else msg.header.frame_id

        occ = [v >= self.occ_thr for v in msg.data]
        visited = [False] * (width * height)
        clusters: List[List[Tuple[int, int]]] = []

        def neighbors(r, c):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        yield rr, cc

        # flood-fill clusters
        for idx, is_occ in enumerate(occ):
            if not is_occ or visited[idx]:
                continue
            r, c = idx_to_rc(idx, width)
            q = deque([(r, c)])
            visited[idx] = True
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for r2, c2 in neighbors(rr, cc):
                    id2 = r2 * width + c2
                    if not visited[id2] and occ[id2]:
                        visited[id2] = True
                        q.append((r2, c2))
                        comp.append((r2, c2))
            if len(comp) >= self.min_cluster_cells:
                clusters.append(comp)
            if len(clusters) >= self.max_rectangles:
                break

        # clusters -> min-area rectangles (world coords)
        rectangles: List[List[Tuple[float, float]]] = []
        for comp in clusters:
            pts = [rc_to_xy(r, c, res, ox, oy) for (r, c) in comp]
            rect = min_area_rect(pts)
            if len(rect) == 4:
                rectangles.append(rect)

        # ---------------- publish Float32MultiArray ----------------
        msg_out = Float32MultiArray()
        # layout: [N, 4, 2]
        dims = [
            MultiArrayDimension(label='rect', size=len(rectangles), stride=len(rectangles) * 4 * 2),
            MultiArrayDimension(label='corner', size=4, stride=4 * 2),
            MultiArrayDimension(label='xy', size=2, stride=2),
        ]
        msg_out.layout = MultiArrayLayout(dim=dims, data_offset=0)
        data = []
        for rect in rectangles:
            # CCW, 4 corners: (x,y)
            for (x, y) in rect:
                data.extend([float(x), float(y)])
        msg_out.data = data
        self.pub_rects.publish(msg_out)

        # ---------------- RViz markers (optional, nice for debugging) ----------------
        markers = MarkerArray()
        clear = Marker()
        clear.header = Header(stamp=now.to_msg(), frame_id=frame)
        clear.ns = "costmap_rectangles"
        clear.id = 0
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for i, rect in enumerate(rectangles, start=1):
            m = Marker()
            m.header = Header(stamp=now.to_msg(), frame_id=frame)
            m.ns = "costmap_rectangles"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = max(0.02, res * 0.5)  # line width
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.pose.orientation.w = 1.0
            loop = rect + [rect[0]]
            m.points = [self._pt(x, y) for (x, y) in loop]
            markers.markers.append(m)

        self.pub_markers.publish(markers)
        self.last_publish_time = now

    @staticmethod
    def _pt(x, y):
        from geometry_msgs.msg import Point
        p = Point()
        p.x, p.y, p.z = float(x), float(y), 0.0
        return p

def main():
    rclpy.init()
    node = CostmapToRectangles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
