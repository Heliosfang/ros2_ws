#!/usr/bin/env python3
import os, math, bisect
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# ================== CONFIG ==================
# Provide 3+ waypoints; the path connects them in order with smooth curves at corners.
POINTS = [
    (-164.732162, -95.141876), #-231.7088012695312, -95.2505874633789
    (-273.014831542969, -95.141876),
    (-273.014831542969, 95.141876),
    (-164.732162,  95.141876),
    # (-110.963745, -68.25939178466797),  # add more to see multiple smooth corners
    # (-110.963745, -14.789828300476074),  # add more to see multiple smooth corners
]

# Dense pre-sampling for smooth resampling (per piece)
LINE_SAMPLES   = 200
CURVE_SAMPLES  = 600

# Corner trimming: where to start/stop the smooth curve along each adjacent line
# Use either TRIM_FRAC (fraction of adjacent segment length) or TRIM_DIST (meters).
TRIM_FRAC = 0.20     # e.g., use 20% of each adjacent segment length (capped)
TRIM_DIST = 50     # or set to a fixed distance (e.g., 8.0). Keep one as None.

# Tangent scaling at the curve ends; >1 gives more "convex" (bulge)
BULGE_GAIN = 1.30    # try 1.0 .. 2.0

# Even spacing: choose one of these (set the other to None)
EVEN_STEP   = 0.50   # meters between points
EVEN_POINTS = None   # or a fixed total number of points

# Output locations
OUTPUT_FILENAME = "bspline.txt"
PKG_NAME = "path_publisher"  # optional share copy if available
# ============================================

def _len(v): return math.hypot(v[0], v[1])
def _normalize(v, eps=1e-12):
    n = _len(v);  return (0.0, 0.0) if n < eps else (v[0]/n, v[1]/n)
def _scale(v, s): return (v[0]*s, v[1]*s)
def _add(a,b): return (a[0]+b[0], a[1]+b[1])
def _sub(a,b): return (a[0]-b[0], a[1]-b[1])

def sample_line(A, B, samples):
    n = max(2, int(samples)); pts=[]
    for i in range(n):
        t = i/(n-1)
        pts.append((A[0] + (B[0]-A[0])*t, A[1] + (B[1]-A[1])*t))
    return pts

# Quintic Hermite with P, P', P'' constraints at both ends
def sample_quintic_hermite(Pa, Pb, Va, Vb, Aa, Ab, samples):
    n = max(2, int(samples)); pts=[]
    for i in range(n):
        t = i/(n-1)
        t2 = t*t; t3 = t2*t; t4 = t3*t; t5 = t4*t
        # Position basis (standard quintic Hermite blending)
        h00 =  1 - 10*t3 + 15*t4 -  6*t5
        h10 =      t  -  6*t3 +  8*t4 -  3*t5
        h20 = 0.5*t2 -  1.5*t3 + 1.5*t4 - 0.5*t5
        h01 =     10*t3 - 15*t4 +  6*t5
        h11 =     -4*t3 +  7*t4 -  3*t5
        h21 = 0.5*t3 -    t4 + 0.5*t5

        x = (h00*Pa[0] + h10*Va[0] + h20*Aa[0] +
             h01*Pb[0] + h11*Vb[0] + h21*Ab[0])
        y = (h00*Pa[1] + h10*Va[1] + h20*Aa[1] +
             h01*Pb[1] + h11*Vb[1] + h21*Ab[1])
        pts.append((x, y))
    return pts

def cumulative_arclength(path):
    s = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        s.append(s[-1] + math.hypot(dx, dy))
    return s

def lerp(p, q, t):
    return (p[0] + (q[0]-p[0])*t, p[1] + (q[1]-p[1])*t)

def even_resample_by_arclength(path, step=None, n_points=None, eps=1e-12):
    assert (step is None) ^ (n_points is None), "Specify exactly one of step or n_points"
    if len(path) < 2: return path[:]
    s = cumulative_arclength(path); L = s[-1]
    if L <= eps: return [path[0]] * (n_points if n_points else 2)

    if step is not None:
        m = max(2, int(round(L/step)) + 1)
        targets = [i*(L/(m-1)) for i in range(m)]
    else:
        m = max(2, int(n_points))
        targets = [i*(L/(m-1)) for i in range(m)]

    out=[]; j=0
    for st in targets:
        if st >= s[-1]:
            out.append(path[-1]); continue
        j = bisect.bisect_right(s, st, j) - 1
        seg = s[j+1] - s[j]
        t = 0.0 if seg <= eps else (st - s[j]) / seg
        out.append(lerp(path[j], path[j+1], t))
    return out

def build_smooth_polyline(points):
    """Return a dense path that follows straight lines, replacing each interior corner
       with a quintic curve that is tangent-aligned to both adjoining segments."""
    assert len(points) >= 3, "Need at least 3 points for a smooth cornered path"

    dense = []
    N = len(points)

    # Helper to add (and avoid duplicates)
    def extend_no_dup(seq):
        if not seq: return
        if not dense:
            dense.extend(seq)
        else:
            if dense[-1] == seq[0]:
                dense.extend(seq[1:])
            else:
                dense.extend(seq)

    for i in range(N-1):
        A = points[i]
        B = points[i+1]
        AB = _sub(B, A)
        lenAB = _len(AB)
        uAB = _normalize(AB)

        # First and last segments are handled mostly straight, corners treated when i in [1..N-2]
        if 0 < i < N-1:
            # We are entering a corner at points[i]
            # Previous segment is (points[i-1] -> points[i]) and next is (points[i] -> points[i+1])
            prev = points[i-1]
            nextp = points[i+1]

            v_prev = _sub(points[i], prev);     L_prev = _len(v_prev); u_prev = _normalize(v_prev)
            v_next = _sub(nextp, points[i]);    L_next = _len(v_next); u_next = _normalize(v_next)

            # Choose trim distances on both sides
            if TRIM_DIST is not None:
                t_prev = min(TRIM_DIST, 0.45 * L_prev)
                t_next = min(TRIM_DIST, 0.45 * L_next)
            else:
                frac = max(0.0, min(0.45, TRIM_FRAC if TRIM_FRAC is not None else 0.2))
                t_prev = frac * L_prev
                t_next = frac * L_next

            # Define the curve endpoints near the corner
            Qm = _sub(points[i], _scale(u_prev, t_prev))  # along previous segment towards previous point
            Qp = _add(points[i], _scale(u_next, t_next))  # along next segment away from corner

            # 1) Add straight line from last placed point up to Qm
            if i == 1:
                # Add initial straight from P0 to Qm
                extend_no_dup(sample_line(points[0], Qm, LINE_SAMPLES))
            else:
                # From previous curve end to Qm (the previous loop already ended at Qm via curve)
                extend_no_dup(sample_line(dense[-1], Qm, LINE_SAMPLES))

            # 2) Add smooth curve from Qm -> Qp with tangents along u_prev and u_next
            chord = _sub(Qp, Qm); Lchord = _len(chord)
            vmag = BULGE_GAIN * Lchord  # scale tangents by chord length
            V0 = _scale(u_prev, vmag)
            V1 = _scale(u_next, vmag)
            A0 = (0.0, 0.0)  # zero curvature at joins
            A1 = (0.0, 0.0)
            extend_no_dup(sample_quintic_hermite(Qm, Qp, V0, V1, A0, A1, CURVE_SAMPLES))

            # 3) If this is the last corner, finish with straight to P_{N-1}
            if i == N-2:
                extend_no_dup(sample_line(Qp, points[-1], LINE_SAMPLES))

        else:
            # i == 0: only add initial straight up to possible first corner handled when i==1
            # i == N-2: handled in the branch above when we add the last straight after the final curve
            if i == 0 and N == 2:
                # Exactly two points: just a single straight line path
                extend_no_dup(sample_line(A, B, LINE_SAMPLES))
            # Otherwise start segment will be added when first corner is processed

    return dense

def main():
    # 1) Build dense piecewise path with smooth corners
    path_dense = build_smooth_polyline(POINTS)

    # 2) Evenly resample the whole path by arc length
    path = even_resample_by_arclength(path_dense, step=EVEN_STEP, n_points=EVEN_POINTS)

    # === Write files ===
    def write_points(fp):
        with open(fp, "w") as f:
            f.write("# x y\n")
            for x, y in path:
                f.write(f"{x:.6f} {y:.6f}\n")

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    pkg_root_src = os.path.abspath(os.path.join(script_dir, ".."))
    out_src_dir  = os.path.join(pkg_root_src, "output")
    os.makedirs(out_src_dir, exist_ok=True)
    out_src_file = os.path.join(out_src_dir, OUTPUT_FILENAME)
    write_points(out_src_file)
    print(f"Wrote {len(path)} evenly spaced points to {out_src_file}")

    # Optional: also copy to package share if available
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory(PKG_NAME)
        out_share_dir = os.path.join(share_dir, "output")
        os.makedirs(out_share_dir, exist_ok=True)
        out_share_file = os.path.join(out_share_dir, OUTPUT_FILENAME)
        write_points(out_share_file)
        print(f"Copied path to {out_share_file}")
    except Exception:
        pass

    # === Plot ===
    plt.figure(figsize=(7,5))
    
    home_dir = os.path.expanduser("~")
    pcd_file = os.path.join(home_dir, "Carla-0916", "HDMaps", "Town05.pcd")
    pcd = o3d.io.read_point_cloud(pcd_file)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(-285, -100, -1),   # x_min, y_min, z_min
    max_bound=(-150, 100,  1)    # x_max, y_max, z_max
    )       
    pcd = pcd.crop(bbox)
    # --- Downsample to reduce number of points ---
    voxel_size = 0.8  # meters, adjust as needed
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(pcd.points)
    plt.scatter(pts[:,0], pts[:,1], s=0.1, c='gray', alpha=0.5)
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    cx, cy = zip(*POINTS)
    plt.plot(xs, ys, label="Even-arc-length path with smooth corners")
    # Show the underlying polyline
    plt.plot(cx, cy, "k--", linewidth=1, label="Original polyline")
    plt.plot(cx, cy, "ro", label="Waypoints")
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title("Multi-line path with quintic corner smoothing")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()