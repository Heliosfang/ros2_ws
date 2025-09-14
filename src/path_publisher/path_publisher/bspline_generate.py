#!/usr/bin/env python3
import os, math, bisect
import matplotlib.pyplot as plt

# ====== CONFIG ======
# P1, P2, P3, P4 = (299.399994, 55.840000), (170.70510864257812, 55.840000), (156.5622705078125, 43.29997863769531), (154.17022705078125, 2.9970593452453613)
P1, P2, P3, P4 = (-164.732162, -95.141876), (-231.7088012695312, -95.2505874633789), (-273.3805847167969, -50.73670959472656), (-273.3805847167969, 55.14298248291016)

# Dense pre-sampling for smooth resampling
LINE1_SAMPLES, CURVE_SAMPLES, LINE3_SAMPLES = 200, 600, 200

# Tangent scaling at the curve ends; >1 gives more "convex" (bulge)
BULGE_GAIN = 1.00   # try 1.0 .. 2.0

# Even spacing: choose one of these (set the other to None)
EVEN_STEP   = 0.50   # meters between points
EVEN_POINTS = None   # or a fixed total number of points
# ====================

def _len(v): return math.hypot(v[0], v[1])
def _normalize(v, eps=1e-12):
    n = _len(v);  return (0.0, 0.0) if n < eps else (v[0]/n, v[1]/n)
def _scale(v, s): return (v[0]*s, v[1]*s)
def _add(a,b): return (a[0]+b[0], a[1]+b[1])

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
        # Position basis (see standard quintic Hermite)
        h00 = 1 - 10*t3 + 15*t4 - 6*t5
        h10 = t - 6*t3 + 8*t4 - 3*t5
        h20 = 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5
        h01 = 10*t3 - 15*t4 + 6*t5
        h11 = -4*t3 + 7*t4 - 3*t5
        h21 = 0.5*t3 - t4 + 0.5*t5

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

def main():
    # Directions along lines
    d12 = _normalize((P2[0]-P1[0], P2[1]-P1[1]))
    d34 = _normalize((P4[0]-P3[0], P4[1]-P3[1]))

    # Use chord length as a scale for endpoint velocities (first derivatives)
    L23 = _len((P3[0]-P2[0], P3[1]-P2[1]))
    vmag = BULGE_GAIN * L23

    # End constraints for the curve: tangent follows the adjoining lines; curvature = 0 at both ends
    V2 = _scale(d12, vmag)   # P'(0)
    V3 = _scale(d34, vmag)   # P'(1)
    A2 = (0.0, 0.0)          # P''(0) = 0  → zero curvature at join with straight line
    A3 = (0.0, 0.0)          # P''(1) = 0

    # Build segments
    seg1 = sample_line(P1, P2, LINE1_SAMPLES)
    seg3 = sample_line(P3, P4, LINE3_SAMPLES)
    seg2 = sample_quintic_hermite(P2, P3, V2, V3, A2, A3, CURVE_SAMPLES)

    # Join (avoid duplicates) and resample evenly by arc length
    path_dense = seg1[:-1] + seg2[1:-1] + seg3[1:]
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
    out_src_file = os.path.join(out_src_dir, "bspline.txt")
    write_points(out_src_file)
    print(f"Wrote {len(path)} evenly spaced points to {out_src_file}")

    # Optional: also copy to package share if available
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("path_publisher")
        out_share_dir = os.path.join(share_dir, "output")
        os.makedirs(out_share_dir, exist_ok=True)
        out_share_file = os.path.join(out_share_dir, "bspline.txt")
        write_points(out_share_file)
        print(f"Copied path to {out_share_file}")
    except Exception:
        pass

    # === Plot ===
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    cx, cy = zip(P1, P2, P3, P4)
    plt.figure(figsize=(7,5))
    plt.plot(xs, ys, label="Even-arc-length resampled path (quintic)")
    plt.plot([P1[0],P2[0]],[P1[1],P2[1]], "k--", linewidth=1, label="Line segments")
    plt.plot([P3[0],P4[0]],[P3[1],P4[1]], "k--", linewidth=1)
    plt.plot(cx, cy, "ro", label="Given points")
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title("Line–Quintic–Line (G²-like joins, adjustable bulge)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
