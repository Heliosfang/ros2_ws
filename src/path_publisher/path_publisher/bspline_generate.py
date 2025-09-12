#!/usr/bin/env python3
import os, math
import matplotlib.pyplot as plt

# ====== CONFIG (edit points/samples as needed) ======
P1, P2, P3, P4 = (299.399994, 55.840000), (166.70510864257812, 55.840000), (154.17022705078125, 43.39997863769531), (154.17022705078125, 2.9970593452453613)
LINE1_SAMPLES, CURVE_SAMPLES, LINE3_SAMPLES = 50, 150, 50
TANGENT_GAIN = 0.5
# ====================================================

def _len(v): return math.hypot(v[0], v[1])
def _normalize(v, eps=1e-9):
    n = _len(v);  return (0.0, 0.0) if n < eps else (v[0]/n, v[1]/n)

def sample_line(A, B, samples):
    n = max(2, int(samples)); pts=[]
    for i in range(n):
        t = i/(n-1)
        pts.append((A[0] + (B[0]-A[0])*t, A[1] + (B[1]-A[1])*t))
    return pts

def sample_cubic_hermite(Pa, Pb, Ta, Tb, samples):
    n = max(2, int(samples)); pts=[]
    for i in range(n):
        t = i/(n-1)
        h00 =  2*t**3 - 3*t**2 + 1
        h10 =      t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 =      t**3 -   t**2
        x = h00*Pa[0] + h10*Ta[0] + h01*Pb[0] + h11*Tb[0]
        y = h00*Pa[1] + h10*Ta[1] + h01*Pb[1] + h11*Tb[1]
        pts.append((x, y))
    return pts

def main():
    # Build the path (line → spline → line)
    seg1 = sample_line(P1, P2, LINE1_SAMPLES)
    seg3 = sample_line(P3, P4, LINE3_SAMPLES)
    mag = TANGENT_GAIN * _len((P3[0]-P2[0], P3[1]-P2[1]))
    T2 = tuple(x*mag for x in _normalize((P2[0]-P1[0], P2[1]-P1[1])))
    T3 = tuple(x*mag for x in _normalize((P4[0]-P3[0], P4[1]-P3[1])))
    seg2 = sample_cubic_hermite(P2, P3, T2, T3, CURVE_SAMPLES)
    path = seg1[:-1] + seg2[:-1] + seg3

    # 1) Write to source tree: <pkg_root>/output/bspline.txt
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    pkg_root_src = os.path.abspath(os.path.join(script_dir, ".."))
    out_src_dir  = os.path.join(pkg_root_src, "output")
    os.makedirs(out_src_dir, exist_ok=True)
    out_src_file = os.path.join(out_src_dir, "bspline.txt")

    def write_points(fp):
        with open(fp, "w") as f:
            f.write("# x y\n")
            for x, y in path:
                f.write(f"{x:.6f} {y:.6f}\n")

    write_points(out_src_file)
    print(f"Wrote {len(path)} points to {out_src_file}")

    # 2) If ROS env is sourced and the package is installed, also copy into share/
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("path_publisher")
        out_share_dir = os.path.join(share_dir, "output")
        os.makedirs(out_share_dir, exist_ok=True)
        out_share_file = os.path.join(out_share_dir, "bspline.txt")
        write_points(out_share_file)
        print(f"Copied path to {out_share_file}")
    except Exception:
        # No ROS env / not installed yet – ignore
        pass

    # Plot
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    cx, cy = zip(P1, P2, P3, P4)
    plt.figure(figsize=(7,5))
    plt.plot(xs, ys, label="Line → Hermite spline → Line")
    plt.plot([P1[0],P2[0]],[P1[1],P2[1]], "k--", linewidth=1, label="Line segments")
    plt.plot([P3[0],P4[0]],[P3[1],P4[1]], "k--", linewidth=1)
    plt.plot(cx, cy, "ro", label="Given points")
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title("Path: straight–spline–straight (C¹ at joins)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
