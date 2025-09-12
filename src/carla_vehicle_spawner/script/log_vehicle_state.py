#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import time
import carla

# -------------------- helpers --------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def resolve_vehicle(world, actor_id: int, role_name: str, blueprint: str, bp_index: int):
    actors = world.get_actors().filter('vehicle.*')
    if actor_id > 0:
        v = actors.find(actor_id)
        if v is not None:
            return v
    if role_name:
        matches = [a for a in actors if a.attributes.get('role_name', '') == role_name]
        if matches:
            return matches[0]
    if blueprint:
        matches = actors.filter(blueprint)
        if not matches:
            matches = [a for a in actors if a.type_id == blueprint]
        if matches:
            idx = max(0, min(bp_index, len(matches) - 1))
            return matches[idx]
    return actors[0] if len(actors) > 0 else None

def world_to_body(vx_w, vy_w, yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # Rotate world -> body (x forward, y left)
    vx_b =  c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b

def plausible_lf_lr(lf, lr):
    wb = lf + lr
    return (lf > 0.0 and lr > 0.0 and 1.5 <= wb <= 4.5)

# -------------------- lf/lr computation --------------------

def lf_lr_from_blueprint(vehicle, com_x_m):
    """Prefer blueprint wheel_base if available."""
    try:
        bp = vehicle.get_world().get_blueprint_library().find(vehicle.type_id)
        if bp.has_attribute("wheel_base"):
            wb = float(bp.get_attribute("wheel_base").as_string())
            # Split around CoM.x (meters). If CoM≈0, this becomes wb/2 each.
            lf = wb / 2.0 - com_x_m
            lr = wb - lf
            # Ensure positive and plausible
            if lf <= 0.0 or lr <= 0.0:
                # If CoM.x was extreme, fall back to symmetric split
                lf = lr = wb / 2.0
            if plausible_lf_lr(lf, lr):
                return lf, lr, wb, "blueprint wheel_base"
    except Exception:
        pass
    return None, None, None, None

def lf_lr_from_wheels(physics, vehicle, com_x_m):
    """Try wheel positions; check meters vs centimeters→meters and choose plausible."""
    wheels = getattr(physics, "wheels", None)
    if not wheels or len(wheels) < 2:
        return None, None, "wheels (unavailable)"

    xs_m_A, xs_m_B = [], []   # A: assume meters, B: cm->m
    front_A, rear_A, front_B, rear_B = [], [], [], []

    for w in wheels:
        x_raw = float(getattr(w.position, "x", 0.0))
        xA = x_raw
        xB = x_raw * 0.01  # cm -> m hypothesis
        xs_m_A.append(xA)
        xs_m_B.append(xB)
        steerable = float(getattr(w, "max_steer_angle", 0.0)) > 1e-3
        if steerable:
            front_A.append(xA); front_B.append(xB)
        else:
            rear_A.append(xA);  rear_B.append(xB)

    # If steer tags missing, use extremes
    if not front_A and xs_m_A:
        front_A = [max(xs_m_A)]; front_B = [max(xs_m_B)]
    if not rear_A and xs_m_A:
        rear_A  = [min(xs_m_A)]; rear_B  = [min(xs_m_B)]

    if not front_A or not rear_A:
        return None, None, "wheels (insufficient)"

    avg = lambda v: sum(v)/len(v)

    lfA = avg(front_A) - com_x_m
    lrA = com_x_m - avg(rear_A)
    lfB = avg(front_B) - com_x_m
    lrB = com_x_m - avg(rear_B)

    A_ok = plausible_lf_lr(lfA, lrA)
    B_ok = plausible_lf_lr(lfB, lrB)

    # If both plausible, prefer the one closer to blueprint wheel_base (if present)
    if A_ok and B_ok:
        try:
            bp = vehicle.get_world().get_blueprint_library().find(vehicle.type_id)
            if bp.has_attribute("wheel_base"):
                wb = float(bp.get_attribute("wheel_base").as_string())
                errA = abs((lfA + lrA) - wb)
                errB = abs((lfB + lrB) - wb)
                if errA <= errB: return lfA, lrA, "wheels (meters)"
                else:            return lfB, lrB, "wheels (cm→m)"
        except Exception:
            pass
        return lfA, lrA, "wheels (meters)"  # fallback preference

    if A_ok: return lfA, lrA, "wheels (meters)"
    if B_ok: return lfB, lrB, "wheels (cm→m)"

    return None, None, "wheels (implausible)"

def lf_lr_from_bbox(vehicle):
    """Fallback: half overall length (not axle distances)."""
    half_len = float(vehicle.bounding_box.extent.x)
    return half_len, half_len, "bbox half-length (fallback)"

def compute_lf_lr(vehicle):
    """
    Final resolver: blueprint wheel_base -> wheels -> bbox.
    Returns (lf, lr, source_str).
    """
    # CoM.x in meters (if available)
    com_x = 0.0
    try:
        phys = vehicle.get_physics_control()
        com_x = float(getattr(phys.center_of_mass, "x", 0.0))
    except Exception:
        phys = None

    # 1) Blueprint
    lf, lr, wb, src = lf_lr_from_blueprint(vehicle, com_x)
    if lf is not None:
        return lf, lr, f"{src} (wb={wb:.3f} m)"

    # 2) Wheels (need physics)
    if phys is not None:
        lf, lr, src = lf_lr_from_wheels(phys, vehicle, com_x)
        if lf is not None:
            return lf, lr, src

    # 3) Bounding box
    lf, lr, src = lf_lr_from_bbox(vehicle)
    return lf, lr, src

# -------------------- mass & Iz --------------------

def get_mass_and_Iz(vehicle):
    """
    Returns (mass, Iz, iz_estimated: bool).
    Iz from physics.moment_of_inertia.z if available; else estimate via bbox dims.
    """
    iz_est = False
    mass = 0.0
    Iz = 0.0
    try:
        phys = vehicle.get_physics_control()
        mass = float(getattr(phys, "mass", 0.0))
        moi = getattr(phys, "moment_of_inertia", None)
        if moi is not None and hasattr(moi, "z"):
            Iz = float(moi.z)
        else:
            bb = vehicle.bounding_box
            L = 2.0 * float(bb.extent.x)
            W = 2.0 * float(bb.extent.y)
            Iz = mass * (L*L + W*W) / 12.0
            iz_est = True
    except Exception:
        bb = vehicle.bounding_box
        L = 2.0 * float(bb.extent.x)
        W = 2.0 * float(bb.extent.y)
        Iz = mass * (L*L + W*W) / 12.0
        iz_est = True
    return mass, Iz, iz_est

# -------------------- main logger --------------------

def main():
    parser = argparse.ArgumentParser(description="Log CARLA vehicle state to CSV (non-ROS).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.1, help="sample period (s), default 0.1s (10 Hz)")

    # Vehicle selection precedence: actor_id > role_name > blueprint
    parser.add_argument("--actor-id", type=int, default=0)
    parser.add_argument("--role-name", default="ego")
    parser.add_argument("--blueprint", default="vehicle.lincoln.mkz")
    parser.add_argument("--bp-index", type=int, default=0)

    parser.add_argument("--out", default="carla_vehicle_log.csv")
    parser.add_argument("--max-steer-rad", type=float, default=0.6,
                        help="assumed max steering angle (rad) for converting steer_norm -> steering_angle_rad")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing CSV instead of appending")

    args = parser.parse_args()

    # Connect
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)

    world = None
    for _ in range(60):
        try:
            world = client.get_world()
            break
        except RuntimeError:
            time.sleep(1.0)
    if world is None:
        print(f"ERROR: Could not connect to CARLA at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)

    # Resolve vehicle with retries
    vehicle = None
    for i in range(120):  # up to ~60s
        vehicle = resolve_vehicle(world, args.actor_id, args.role_name, args.blueprint, args.bp_index)
        if vehicle is not None:
            break
        if i % 4 == 0:
            print("Waiting for vehicle to appear...")
        time.sleep(0.5)
    if vehicle is None:
        print("ERROR: No vehicle found to track.", file=sys.stderr)
        sys.exit(2)

    print(f"Logging vehicle id={vehicle.id}, type={vehicle.type_id}, role_name='{vehicle.attributes.get('role_name','')}'")

    # Physics snapshot
    mass, Iz, iz_estimated = get_mass_and_Iz(vehicle)
    lf, lr, lf_lr_src = compute_lf_lr(vehicle)

    out_path = os.path.abspath(args.out)
    mode = "w" if args.overwrite or not os.path.exists(out_path) else "a"
    first_write = (mode == "w")
    f = open(out_path, mode, newline="")
    writer = csv.writer(f)

    if first_write:
        writer.writerow(["t","x","y","phi_rad","vx_body","vy_body","omega_rad_s","throttle","steering_angle_rad"])
        f.flush()

    print(f"Writing CSV to: {out_path}")

    # Log loop
    count = 0
    dt = max(0.01, float(args.dt))
    max_angle = max(1e-3, float(args.max_steer_rad))

    try:
        next_t = time.time()
        while True:
            snap = world.get_snapshot()
            t_sim = getattr(snap.timestamp, "elapsed_seconds", 0.0)

            tf = vehicle.get_transform()
            loc = tf.location
            yaw_rad = math.radians(float(tf.rotation.yaw))

            v_world = vehicle.get_velocity()
            vx_b, vy_b = world_to_body(float(v_world.x), float(v_world.y), yaw_rad)

            omega_rad_s = math.radians(float(vehicle.get_angular_velocity().z))

            try:
                ctrl = vehicle.get_control()
                throttle = float(ctrl.throttle)   # [0..1]
                steer_norm = float(ctrl.steer)    # [-1..1], +right(clockwise), -left(CCW)
                steering_angle_rad = clamp(steer_norm, -1.0, 1.0) * max_angle
            except Exception:
                throttle = 0.0
                steering_angle_rad = 0.0

            writer.writerow([
                f"{t_sim:.6f}",
                f"{loc.x:.6f}", f"{loc.y:.6f}", f"{yaw_rad:.6f}",
                f"{vx_b:.6f}", f"{vy_b:.6f}", f"{omega_rad_s:.6f}",
                f"{throttle:.6f}", f"{steering_angle_rad:.6f}"
            ])
            count += 1
            if count % 10 == 0:
                f.flush()
            print(f"\rSaved rows: {count}", end="", flush=True)

            # pace to dt
            next_t += dt
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.time()

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass

        # Build summary note
        est_tags = []
        if iz_estimated:
            est_tags.append("Iz estimated")
        if lf_lr_src.startswith("bbox"):
            est_tags.append("lf/lr from bbox")
        summary_note = "" if not est_tags else f" [{'; '.join(est_tags)}]"

        print(f"Physics summary: mass={mass:.2f} kg, lf={lf:.3f} m, lr={lr:.3f} m, Iz={Iz:.2f} kg*m^2")
        print(f"lf/lr source: {lf_lr_src}{summary_note}")
        print(f"Done. Total rows saved: {count}\nFile: {out_path}")

# --------------------

if __name__ == "__main__":
    main()
