#!/usr/bin/env python3
import argparse
import math
import sys
import time
import carla

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
            idx = max(0, min(bp_index, len(matches)-1))
            return matches[idx]
    return actors[0] if len(actors) > 0 else None

def plausible_lf_lr(lf, lr):
    wb = lf + lr
    return (lf > 0.0 and lr > 0.0 and 1.5 <= wb <= 4.5)

def compute_lf_lr_from_wheels(vehicle):
    """
    Try wheel positions as meters; if not plausible, try centimeters->meters.
    Returns (lf, lr, source_str).
    """
    phys = vehicle.get_physics_control()
    com_x_m = float(getattr(phys.center_of_mass, "x", 0.0))
    wheels = getattr(phys, "wheels", None)
    if not wheels or len(wheels) < 2:
        raise RuntimeError("No wheel data available")

    # Collect front (steerable) and rear (non-steerable) x positions
    xs_m_guess_m = []
    xs_m_guess_cm = []
    front_m, rear_m = [], []
    front_cm, rear_cm = [], []

    for w in wheels:
        x_raw = float(getattr(w.position, "x", 0.0))
        # Hypothesis A: already meters
        x_mA = x_raw
        # Hypothesis B: centimeters -> meters
        x_mB = x_raw * 0.01

        xs_m_guess_m.append(x_mA)
        xs_m_guess_cm.append(x_mB)

        if float(getattr(w, "max_steer_angle", 0.0)) > 1e-3:
            front_m.append(x_mA); front_cm.append(x_mB)
        else:
            rear_m.append(x_mA);  rear_cm.append(x_mB)

    # If steer tagging absent, use extremes
    if not front_m and xs_m_guess_m:
        front_m = [max(xs_m_guess_m)]
        front_cm = [max(xs_m_guess_cm)]
    if not rear_m and xs_m_guess_m:
        rear_m = [min(xs_m_guess_m)]
        rear_cm = [min(xs_m_guess_cm)]

    def avg(v): return sum(v)/len(v)

    # Hypothesis A: meters
    lfA = avg(front_m) - com_x_m
    lrA = com_x_m - avg(rear_m)

    # Hypothesis B: centimeters
    lfB = avg(front_cm) - com_x_m
    lrB = com_x_m - avg(rear_cm)

    A_ok = plausible_lf_lr(lfA, lrA)
    B_ok = plausible_lf_lr(lfB, lrB)

    if A_ok and not B_ok:
        return lfA, lrA, "wheels (meters)"
    if B_ok and not A_ok:
        return lfB, lrB, "wheels (centimeters→meters)"
    if A_ok and B_ok:
        # pick the one closer to blueprint wheel_base if available
        try:
            bp = vehicle.get_world().get_blueprint_library().find(vehicle.type_id)
            wb_attr = bp.get_attribute("wheel_base") if bp.has_attribute("wheel_base") else None
            if wb_attr:
                wb = float(wb_attr.as_string())
                errA = abs((lfA+lrA) - wb)
                errB = abs((lfB+lrB) - wb)
                return (lfA, lrA, "wheels (meters)") if errA <= errB else (lfB, lrB, "wheels (cm→m)")
        except Exception:
            pass
        # otherwise prefer meters hypothesis
        return lfA, lrA, "wheels (meters)"

    # If neither plausible, fall back to bbox half-lengths (not axle distances)
    half_len = float(vehicle.bounding_box.extent.x)
    return half_len, half_len, "bbox half-length (fallback)"

def get_mass_and_Iz(vehicle):
    """
    Returns (mass, Iz, iz_estimated_flag).
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

def main():
    ap = argparse.ArgumentParser(description="Measure lf/lr (CoM→axles) from wheel locations in CARLA.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--actor-id", type=int, default=0)
    ap.add_argument("--role-name", default="ego")
    ap.add_argument("--blueprint", default="vehicle.lincoln.mkz")
    ap.add_argument("--bp-index", type=int, default=0)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)

    world = None
    for _ in range(60):
        try:
            world = client.get_world(); break
        except RuntimeError:
            time.sleep(1.0)
    if world is None:
        print(f"ERROR: cannot connect to CARLA at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)

    vehicle = None
    for i in range(120):
        vehicle = resolve_vehicle(world, args.actor_id, args.role_name, args.blueprint, args.bp_index)
        if vehicle: break
        if i % 4 == 0: print("Waiting for vehicle…")
        time.sleep(0.5)
    if not vehicle:
        print("ERROR: no vehicle found", file=sys.stderr)
        sys.exit(2)

    print(f"Vehicle: id={vehicle.id}, type={vehicle.type_id}, role='{vehicle.attributes.get('role_name','')}'")

    try:
        lf, lr, src = compute_lf_lr_from_wheels(vehicle)
    except Exception as e:
        half_len = float(vehicle.bounding_box.extent.x)
        lf, lr, src = half_len, half_len, f"bbox half-length (fallback due to error: {e})"

    mass, Iz, iz_est = get_mass_and_Iz(vehicle)
    wb = lf + lr

    print(f"Result source: {src}")
    print(f"mass = {mass:.2f} kg")
    print(f"lf = {lf:.3f} m, lr = {lr:.3f} m  (wheelbase ≈ {wb:.3f} m)")
    print(f"Iz = {Iz:.2f} kg·m^2{' [estimated]' if iz_est else ''}")

    # If blueprint has wheel_base, compare:
    try:
        bp = world.get_blueprint_library().find(vehicle.type_id)
        if bp.has_attribute("wheel_base"):
            wb_bp = float(bp.get_attribute("wheel_base").as_string())
            print(f"Blueprint wheel_base = {wb_bp:.3f} m (Δ = {(wb - wb_bp):+.3f} m)")
    except Exception:
        pass

if __name__ == "__main__":
    main()
