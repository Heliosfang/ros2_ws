#!/usr/bin/env python3
# Minimal CARLA 0.9.16 script with native ROS 2 sensor publishing

import argparse
import logging
import carla

# ---- Simple config (edit these or override via CLI) -------------------------
CONFIG = {
    "carla_host": "127.0.0.1",
    "carla_port": 2000,
    "carla_timeout": 10.0,
    "carla_map": "Town01",             # set "" to keep current world

    "vehicle_blueprint": "vehicle.lincoln.mkz_2017",
    "vehicle_role_name": "ego",
    "vehicle_ros_name":  "ego",
    "spawn_index": 4,                   # index in get_spawn_points()

    # LiDAR: 2D layer
    "lidar_role_name": "lidar",
    "lidar_ros_name":  "lidar",
    "lidar_mount_xyz": (0.0, 0.0, 2.0),
    "lidar_range": 30.0,
    "lidar_upper_fov": 0.0,
    "lidar_lower_fov": 0.0,
    "lidar_points_per_second": 200000,
    "lidar_rotation_frequency": 10.0,
    "lidar_dropoff_general_rate": 0.0,
    "lidar_dropoff_intensity_limit": 1.0,
    "lidar_dropoff_zero_intensity": 0.0,

    # Camera
    "enable_camera": True,
    "camera_role_name": "rgb",
    "camera_ros_name":  "rgb",
    "camera_mount_xyz": (1.2, 0.0, 1.4),
    "camera_mount_rpy": (-5.0, 0.0, 0.0),  # pitch, roll, yaw (deg)
    "camera_width": 800,
    "camera_height": 600,
    "camera_fov": 90.0,
    "camera_sensor_tick": 0.0,

    "autopilot": True,
    "fixed_delta_seconds": 0.05,        # 20 Hz
}

# ---------------------------------------------------------------------------

def pick_spawn(world, index: int) -> carla.Transform:
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points available.")
    index = max(0, min(index, len(spawns) - 1))
    return spawns[index]

def main():
    parser = argparse.ArgumentParser("CARLA minimal spawner (native ROS2 sensors)")
    parser.add_argument("--host", default=CONFIG["carla_host"])
    parser.add_argument("--port", type=int, default=CONFIG["carla_port"])
    parser.add_argument("--map", default=CONFIG["carla_map"])
    parser.add_argument("--index", type=int, default=CONFIG["spawn_index"])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(CONFIG["carla_timeout"])

    world = client.get_world()
    if args.map and world.get_map().name.split("/")[-1] != args.map:
        logging.info(f"Loading map: {args.map}")
        world = client.load_world(args.map)

    # Synchronous sim for determinism
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = CONFIG["fixed_delta_seconds"]
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    actors = []
    try:
        bps = world.get_blueprint_library()

        # ---- Vehicle ----
        vbp = bps.find(CONFIG["vehicle_blueprint"])
        vbp.set_attribute("role_name", CONFIG["vehicle_role_name"])
        # Native ROS2 name (if your build uses it)
        if vbp.has_attribute("ros_name"):
            vbp.set_attribute("ros_name", CONFIG["vehicle_ros_name"])

        vehicle = world.spawn_actor(vbp, pick_spawn(world, args.index))
        actors.append(vehicle)
        logging.info(f"Spawned vehicle id={vehicle.id}")

        if CONFIG["autopilot"]:
            vehicle.set_autopilot(True, tm.get_port())

        # ---- LiDAR (2D) ----
        lbp = bps.find("sensor.lidar.ray_cast")
        lbp.set_attribute("role_name", CONFIG["lidar_role_name"])
        if lbp.has_attribute("ros_name"):
            lbp.set_attribute("ros_name", CONFIG["lidar_ros_name"])
        lbp.set_attribute("channels", "1")
        lbp.set_attribute("range", str(CONFIG["lidar_range"]))
        lbp.set_attribute("upper_fov", str(CONFIG["lidar_upper_fov"]))
        lbp.set_attribute("lower_fov", str(CONFIG["lidar_lower_fov"]))
        lbp.set_attribute("points_per_second", str(CONFIG["lidar_points_per_second"]))
        lbp.set_attribute("rotation_frequency", str(CONFIG["lidar_rotation_frequency"]))
        lbp.set_attribute("dropoff_general_rate", str(CONFIG["lidar_dropoff_general_rate"]))
        lbp.set_attribute("dropoff_intensity_limit", str(CONFIG["lidar_dropoff_intensity_limit"]))
        lbp.set_attribute("dropoff_zero_intensity", str(CONFIG["lidar_dropoff_zero_intensity"]))

        lx, ly, lz = CONFIG["lidar_mount_xyz"]
        lidar_tf = carla.Transform(carla.Location(x=lx, y=ly, z=lz))
        lidar = world.spawn_actor(lbp, lidar_tf, attach_to=vehicle)
        actors.append(lidar)

        # ✅ Enable LiDAR for ROS 2
        if hasattr(lidar, "enable_for_ros"):
            lidar.enable_for_ros()
            logging.info("LiDAR enabled for ROS2.")

        # ---- Camera (optional) ----
        if CONFIG["enable_camera"]:
            cbp = bps.find("sensor.camera.rgb")
            cbp.set_attribute("role_name", CONFIG["camera_role_name"])
            if cbp.has_attribute("ros_name"):
                cbp.set_attribute("ros_name", CONFIG["camera_ros_name"])
            cbp.set_attribute("image_size_x", str(CONFIG["camera_width"]))
            cbp.set_attribute("image_size_y", str(CONFIG["camera_height"]))
            cbp.set_attribute("fov", str(CONFIG["camera_fov"]))
            cbp.set_attribute("sensor_tick", str(CONFIG["camera_sensor_tick"]))

            cx, cy, cz = CONFIG["camera_mount_xyz"]
            pitch, roll, yaw = CONFIG["camera_mount_rpy"]
            cam_tf = carla.Transform(
                carla.Location(x=cx, y=cy, z=cz),
                carla.Rotation(pitch=pitch, roll=roll, yaw=yaw)
            )
            camera = world.spawn_actor(cbp, cam_tf, attach_to=vehicle)
            actors.append(camera)

            # ✅ Enable Camera for ROS 2
            if hasattr(camera, "enable_for_ros"):
                camera.enable_for_ros()
                logging.info("Camera enabled for ROS2.")

        # Prime one tick to register actors/topics
        world.tick()
        logging.info("Running... Ctrl+C to exit.")

        # Simple loop
        while True:
            world.tick()

    except KeyboardInterrupt:
        logging.info("Interrupted. Cleaning up...")
    finally:
        world.apply_settings(original_settings)
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass
        logging.info("All actors destroyed. Bye.")

if __name__ == "__main__":
    main()
