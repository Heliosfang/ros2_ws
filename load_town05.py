#!/usr/bin/env python3
import carla
import time

def main():
    # Connect to CARLA


    # Wait until the server is ready
    world = None
    for _ in range(30):  # try for ~30s
        try:
            client = carla.Client("127.0.0.1", 2000)
            client.set_timeout(10.0)
            world = client.get_world()
            break
        except RuntimeError:
            print("Waiting for CARLA server...")
            time.sleep(1.0)

    if world is None:
        print("Could not connect to CARLA server.")
        return
    bp_lib = world.get_blueprint_library()
    print([bp.id for bp in bp_lib.filter("static.prop*")])

    # Switch to Town05
    print("Loading Town05...")
    client.load_world("Town05")   # or "Town05_Opt" for the lighter version
    print("Town05 loaded.")

if __name__ == "__main__":
    main()
