#!/bin/bash
# Force Vulkan to use NVIDIA
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Launch CARLA with Vulkan, low quality, no sound, and Town01 by default
./CarlaUE4.sh -vulkan -quality-level=Low "$@" & 
CARLA_PID=$!

python3 load_town05.py

wait $CARLA_PID
