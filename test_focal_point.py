#!/usr/bin/env python3
import json
import glob
import os

# Read the scene configuration
with open("build/scene.json", "r") as f:
    scene_config = json.load(f)

scene_focal_point = scene_config["camera"]["focal_point"]
print(f"Scene focal point: {scene_focal_point}")

# Check frame files
frame_files = sorted(glob.glob(os.path.join("build/frames", "frame_*.json")))

if frame_files:
    print(f"\nChecking {len(frame_files)} frame files...")

    # Check first few frames
    for i, frame_file in enumerate(frame_files[:5]):
        with open(frame_file, "r") as f:
            frame_data = json.load(f)

        frame_focal_point = frame_data["camera"]["focal_point"]
        frame_name = os.path.basename(frame_file)

        if frame_focal_point != scene_focal_point:
            print(f"  {frame_name}: MISMATCH - {frame_focal_point}")
        else:
            print(f"  {frame_name}: OK - {frame_focal_point}")

    # Check if all frames have the same focal point
    all_focal_points = set()
    for frame_file in frame_files:
        with open(frame_file, "r") as f:
            frame_data = json.load(f)
        all_focal_points.add(tuple(frame_data["camera"]["focal_point"]))

    if len(all_focal_points) == 1:
        focal_point = list(all_focal_points)[0]
        if list(focal_point) == scene_focal_point:
            print(f"\n✓ All {len(frame_files)} frames have the correct focal point!")
        else:
            print(
                f"\n✗ All frames have focal point {focal_point}, but scene has {scene_focal_point}"
            )
    else:
        print(f"\n✗ Frames have {len(all_focal_points)} different focal points!")
        for fp in all_focal_points:
            print(f"  - {list(fp)}")
else:
    print("\nNo frame files found to check.")
