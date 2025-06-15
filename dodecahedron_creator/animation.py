# Animation logic, timing, and transformations

import time
import numpy as np
import json
import os
import glob
import easing_functions as easing
from vedo import Plotter

from .config import (
    read_config_file,
    CONFIG_FILE_PATH,
)

# Animation constants
TIMER_INTERVAL_MS = 20
VIEWER_FPS = 50
DEGREES_PER_ROTATION = 360.0
MS_PER_SECOND = 1000.0


def get_easing_function(easing_type):
    easing_map = {
        "linear": lambda t: t,
        "quadIn": easing.QuadEaseIn(),
        "quadOut": easing.QuadEaseOut(),
        "quadInOut": easing.QuadEaseInOut(),
        "cubicIn": easing.CubicEaseIn(),
        "cubicOut": easing.CubicEaseOut(),
        "cubicInOut": easing.CubicEaseInOut(),
        "quartIn": easing.QuarticEaseIn(),
        "quartOut": easing.QuarticEaseOut(),
        "quartInOut": easing.QuarticEaseInOut(),
        "quintIn": easing.QuinticEaseIn(),
        "quintOut": easing.QuinticEaseOut(),
        "quintInOut": easing.QuinticEaseInOut(),
        "sineIn": easing.SineEaseIn(),
        "sineOut": easing.SineEaseOut(),
        "sineInOut": easing.SineEaseInOut(),
        "expoIn": easing.ExponentialEaseIn(),
        "expoOut": easing.ExponentialEaseOut(),
        "expoInOut": easing.ExponentialEaseInOut(),
        "circIn": easing.CircularEaseIn(),
        "circOut": easing.CircularEaseOut(),
        "circInOut": easing.CircularEaseInOut(),
        "backIn": easing.BackEaseIn(),
        "backOut": easing.BackEaseOut(),
        "backInOut": easing.BackEaseInOut(),
        "elasticIn": easing.ElasticEaseIn(),
        "elasticOut": easing.ElasticEaseOut(),
        "elasticInOut": easing.ElasticEaseInOut(),
        "bounceIn": easing.BounceEaseIn(),
        "bounceOut": easing.BounceEaseOut(),
        "bounceInOut": easing.BounceEaseInOut(),
    }

    func = easing_map.get(easing_type)
    if func is None:
        print(f"Warning: Unknown easing type '{easing_type}', using quadInOut")
        return easing.QuadEaseInOut()
    return func


def setup_animation_state():
    initial_config = read_config_file()
    if initial_config:
        animation_state = {
            "rotation_enabled": False,
            "rotation_speed": initial_config["speed"],
            "rotation_azimuth": initial_config["azimuth"],
            "rotation_elevation": initial_config["elevation"],
            "pause_duration": initial_config["pause"],
            "total_rotation": 0.0,
            "is_paused": False,
            "pause_start_time": None,
            "initial_transform": None,
            "rotation_progress": 0.0,
            "rotations": initial_config["rotations"],
            "current_rotation": 0,
            "easing_type": initial_config["easing"],
            "easing_func": get_easing_function(initial_config["easing"]),
            "continuous": initial_config["continuous"],
            "capture_fps": initial_config["capture_fps"],
            "frame_counter": 0,
            "last_capture_time": None,
            "first_cycle_complete": False,
        }
        print(f"\nLoaded config from {CONFIG_FILE_PATH}:")
        print(f"  Azimuth: {initial_config['azimuth']}°")
        print(f"  Elevation: {initial_config['elevation']}°")
        print(f"  Speed: {initial_config['speed']}°/frame (max speed)")
        if initial_config["continuous"]:
            print("  Mode: Continuous (no pauses)")
        else:
            print(f"  Pause: {initial_config['pause']}s after each rotation")
        print(
            f"  Rotations: {initial_config['rotations']} ({initial_config['rotations'] * int(DEGREES_PER_ROTATION)}° total)"
        )
        print(f"  Easing: {initial_config['easing']}")
        if initial_config["capture_fps"] > 0:
            print(f"  Frame capture: {initial_config['capture_fps']} fps (JSON)")
    else:
        animation_state = {
            "rotation_enabled": False,
            "rotation_speed": 1.0,
            "rotation_azimuth": 0.0,
            "rotation_elevation": 0.0,
            "pause_duration": 1.0,
            "total_rotation": 0.0,
            "is_paused": False,
            "pause_start_time": None,
            "initial_transform": None,
            "rotation_progress": 0.0,
            "rotations": 1,
            "current_rotation": 0,
            "easing_type": "quadInOut",
            "easing_func": get_easing_function("quadInOut"),
            "continuous": False,
            "capture_fps": 0,
            "frame_counter": 0,
            "last_capture_time": None,
            "first_cycle_complete": False,
        }
    return animation_state


def update_animation_config_from_file(animation_state):
    config_from_file = read_config_file()
    if config_from_file is not None:
        # Check if continuous mode changed
        old_continuous = animation_state["continuous"]
        new_continuous = config_from_file["continuous"]

        # Check if pause duration changed during a pause
        old_pause_duration = animation_state["pause_duration"]
        new_pause_duration = config_from_file["pause"]

        animation_state["rotation_azimuth"] = config_from_file["azimuth"]
        animation_state["rotation_elevation"] = config_from_file["elevation"]
        animation_state["rotation_speed"] = config_from_file["speed"]
        animation_state["pause_duration"] = config_from_file["pause"]
        animation_state["rotations"] = config_from_file["rotations"]
        animation_state["continuous"] = config_from_file["continuous"]
        animation_state["capture_fps"] = config_from_file["capture_fps"]

        # If pause duration changed while paused, notify
        if old_pause_duration != new_pause_duration:
            if animation_state["is_paused"] and not new_continuous:
                print(
                    f"\nPause duration changed from {old_pause_duration}s to {new_pause_duration}s"
                )
            elif animation_state["rotation_enabled"]:
                print(
                    f"\nPause duration updated to {new_pause_duration}s (will apply after current rotation)"
                )

        # If switching from continuous to non-continuous and we're paused, update pause time
        if old_continuous and not new_continuous and animation_state["is_paused"]:
            # Reset pause to use new duration
            animation_state["pause_start_time"] = time.time()
            print(
                f"\nPause duration updated to {animation_state['pause_duration']} seconds"
            )

        # If switching from non-continuous to continuous, cancel any pause
        if not old_continuous and new_continuous and animation_state["is_paused"]:
            animation_state["is_paused"] = False
            animation_state["pause_start_time"] = None
            print("\nContinuous mode enabled, cancelling pause")

        if config_from_file["easing"] != animation_state["easing_type"]:
            animation_state["easing_type"] = config_from_file["easing"]
            animation_state["easing_func"] = get_easing_function(
                config_from_file["easing"]
            )


def handle_animation_pause_logic(animation_state):
    if not animation_state["continuous"] and animation_state["is_paused"]:
        current_time = time.time()
        elapsed_pause = current_time - animation_state["pause_start_time"]
        if elapsed_pause >= animation_state["pause_duration"]:
            animation_state["is_paused"] = False
            animation_state["pause_start_time"] = None
            print("Pause complete, resuming animation...")
            return True
        return False
    return True


def calculate_rotation_and_apply(animation_state, plotter, mesh, config=None):
    azimuth_rad = np.radians(animation_state["rotation_azimuth"])
    elevation_rad = np.radians(animation_state["rotation_elevation"])

    camera = plotter.camera
    camera_pos = np.array(camera.GetPosition())

    # Use focal point from config if available to ensure consistency
    if config and "camera" in config and "focal_point" in config["camera"]:
        focal_point = np.array(config["camera"]["focal_point"])
    else:
        focal_point = np.array(camera.GetFocalPoint())

    camera_up = np.array(camera.GetViewUp())

    view_direction = focal_point - camera_pos
    view_direction = view_direction / np.linalg.norm(view_direction)

    camera_right = np.cross(view_direction, camera_up)
    camera_right = camera_right / np.linalg.norm(camera_right)

    camera_up = np.cross(camera_right, view_direction)
    camera_up = camera_up / np.linalg.norm(camera_up)

    rotation_axis_camera = np.array(
        [
            -np.sin(azimuth_rad),
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )

    rotation_axis = (
        rotation_axis_camera[0] * camera_right
        + rotation_axis_camera[1] * camera_up
        + rotation_axis_camera[2] * view_direction
    )

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    max_speed = animation_state["rotation_speed"]
    progress_increment = max_speed / DEGREES_PER_ROTATION
    new_progress = animation_state["rotation_progress"] + progress_increment

    if new_progress >= 1.0:
        new_progress = 1.0

    total_rotations = animation_state["rotations"]
    current_rotation = animation_state["current_rotation"]
    overall_progress = (current_rotation + new_progress) / total_rotations

    eased_overall_current = animation_state["easing_func"](overall_progress)

    prev_overall_progress = (
        current_rotation + animation_state["rotation_progress"]
    ) / total_rotations
    eased_overall_previous = animation_state["easing_func"](prev_overall_progress)

    rotation_amount = (eased_overall_current - eased_overall_previous) * (
        total_rotations * DEGREES_PER_ROTATION
    )

    mesh.rotate(rotation_amount, axis=rotation_axis, point=mesh.pos())

    animation_state["rotation_progress"] = new_progress
    animation_state["total_rotation"] = overall_progress * (
        total_rotations * DEGREES_PER_ROTATION
    )

    return new_progress


def handle_rotation_completion(animation_state, mode="positioning", plotter=None):
    if animation_state["rotation_progress"] >= 1.0:
        animation_state["current_rotation"] += 1

        if animation_state["current_rotation"] >= animation_state["rotations"]:
            animation_state["rotation_progress"] = 0.0
            animation_state["total_rotation"] = 0.0
            animation_state["current_rotation"] = 0

            if not animation_state["continuous"]:
                animation_state["is_paused"] = True
                animation_state["pause_start_time"] = time.time()
                print(f"\nPausing for {animation_state['pause_duration']} seconds...")
        else:
            animation_state["rotation_progress"] = 0.0
            animation_state["total_rotation"] = 0.0


def handle_timer(evt, plotter, mesh, config, animation_state, mode="positioning"):
    update_animation_config_from_file(animation_state)

    if animation_state["rotation_enabled"]:
        if not handle_animation_pause_logic(animation_state):
            return

        new_progress = calculate_rotation_and_apply(
            animation_state, plotter, mesh, config
        )

        if new_progress >= 1.0:
            handle_rotation_completion(animation_state, mode, plotter)

        plotter.render()


def capture_frame_as_json(plotter, mesh, frame_number, config):
    camera = plotter.camera

    transform_matrix = (
        mesh.transform.matrix.tolist()
        if hasattr(mesh.transform, "matrix")
        else [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    # Use the focal point from the saved scene config to ensure consistency
    focal_point = config["camera"].get("focal_point", list(camera.GetFocalPoint()))

    # Debug: Print focal point info for first frame
    if frame_number == 1:
        print(
            f"  Frame 1 - Config focal point: {config['camera'].get('focal_point', 'Not set')}"
        )
        print(f"  Frame 1 - Camera focal point: {list(camera.GetFocalPoint())}")
        print(f"  Frame 1 - Using focal point: {focal_point}")

    # Store only geometry and camera data, no style information
    scene_data = {
        "camera": {
            "position": list(camera.GetPosition()),
            "focal_point": focal_point,
            "view_up": list(camera.GetViewUp()),
            "view_angle": camera.GetViewAngle(),
            "clipping_range": list(camera.GetClippingRange()),
        },
        "mesh": {
            "position": list(mesh.pos()),
            "transform_matrix": transform_matrix,
        },
        "viewport": {
            "size": list(plotter.window.GetSize()),
        },
    }

    # Always save to shared frames directory
    frames_dir = "build/frames"
    os.makedirs(frames_dir, exist_ok=True)
    json_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.json")

    with open(json_path, "w") as f:
        json.dump(scene_data, f, indent=2)

    print(f"  Captured frame {frame_number} -> {json_path}")


def run_headless_recording(config):
    """Run a headless animation cycle to capture frames"""
    # Import here to avoid circular dependency
    from .utils import setup_mesh, setup_camera

    print("\n" + "=" * 60)
    print("HEADLESS RECORDING")
    print("=" * 60)

    # Clear existing frame files in shared frames directory
    frames_dir = "build/frames"
    os.makedirs(frames_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(frames_dir, "frame_*.json"))
    if existing_files:
        for f in existing_files:
            os.remove(f)
        print(f"Cleared {len(existing_files)} existing frame files")

    # Set up animation state
    animation_state = setup_animation_state()
    animation_state["rotation_enabled"] = True
    animation_state["frame_counter"] = 0
    animation_state["last_capture_time"] = 0  # Start capturing immediately

    # Create offscreen renderer
    mesh = setup_mesh(config)
    plotter = Plotter(
        bg=config["viewport"]["background_color"],
        offscreen=True,
        size=tuple(config["viewport"]["size"]),
    )
    plotter.add(mesh)
    setup_camera(plotter, config)
    plotter.show(axes=0, interactive=False, resetcam=False)

    # Store initial transform
    animation_state["initial_transform"] = mesh.transform.clone()

    print(f"Recording at {animation_state['capture_fps']} fps...")

    # Run animation loop
    frames_captured = 0
    last_time = time.time() * MS_PER_SECOND

    while animation_state["current_rotation"] < animation_state["rotations"]:
        current_time = time.time() * MS_PER_SECOND

        # Update animation if not paused
        if not animation_state["is_paused"]:
            new_progress = calculate_rotation_and_apply(
                animation_state, plotter, mesh, config
            )

            # Capture frame at specified FPS (only when animating, not during pause)
            if animation_state["capture_fps"] > 0:
                frame_interval = MS_PER_SECOND / animation_state["capture_fps"]
                if (
                    current_time - animation_state["last_capture_time"]
                    >= frame_interval
                ):
                    animation_state["frame_counter"] += 1
                    capture_frame_as_json(
                        plotter,
                        mesh,
                        animation_state["frame_counter"],
                        config,
                    )
                    animation_state["last_capture_time"] = current_time
                    frames_captured += 1

            # Handle rotation completion
            if new_progress >= 1.0:
                animation_state["current_rotation"] += 1
                if animation_state["current_rotation"] < animation_state["rotations"]:
                    animation_state["rotation_progress"] = 0.0
                    animation_state["total_rotation"] = 0.0

                    # Capture one final frame at the end position before pause
                    if (
                        not animation_state["continuous"]
                        and animation_state["capture_fps"] > 0
                    ):
                        animation_state["frame_counter"] += 1
                        capture_frame_as_json(
                            plotter,
                            mesh,
                            animation_state["frame_counter"],
                            config,
                        )
                        frames_captured += 1

                    # Handle pause if not continuous
                    if not animation_state["continuous"]:
                        animation_state["is_paused"] = True
                        animation_state["pause_start_time"] = time.time()
        else:
            # Handle pause logic (skip frame capture during pause)
            if handle_animation_pause_logic(animation_state):
                animation_state["is_paused"] = False
                # Reset capture time after pause to avoid frame burst
                animation_state["last_capture_time"] = time.time() * MS_PER_SECOND

        # Small delay to control animation speed (matching TIMER_INTERVAL_MS)
        elapsed = current_time - last_time
        if elapsed < TIMER_INTERVAL_MS:
            time.sleep((TIMER_INTERVAL_MS - elapsed) / MS_PER_SECOND)
        last_time = current_time

    plotter.close()

    print(f"\nRecording complete: {frames_captured} JSON frames saved to {frames_dir}/")
    print("=" * 60 + "\n")

    return frames_captured > 0
