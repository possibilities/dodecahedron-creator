from vedo import Mesh, Plotter
import vedo
import numpy as np
import svgwrite
from collections import defaultdict
import json
import os
import argparse
import yaml
import time
import easing_functions as easing
import cairosvg
import glob
import re
from PIL import Image

MODEL_PATH = "resources/dodecahedron.obj"
SVG_PATH = "build/dodecahedron.svg"
SCENE_CONFIG_PATH = "build/scene.json"
CONFIG_FILE_PATH = "config.yaml"
GIF_PATH = "build/animation.gif"

TIMER_INTERVAL_MS = 20
VIEWER_FPS = 50
DEGREES_PER_ROTATION = 360.0
MS_PER_SECOND = 1000.0
MIN_GIF_FRAME_DURATION_MS = 20

FOV_ADJUSTMENT_STEP = 5
MAX_FOV_DEGREES = 120
MIN_FOV_DEGREES = 5
DEFAULT_VIEWER_SIZE = (1400, 900)

DEFAULT_SVG_STROKE_WIDTH = 12
DEFAULT_MESH_LINE_WIDTH = 4
DEFAULT_BACKGROUND_COLOR = "white"
DEFAULT_MESH_COLOR = "black"
DEFAULT_EDGE_COLOR = [1, 1, 1]
DEFAULT_SVG_FILL = "black"
DEFAULT_SVG_STROKE = "white"

FRAMES_DIR = "build/frames"
FRAME_JSON_PATTERN = "frame_*.json"
FRAME_SVG_PATTERN = "frame_*.svg"
FRAME_FILENAME_FORMAT = "frame_{:04d}.json"

BBOX_PADDING_MULTIPLIER = 2
IDENTITY_MATRIX_4X4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class SvgGenerationContext:
    def __init__(self):
        self.global_bbox = None

    def reset(self):
        self.global_bbox = None

    def get_or_calculate_bbox(self, config):
        if self.global_bbox is None:
            self.global_bbox = get_global_bounding_box(config)
        return self.global_bbox


def read_config_file():
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)
            return {
                "azimuth": config.get("azimuth", 0),
                "elevation": config.get("elevation", 0),
                "speed": config.get("speed", 1.0),
                "pause": config.get("pause", 1.0),
                "rotations": config.get("rotations", 1),
                "easing": config.get("easing", "quadInOut"),
                "continuous": config.get("continuous", False),
                "capture_fps": config.get("capture_fps", 0),
                "raster_height": config.get("raster_height", 100),
                "svg": config.get(
                    "svg",
                    {
                        "stroke_width": DEFAULT_SVG_STROKE_WIDTH,
                        "background": DEFAULT_BACKGROUND_COLOR,
                        "fill": DEFAULT_SVG_FILL,
                        "stroke": DEFAULT_SVG_STROKE,
                        "stroke_linecap": "round",
                        "stroke_linejoin": "round",
                    },
                ),
            }
    except Exception:
        return None


def load_configuration(ignore_saved=False):
    config_file = SCENE_CONFIG_PATH

    if not ignore_saved and os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"Loaded scene from {SCENE_CONFIG_PATH}")
        return config
    else:
        if ignore_saved:
            print("Using fresh scene configuration (--fresh flag)")
        else:
            print("Using default scene configuration")
        return {
            "camera": {},
            "mesh": {
                "color": "black",
                "linewidth": 4,
                "edge_color": [1, 1, 1],
            },
            "viewport": {"background_color": "white"},
        }


def setup_mesh(config):
    mesh = Mesh(MODEL_PATH)

    if "transform_matrix" in config["mesh"]:
        transform_matrix = np.array(config["mesh"]["transform_matrix"])
        mesh.apply_transform(transform_matrix)

    if "position" in config["mesh"]:
        mesh.pos(config["mesh"]["position"])

    mesh.color(config["mesh"]["color"])
    mesh.lighting("off")
    mesh.flat()

    mesh.linewidth(config["mesh"]["linewidth"])
    mesh.properties.EdgeVisibilityOn()
    mesh.properties.SetEdgeColor(*config["mesh"]["edge_color"])
    mesh.properties.SetLineWidth(config["mesh"]["linewidth"])

    return mesh


def create_viewer(config):
    plotter = Plotter(size=(1400, 900), bg=config["viewport"]["background_color"])
    return plotter


def setup_camera(plotter, config):
    camera = plotter.camera
    if "position" in config["camera"]:
        camera.SetPosition(config["camera"]["position"])
    if "focal_point" in config["camera"]:
        camera.SetFocalPoint(config["camera"]["focal_point"])
    if "view_up" in config["camera"]:
        camera.SetViewUp(config["camera"]["view_up"])
    if "view_angle" in config["camera"]:
        camera.SetViewAngle(config["camera"]["view_angle"])
    if "clipping_range" in config["camera"]:
        camera.SetClippingRange(config["camera"]["clipping_range"])
    return camera


def run_interactive_session(plotter):
    plotter.interactive()


def save_configuration(plotter, mesh, config, animation_state):
    camera = plotter.camera

    final_position = list(camera.GetPosition())
    final_focal_point = list(camera.GetFocalPoint())
    final_view_up = list(camera.GetViewUp())
    final_view_angle = camera.GetViewAngle()
    final_clipping_range = list(camera.GetClippingRange())

    final_mesh_position = list(mesh.pos())

    transform_matrix = (
        mesh.transform.matrix.tolist()
        if hasattr(mesh.transform, "matrix")
        else config["mesh"]["transform_matrix"]
    )

    scene_data = {
        "camera": {
            "position": final_position,
            "focal_point": final_focal_point,
            "view_up": final_view_up,
            "view_angle": final_view_angle,
            "clipping_range": final_clipping_range,
        },
        "mesh": {
            "position": final_mesh_position,
            "transform_matrix": transform_matrix,
            "color": config["mesh"]["color"],
            "linewidth": config["mesh"]["linewidth"],
            "edge_color": config["mesh"]["edge_color"],
        },
        "viewport": {
            "size": list(plotter.window.GetSize()),
            "background_color": config["viewport"]["background_color"],
        },
        "svg": config.get(
            "svg",
            {
                "stroke_width": DEFAULT_SVG_STROKE_WIDTH,
                "background": DEFAULT_BACKGROUND_COLOR,
                "fill": DEFAULT_SVG_FILL,
                "stroke": DEFAULT_SVG_STROKE,
                "stroke_linecap": "round",
                "stroke_linejoin": "round",
            },
        ),
    }

    os.makedirs("build", exist_ok=True)
    with open(SCENE_CONFIG_PATH, "w") as f:
        json.dump(scene_data, f, indent=2)

    print(f"Saved scene to {SCENE_CONFIG_PATH}")
    return True


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


def handle_frame_capture(animation_state, plotter, mesh, config, mode="positioning"):
    # Frame capture is now only done in headless mode
    pass


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


def handle_key_press(evt, plotter, animation_state, mesh, config, mode="positioning"):
    if mode == "animation":
        # No special keys in animation preview mode
        return

    if evt.keypress == "Up":
        cam = plotter.camera
        current_fov = cam.GetViewAngle()
        new_fov = min(current_fov + FOV_ADJUSTMENT_STEP, MAX_FOV_DEGREES)
        cam.SetViewAngle(new_fov)
        print(f"FOV increased to {new_fov}°")
        plotter.render()
    elif evt.keypress == "Down":
        cam = plotter.camera
        current_fov = cam.GetViewAngle()
        new_fov = max(current_fov - 5, 5)
        cam.SetViewAngle(new_fov)
        print(f"FOV decreased to {new_fov}°")
        plotter.render()
    elif evt.keypress == "space":
        if mode == "positioning":
            return

        animation_state["rotation_enabled"] = not animation_state["rotation_enabled"]
        animation_state["total_rotation"] = 0.0
        animation_state["is_paused"] = False
        animation_state["pause_start_time"] = None
        animation_state["rotation_progress"] = 0.0
        animation_state["current_rotation"] = 0

        if animation_state["rotation_enabled"]:
            animation_state["initial_transform"] = mesh.transform.clone()
            animation_state["frame_counter"] = 0
            animation_state["last_capture_time"] = time.time() * MS_PER_SECOND
            animation_state["first_cycle_complete"] = False

        status = "started" if animation_state["rotation_enabled"] else "stopped"
        print(f"Rotation animation {status}")


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
            "color": config["mesh"]["color"],
            "linewidth": config["mesh"]["linewidth"],
            "edge_color": config["mesh"]["edge_color"],
        },
        "viewport": {
            "size": list(plotter.window.GetSize()),
            "background_color": config["viewport"]["background_color"],
        },
        "svg": config.get(
            "svg",
            {
                "stroke_width": DEFAULT_SVG_STROKE_WIDTH,
                "background": DEFAULT_BACKGROUND_COLOR,
                "fill": DEFAULT_SVG_FILL,
                "stroke": DEFAULT_SVG_STROKE,
                "stroke_linecap": "round",
                "stroke_linejoin": "round",
            },
        ),
    }

    frames_dir = FRAMES_DIR
    os.makedirs(frames_dir, exist_ok=True)
    json_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.json")

    with open(json_path, "w") as f:
        json.dump(scene_data, f, indent=2)

    print(f"  Captured frame {frame_number} -> {json_path}")


def setup_scene_components(use_fresh=False):
    config = load_configuration(ignore_saved=use_fresh)
    mesh = setup_mesh(config)
    plotter = create_viewer(config)
    plotter.add(mesh)
    return config, mesh, plotter


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


def register_event_handlers(plotter, mesh, config, animation_state, mode="positioning"):
    plotter.add_callback(
        "on key press",
        lambda evt: handle_key_press(evt, plotter, animation_state, mesh, config, mode),
    )
    plotter.add_callback(
        "timer",
        lambda evt: handle_timer(evt, plotter, mesh, config, animation_state, mode),
        enable_picking=False,
    )
    plotter.timer_callback("create", dt=20)


def print_keybindings(mode="positioning"):
    print("\n=== Keybindings ===")

    if mode == "positioning":
        print("Camera Controls:")
        print("  Mouse Left   : Rotate camera")
        print("  Mouse Right  : Zoom/dolly")
        print("  Mouse Middle : Pan/translate")
        print(f"  Up Arrow     : Increase FOV by {FOV_ADJUSTMENT_STEP}°")
        print(f"  Down Arrow   : Decrease FOV by {FOV_ADJUSTMENT_STEP}°")
        print("\nUtility:")
        print("  q            : Quit and save scene")
        print("  r            : Reset camera")
        print("  s            : Screenshot")
    else:
        print("Animation Preview:")
        print("  Animation runs automatically")
        print("  Camera controls are disabled during animation")
        print("\nAnimation Settings (via config.yaml):")
        print(f"  azimuth      : Horizontal angle (0-{int(DEGREES_PER_ROTATION)}°)")
        print("  elevation    : Vertical angle (-90° to +90°)")
        print("  speed        : Rotation speed (0-10°/frame)")
        print("  rotations    : Number of full rotations")
        print("  easing       : Animation easing function")
        print("\nUtility:")
        print("  q            : Quit and proceed to recording")

    print("==================\n")


def run_interactive_viewer(plotter, config, use_fresh, mode="positioning"):
    config_file = SCENE_CONFIG_PATH
    setup_camera(plotter, config)
    if not use_fresh and os.path.exists(config_file) and config["camera"]:
        plotter.show(axes=0, interactive=False, resetcam=False)
    else:
        plotter.show(axes=0, interactive=False, resetcam=True)

    run_interactive_session(plotter)


def configure_scene_in_viewer(use_fresh=False, mode="positioning"):
    config, mesh, plotter = setup_scene_components(use_fresh)
    animation_state = setup_animation_state()
    register_event_handlers(plotter, mesh, config, animation_state, mode)

    if mode == "animation":
        print("\n" + "=" * 60)
        print("ANIMATION PREVIEW MODE")
        print("=" * 60)
        print("Animation will start automatically")
        print("Close the window to proceed to recording")
        print("=" * 60 + "\n")

        mode_text = vedo.Text2D(
            "Animation Preview",
            pos="top-left",
            c="black",
            font="Fira Code",
            s=1.2,
            bg="white",
            alpha=0.8,
        )
        plotter.add(mode_text)
    else:
        print("\n" + "=" * 60)
        print("POSITIONING MODE")
        print("=" * 60)
        print("Adjust the view for the final static SVG output")
        print("Close the viewer when positioning is complete")
        print("=" * 60 + "\n")

        mode_text = vedo.Text2D(
            "Positioning Mode",
            pos="top-left",
            c="black",
            font="Fira Code",
            s=1.2,
            bg="white",
            alpha=0.8,
        )
        plotter.add(mode_text)

    print_keybindings(mode)

    if mode == "animation":
        animation_state["rotation_enabled"] = True
        animation_state["initial_transform"] = mesh.transform.clone()
        animation_state["frame_counter"] = 0
        animation_state["last_capture_time"] = time.time() * MS_PER_SECOND
        animation_state["first_cycle_complete"] = False

        print("\nAnimation started automatically")

    run_interactive_viewer(plotter, config, use_fresh, mode)

    if mode == "positioning":
        was_saved = save_configuration(plotter, mesh, config, animation_state)
    else:
        was_saved = False

    plotter.close()
    return was_saved


def load_scene_data():
    with open(SCENE_CONFIG_PATH, "r") as f:
        return json.load(f)


def setup_offscreen_renderer(config):
    mesh = setup_mesh(config)

    plotter = Plotter(
        bg=config["viewport"]["background_color"],
        offscreen=True,
        size=tuple(config["viewport"]["size"]),
    )
    plotter.add(mesh)

    camera = setup_camera(plotter, config)
    plotter.show(axes=0, interactive=False, resetcam=False)

    return plotter, mesh, camera


def calculate_visible_geometry(mesh, camera):
    camera_pos = np.array(camera.GetPosition())
    focal_point = np.array(camera.GetFocalPoint())
    view_direction = camera_pos - focal_point
    view_direction = view_direction / np.linalg.norm(view_direction)

    vertices = mesh.vertices
    faces = mesh.cells

    visible_faces = []
    for face in faces:
        face_verts = [vertices[i] for i in face]

        v1 = face_verts[1] - face_verts[0]
        v2 = face_verts[2] - face_verts[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        face_center = np.mean(face_verts, axis=0)
        to_camera = camera_pos - face_center

        if np.dot(normal, to_camera) > 0:
            visible_faces.append(face)

    print(f"Visible faces: {len(visible_faces)} out of {len(faces)}")

    visible_edges_set = set()
    all_edges_count = {}

    for face in visible_faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            edge = tuple(sorted([v1, v2]))
            visible_edges_set.add(edge)
            all_edges_count[edge] = all_edges_count.get(edge, 0) + 1

    final_edges = []
    for edge, count in all_edges_count.items():
        final_edges.append(edge)

    print(f"Final visible edges: {len(final_edges)}")

    return final_edges, vertices


def project_to_2d(plotter, edges, vertices):
    renderer = plotter.renderer

    projected_edges = []
    for edge in edges:
        idx1, idx2 = edge
        p1 = vertices[idx1]
        p2 = vertices[idx2]

        renderer.SetWorldPoint(*p1, 1.0)
        renderer.WorldToDisplay()
        screen_p1 = renderer.GetDisplayPoint()[:2]

        renderer.SetWorldPoint(*p2, 1.0)
        renderer.WorldToDisplay()
        screen_p2 = renderer.GetDisplayPoint()[:2]

        projected_edges.append(
            {
                "start_3d": p1.tolist(),
                "end_3d": p2.tolist(),
                "start_2d": list(screen_p1),
                "end_2d": list(screen_p2),
            }
        )

    return projected_edges


def cross_product_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def get_global_bounding_box(config):
    plotter, mesh, camera = setup_offscreen_renderer(config)

    all_mesh_vertices = mesh.vertices
    renderer = plotter.renderer
    all_2d_points = []

    for vertex in all_mesh_vertices:
        renderer.SetWorldPoint(*vertex, 1.0)
        renderer.WorldToDisplay()
        screen_point = renderer.GetDisplayPoint()[:2]
        all_2d_points.append(screen_point)

    plotter.close()

    if all_2d_points:
        viewport_size = tuple(config["viewport"]["size"])
        svg_points = [(p[0], viewport_size[1] - p[1]) for p in all_2d_points]

        min_x = min(p[0] for p in svg_points)
        max_x = max(p[0] for p in svg_points)
        min_y = min(p[1] for p in svg_points)
        max_y = max(p[1] for p in svg_points)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        radius = max(
            max(abs(p[0] - center_x) for p in svg_points),
            max(abs(p[1] - center_y) for p in svg_points),
        )

        padding = config["svg"]["stroke_width"] * BBOX_PADDING_MULTIPLIER
        radius += padding
        return {
            "min_x": center_x - radius,
            "max_x": center_x + radius,
            "min_y": center_y - radius,
            "max_y": center_y + radius,
            "size": radius * 2,
        }

    return None


def setup_svg_renderer(json_path, context):
    with open(json_path, "r") as f:
        config = json.load(f)

    context.get_or_calculate_bbox(config)

    plotter, mesh, camera = setup_offscreen_renderer(config)
    visible_edges, vertices = calculate_visible_geometry(mesh, camera)
    projected_edges = project_to_2d(plotter, visible_edges, vertices)
    plotter.close()

    return config, projected_edges


def calculate_svg_bounds(config, projected_edges, context):
    viewport_size = tuple(config["viewport"]["size"])
    svg_config = config["svg"]

    if context.global_bbox:
        return {
            "min_x": context.global_bbox["min_x"],
            "max_x": context.global_bbox["max_x"],
            "min_y": context.global_bbox["min_y"],
            "max_y": context.global_bbox["max_y"],
            "svg_width": context.global_bbox["size"],
            "svg_height": context.global_bbox["size"],
        }
    else:
        all_vertices = set()
        for edge in projected_edges:
            start = (edge["start_2d"][0], viewport_size[1] - edge["start_2d"][1])
            end = (edge["end_2d"][0], viewport_size[1] - edge["end_2d"][1])
            all_vertices.add(start)
            all_vertices.add(end)

        vertices_list = list(all_vertices)

        if vertices_list:
            min_x = min(v[0] for v in vertices_list)
            max_x = max(v[0] for v in vertices_list)
            min_y = min(v[1] for v in vertices_list)
            max_y = max(v[1] for v in vertices_list)

            padding = svg_config["stroke_width"]
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding

            svg_width = max_x - min_x
            svg_height = max_y - min_y
            max_dim = max(svg_width, svg_height)

            if svg_width < max_dim:
                x_offset = (max_dim - svg_width) / 2
                min_x -= x_offset
                max_x += x_offset
                svg_width = max_dim

            if svg_height < max_dim:
                y_offset = (max_dim - svg_height) / 2
                min_y -= y_offset
                max_y += y_offset
                svg_height = max_dim

            return {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "svg_width": svg_width,
                "svg_height": svg_height,
            }
        else:
            return {
                "min_x": 0,
                "min_y": 0,
                "svg_width": viewport_size[0],
                "svg_height": viewport_size[1],
            }


def generate_svg_content(config, projected_edges, bounds, svg_path):
    viewport_size = tuple(config["viewport"]["size"])
    svg_config = config["svg"]

    min_x = bounds["min_x"]
    min_y = bounds["min_y"]
    svg_width = bounds["svg_width"]
    svg_height = bounds["svg_height"]

    all_vertices = set()
    for edge in projected_edges:
        start = (edge["start_2d"][0], viewport_size[1] - edge["start_2d"][1])
        end = (edge["end_2d"][0], viewport_size[1] - edge["end_2d"][1])
        all_vertices.add(start)
        all_vertices.add(end)

    vertices_list = list(all_vertices)

    dwg = svgwrite.Drawing(svg_path, size=(svg_width, svg_height))

    dwg.add(
        dwg.rect(
            insert=(0, 0), size=(svg_width, svg_height), fill=svg_config["background"]
        )
    )

    hull_vertices = convex_hull(vertices_list)
    translated_hull = [(x - min_x, y - min_y) for x, y in hull_vertices]
    dwg.add(dwg.polygon(points=translated_hull, fill=svg_config["fill"], stroke="none"))

    vertex_map = defaultdict(list)
    edge_used = [False] * len(projected_edges)

    for i, edge in enumerate(projected_edges):
        start = (edge["start_2d"][0], edge["start_2d"][1])
        end = (edge["end_2d"][0], edge["end_2d"][1])
        vertex_map[start].append((i, end))
        vertex_map[end].append((i, start))

    paths = []
    for i, edge in enumerate(projected_edges):
        if edge_used[i]:
            continue

        path = []
        start = (edge["start_2d"][0], edge["start_2d"][1])
        end = (edge["end_2d"][0], edge["end_2d"][1])

        edge_used[i] = True
        path.append(start)
        path.append(end)

        current = end
        while True:
            found = False
            for edge_idx, next_vertex in vertex_map[current]:
                if not edge_used[edge_idx]:
                    edge_used[edge_idx] = True
                    path.append(next_vertex)
                    current = next_vertex
                    found = True
                    break
            if not found:
                break

        current = start
        while True:
            found = False
            for edge_idx, next_vertex in vertex_map[current]:
                if not edge_used[edge_idx]:
                    edge_used[edge_idx] = True
                    path.insert(0, next_vertex)
                    current = next_vertex
                    found = True
                    break
            if not found:
                break

        paths.append(path)

    for path in paths:
        flipped_path = [(p[0], viewport_size[1] - p[1]) for p in path]
        translated_path = [(x - min_x, y - min_y) for x, y in flipped_path]

        dwg.add(
            dwg.polyline(
                points=translated_path,
                fill="none",
                stroke=svg_config["stroke"],
                stroke_width=svg_config["stroke_width"],
                stroke_linecap=svg_config.get("stroke_linecap", "round"),
                stroke_linejoin=svg_config.get("stroke_linejoin", "round"),
            )
        )

    dwg.save()
    return svg_path


def create_svg_from_json(json_path, svg_path, context=None):
    if context is None:
        context = SvgGenerationContext()

    config, projected_edges = setup_svg_renderer(json_path, context)
    bounds = calculate_svg_bounds(config, projected_edges, context)
    return generate_svg_content(config, projected_edges, bounds, svg_path)


def rasterize_svg_frames(svg_paths, target_height):
    for svg_path in svg_paths:
        with open(svg_path, "r") as f:
            svg_content = f.read()

        width_match = re.search(r'width="([\d.]+)"', svg_content)
        height_match = re.search(r'height="([\d.]+)"', svg_content)

        if width_match and height_match:
            svg_width = float(width_match.group(1))
            svg_height = float(height_match.group(1))

            scale_factor = target_height / svg_height
            target_width = int(svg_width * scale_factor)

            png_path = svg_path.replace(".svg", ".png")
            cairosvg.svg2png(
                url=svg_path,
                write_to=png_path,
                output_width=target_width,
                output_height=target_height,
            )
            print(f"  Rasterized to {png_path} ({target_width}x{target_height}px)")
        else:
            png_path = svg_path.replace(".svg", ".png")
            try:
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=png_path,
                    output_height=target_height,
                )
                print(f"  Rasterized to {png_path} (height: {target_height}px)")
            except Exception as e:
                print(f"  Error rasterizing {svg_path}: {e}")


def create_animated_gif(png_paths, output_path, capture_fps, animation_config):
    images = []
    for png_path in png_paths:
        if os.path.exists(png_path):
            img = Image.open(png_path)
            images.append(img)

    if images:
        viewer_fps = VIEWER_FPS
        degrees_per_viewer_frame = animation_config.get("speed", 1.0)
        total_degrees = animation_config.get("rotations", 1) * DEGREES_PER_ROTATION

        viewer_frames_for_animation = total_degrees / degrees_per_viewer_frame
        total_animation_time_ms = (
            viewer_frames_for_animation / viewer_fps
        ) * MS_PER_SECOND

        duration_per_frame = int(total_animation_time_ms / len(images))
        duration_per_frame = max(duration_per_frame, MIN_GIF_FRAME_DURATION_MS)

        # Create duration list for each frame
        durations = [duration_per_frame] * len(images)

        # Add pause duration to the last frame if not in continuous mode
        if not animation_config.get("continuous", False):
            pause_duration_ms = animation_config.get("pause", 1.0) * MS_PER_SECOND
            durations[-1] += int(pause_duration_ms)

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
        )

        actual_fps = MS_PER_SECOND / duration_per_frame
        total_gif_time = sum(durations) / MS_PER_SECOND
        print(f"Created animated GIF: {output_path}")
        print(f"  Frames: {len(images)}")
        print(f"  Duration per frame: {duration_per_frame}ms")
        if not animation_config.get("continuous", False):
            print(
                f"  Last frame duration: {durations[-1]}ms (includes {animation_config.get('pause', 1.0)}s pause)"
            )
        print(f"  Effective playback rate: {actual_fps:.1f} fps")
        print(f"  Total GIF duration: {total_gif_time:.2f}s")
        return True
    return False


def create_svg_from_scene():
    print("Generating SVG...")

    context = SvgGenerationContext()

    svg_path = create_svg_from_json(SCENE_CONFIG_PATH, SVG_PATH, context)
    print(f"SVG saved as {svg_path}")

    frame_json_files = sorted(glob.glob(os.path.join(FRAMES_DIR, FRAME_JSON_PATTERN)))

    if frame_json_files:
        print(f"\nConverting {len(frame_json_files)} frame JSON files to SVG...")
        svg_paths = []
        for json_path in frame_json_files:
            svg_path = json_path.replace(".json", ".svg")
            create_svg_from_json(json_path, svg_path, context)
            print(f"  Created {svg_path}")
            svg_paths.append(svg_path)
        print("Frame conversion complete!")

        config = read_config_file()
        if config and config.get("raster_height", 0) > 0:
            print(
                f"\nRasterizing {len(svg_paths)} SVG frames to PNG (height: {config['raster_height']}px)..."
            )
            rasterize_svg_frames(svg_paths, config["raster_height"])
            print("Rasterization complete!")

            png_paths = [p.replace(".svg", ".png") for p in svg_paths]
            if (
                all(os.path.exists(p) for p in png_paths)
                and config.get("capture_fps", 0) > 0
            ):
                print(f"\nCreating animated GIF from {len(png_paths)} frames...")
                create_animated_gif(png_paths, GIF_PATH, config["capture_fps"], config)


def run_headless_recording(config):
    """Run a headless animation cycle to capture frames"""
    print("\n" + "=" * 60)
    print("HEADLESS RECORDING")
    print("=" * 60)

    # Clear existing frame files
    frames_dir = FRAMES_DIR
    os.makedirs(frames_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(frames_dir, "frame_*.json"))
    existing_files.extend(glob.glob(os.path.join(frames_dir, "frame_*.svg")))
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
                        plotter, mesh, animation_state["frame_counter"], config
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
                            plotter, mesh, animation_state["frame_counter"], config
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

    print(f"\nRecording complete: {frames_captured} JSON frames saved to {FRAMES_DIR}/")
    print("=" * 60 + "\n")

    return frames_captured > 0


def check_for_captured_frames():
    frame_json_files = glob.glob(os.path.join(FRAMES_DIR, FRAME_JSON_PATTERN))
    return len(frame_json_files) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D dodecahedron viewer and SVG generator"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with default view, ignoring saved scene.json",
    )
    args = parser.parse_args()

    config = read_config_file()
    capture_enabled = config and config.get("capture_fps", 0) > 0

    if capture_enabled:
        print("\n" + "=" * 60)
        print("THREE-PHASE WORKFLOW ENABLED")
        print("=" * 60)
        print("Phase 1: Positioning for static SVG")
        print("Phase 2: Animation preview")
        print("Phase 3: Headless recording")
        print("=" * 60 + "\n")

        was_saved = configure_scene_in_viewer(use_fresh=args.fresh, mode="positioning")

        if was_saved:
            print("\n" + "=" * 60)
            print("Positioning complete!")
            print("Now opening animation preview...")
            print("=" * 60 + "\n")

            configure_scene_in_viewer(use_fresh=False, mode="animation")

            print("\n" + "=" * 60)
            print("Animation preview complete!")
            print("Starting headless recording...")
            print("=" * 60 + "\n")

            # Load the saved scene configuration
            scene_config = load_scene_data()

            # Run headless recording
            frames_recorded = run_headless_recording(scene_config)

            if frames_recorded:
                print("\n" + "=" * 60)
                print("Headless recording complete!")
                print("Generating outputs...")
                print("=" * 60 + "\n")
                create_svg_from_scene()
            else:
                print(
                    "\nNo animation frames captured - only static SVG will be generated."
                )
                create_svg_from_scene()
        else:
            print("\nPositioning cancelled - no outputs will be generated.")
    else:
        was_saved = configure_scene_in_viewer(use_fresh=args.fresh, mode="positioning")
        if was_saved:
            create_svg_from_scene()
        else:
            print("\n" + "=" * 60)
            print("⚠️  WARNING: ANIMATION WAS STILL RUNNING! ⚠️")
            print("=" * 60)
            print("Nothing was saved because the rotation animation was")
            print("active when you closed the viewer.")
            print("")
            print("• Scene was NOT saved to scene.json")
            print("• SVG was NOT generated")
            print("")
            print("To save your work:")
            print("1. Run the program again")
            print("2. Press SPACEBAR to stop the animation")
            print("3. Then close the viewer")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
