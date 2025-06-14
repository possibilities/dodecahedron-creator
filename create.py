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
import copy
import requests
import math

MODEL_PATH = "resources/dodecahedron.obj"
CONFIG_FILE_PATH = "config.yaml"
SHARED_SCENE_PATH = "build/shared_scene.json"

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

FRAME_JSON_PATTERN = "frame_*.json"
FRAME_SVG_PATTERN = "frame_*.svg"
FRAME_FILENAME_FORMAT = "frame_{:04d}.json"

BBOX_PADDING_MULTIPLIER = 2
IDENTITY_MATRIX_4X4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

# Common color names to RGB values
COLOR_MAP = {
    "black": [0, 0, 0],
    "white": [1, 1, 1],
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "yellow": [1, 1, 0],
    "cyan": [0, 1, 1],
    "magenta": [1, 0, 1],
    "orange": [1, 0.647, 0],
    "purple": [0.5, 0, 0.5],
    "brown": [0.647, 0.165, 0.165],
    "pink": [1, 0.753, 0.796],
    "gray": [0.5, 0.5, 0.5],
    "grey": [0.5, 0.5, 0.5],
    "lightblue": [0.678, 0.847, 0.902],
    "darkgreen": [0, 0.392, 0],
    "gold": [1, 0.843, 0],
    "silver": [0.753, 0.753, 0.753],
}


def fetch_registry() -> dict:
    url = "https://tweakcn.com/r/registry.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def hsl_to_hex(hsl_string: str) -> str:
    """Convert HSL string to hex color."""
    # Extract HSL values
    match = re.match(r"([\d.]+)\s+([\d.]+)%?\s+([\d.]+)%?", hsl_string.strip())
    if not match:
        return "#000000"

    h = float(match.group(1))
    s = float(match.group(2)) / 100
    lightness = float(match.group(3)) / 100

    # Convert HSL to RGB
    c = (1 - abs(2 * lightness - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = lightness - c / 2

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return f"#{r:02x}{g:02x}{b:02x}"


def oklch_to_hex(oklch_string: str) -> str:
    """Convert OKLCH to approximate hex (simplified conversion)."""
    # Extract OKLCH values
    match = re.match(
        r"oklch\(([\d.]+)(?:%?)\s+([\d.]+)\s+([\d.]+)\)", oklch_string.strip()
    )
    if not match:
        # Try without oklch prefix
        match = re.match(r"([\d.]+)(?:%?)\s+([\d.]+)\s+([\d.]+)", oklch_string.strip())
        if not match:
            return "#000000"

    lightness = float(match.group(1))
    c = float(match.group(2))
    h = float(match.group(3))

    # For grayscale (when chroma is 0), just use lightness
    if c == 0:
        gray = int(lightness * 255)
        return f"#{gray:02x}{gray:02x}{gray:02x}"

    # Very simplified OKLCH to RGB conversion
    # This is an approximation - proper conversion requires complex color space math

    # Convert to approximate RGB based on hue
    h_rad = h * 3.14159 / 180

    # Approximate RGB conversion
    if lightness > 1:
        lightness = lightness / 100  # Handle percentage

    # Base color from hue
    r = lightness + c * math.cos(h_rad)
    g = lightness + c * math.cos(h_rad - 2.094)
    b = lightness + c * math.cos(h_rad + 2.094)

    # Clamp and convert to 0-255
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))

    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_hex(rgb_string: str) -> str:
    """Convert RGB string to hex color."""
    match = re.match(r"([\d.]+)\s+([\d.]+)\s+([\d.]+)", rgb_string.strip())
    if not match:
        return "#000000"

    r = int(float(match.group(1)))
    g = int(float(match.group(2)))
    b = int(float(match.group(3)))

    return f"#{r:02x}{g:02x}{b:02x}"


def convert_color_to_hex(color_value: str) -> str:
    """Convert any CSS color format to hex."""
    color_value = color_value.strip()

    # Already hex
    if color_value.startswith("#"):
        return color_value

    # RGB format
    if any(x in color_value.lower() for x in ["rgb", "255"]):
        return rgb_to_hex(color_value)

    # OKLCH format
    if "oklch" in color_value.lower() or (
        len(color_value.split()) == 3 and any("." in x for x in color_value.split())
    ):
        parts = color_value.split()
        if len(parts) == 3 and all(any(c.isdigit() for c in p) for p in parts):
            return oklch_to_hex(color_value)

    # HSL format (default)
    return hsl_to_hex(color_value)


def extract_theme_colors(theme: dict) -> dict:
    result = {
        "name": theme.get("name", "Unknown"),
        "title": theme.get("title", "Unknown"),
        "light": None,
        "dark": None,
    }

    css_vars = theme.get("cssVars", {})

    if "light" in css_vars:
        light_vars = css_vars["light"]
        bg_raw = light_vars.get("background", "")
        fg_raw = light_vars.get("foreground", "")

        result["light"] = {
            "background": convert_color_to_hex(bg_raw),
            "foreground": convert_color_to_hex(fg_raw),
            "background_raw": bg_raw,
            "foreground_raw": fg_raw,
        }

    if "dark" in css_vars:
        dark_vars = css_vars["dark"]
        bg_raw = dark_vars.get("background", "")
        fg_raw = dark_vars.get("foreground", "")

        result["dark"] = {
            "background": convert_color_to_hex(bg_raw),
            "foreground": convert_color_to_hex(fg_raw),
            "background_raw": bg_raw,
            "foreground_raw": fg_raw,
        }

    return result


def generate_themes_data() -> dict:
    """Generate themes data by fetching from registry."""
    try:
        registry = fetch_registry()
        themes = registry.get("items", [])

        themes_data = {}

        for theme in themes:
            if theme.get("type") == "registry:style":
                colors = extract_theme_colors(theme)

                # Add light mode
                if colors["light"]:
                    theme_name = f"{colors['name']}-light"
                    themes_data[theme_name] = {
                        "foreground": colors["light"]["foreground"],
                        "background": colors["light"]["background"],
                    }

                # Add dark mode
                if colors["dark"]:
                    theme_name = f"{colors['name']}-dark"
                    themes_data[theme_name] = {
                        "foreground": colors["dark"]["foreground"],
                        "background": colors["dark"]["background"],
                    }

        return themes_data
    except Exception as e:
        print(f"Warning: Could not fetch themes from registry: {e}")
        return {}


def color_to_rgb(color):
    """Convert color name or hex to RGB values [0-1]. Returns the color unchanged if it's already RGB."""
    if isinstance(color, list) and len(color) == 3:
        return color
    if isinstance(color, str):
        # Check for hex color format
        if color.startswith("#"):
            # Remove the # and convert hex to RGB
            hex_color = color.lstrip("#")
            if len(hex_color) == 6:
                try:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    return [r, g, b]
                except ValueError:
                    pass
        # Check for color name
        color_lower = color.lower()
        if color_lower in COLOR_MAP:
            return COLOR_MAP[color_lower]
    # Default to white if color not found
    return [1, 1, 1]


def get_build_dir(style_name=None):
    """Get the build directory for a specific style."""
    if style_name:
        return os.path.join("build", style_name)
    return "build"


def get_svg_path(style_name):
    """Get the SVG output path for a specific style."""
    return os.path.join(get_build_dir(style_name), "dodecahedron.svg")


def get_scene_config_path(style_name):
    """Get the scene config path for a specific style."""
    return os.path.join(get_build_dir(style_name), "scene.json")


def get_frames_dir(style_name):
    """Get the frames directory for a specific style."""
    return os.path.join(get_build_dir(style_name), "frames")


def get_gif_path(style_name):
    """Get the GIF output path for a specific style."""
    return os.path.join(get_build_dir(style_name), "animation.gif")


class SvgGenerationContext:
    def __init__(self):
        self.global_bbox = None

    def reset(self):
        self.global_bbox = None

    def get_or_calculate_bbox(self, config):
        if self.global_bbox is None:
            self.global_bbox = get_global_bounding_box(config)
        return self.global_bbox


# Cache for themes to avoid fetching during animation
_themes_cache = None


def load_themes():
    """Load themes by fetching from registry (cached)."""
    global _themes_cache
    if _themes_cache is None:
        _themes_cache = generate_themes_data()
    return _themes_cache


def read_config_file():
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)

            # Load theme names
            theme_names = config.get("themes", [])
            if not theme_names:
                # Fallback to default themes if none defined
                theme_names = ["modern-minimal-light", "modern-minimal-dark"]

            # Load themes from themes.json
            available_themes = load_themes()

            # Convert themes to styles format
            styles = []
            for theme_name in theme_names:
                if theme_name in available_themes:
                    theme_data = available_themes[theme_name]
                    style = {
                        "name": theme_name,
                        "background": theme_data["background"],
                        "foreground": theme_data["foreground"],
                    }
                    styles.append(style)
                else:
                    print(f"Warning: Theme '{theme_name}' not found in themes.json")

            if not styles:
                print("Error: No valid themes found. Using defaults.")
                styles = [
                    {
                        "name": "black-on-white",
                        "background": "white",
                        "foreground": "black",
                    },
                    {
                        "name": "white-on-black",
                        "background": "black",
                        "foreground": "white",
                    },
                ]

            # Add common stroke settings to each style
            stroke_width = config.get("stroke_width", DEFAULT_SVG_STROKE_WIDTH)
            stroke_linecap = config.get("stroke_linecap", "round")
            stroke_linejoin = config.get("stroke_linejoin", "round")

            for style in styles:
                style["stroke_width"] = stroke_width
                style["stroke_linecap"] = stroke_linecap
                style["stroke_linejoin"] = stroke_linejoin
                # Fill and stroke derived from foreground/background
                style["fill"] = style.get("foreground", "black")
                style["stroke"] = style.get("background", "white")

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
                "stroke_width": stroke_width,
                "stroke_linecap": stroke_linecap,
                "stroke_linejoin": stroke_linejoin,
                "styles": styles,
                # Keep first style as default for backward compatibility
                "svg": styles[0]
                if styles
                else {
                    "stroke_width": stroke_width,
                    "background": DEFAULT_BACKGROUND_COLOR,
                    "fill": DEFAULT_SVG_FILL,
                    "stroke": DEFAULT_SVG_STROKE,
                    "stroke_linecap": stroke_linecap,
                    "stroke_linejoin": stroke_linejoin,
                },
            }
    except Exception:
        return None


def load_configuration(ignore_saved=False, scene_path=None, style=None):
    config_file = scene_path or SHARED_SCENE_PATH

    # Use provided style or get from config file
    if style:
        svg_settings = style
    else:
        config_from_file = read_config_file()
        svg_settings = (
            config_from_file.get(
                "svg",
                {
                    "stroke_width": DEFAULT_SVG_STROKE_WIDTH,
                    "background": DEFAULT_BACKGROUND_COLOR,
                    "fill": DEFAULT_SVG_FILL,
                    "stroke": DEFAULT_SVG_STROKE,
                    "stroke_linecap": "round",
                    "stroke_linejoin": "round",
                },
            )
            if config_from_file
            else {
                "stroke_width": DEFAULT_SVG_STROKE_WIDTH,
                "background": DEFAULT_BACKGROUND_COLOR,
                "fill": DEFAULT_SVG_FILL,
                "stroke": DEFAULT_SVG_STROKE,
                "stroke_linecap": "round",
                "stroke_linejoin": "round",
            }
        )

    if not ignore_saved and os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        # Always use fresh SVG settings from config.yaml
        config["svg"] = svg_settings
        # Update viewer colors from SVG settings
        config["viewport"]["background_color"] = svg_settings.get("background", "white")
        config["mesh"]["color"] = svg_settings.get("fill", "black")
        config["mesh"]["edge_color"] = color_to_rgb(svg_settings.get("stroke", "white"))
        print(f"Loaded scene from {config_file}")
        return config
    else:
        if ignore_saved:
            print("Using fresh scene configuration (--fresh flag)")
        else:
            print("Using default scene configuration")
        return {
            "camera": {},
            "mesh": {
                "color": svg_settings.get("fill", "black"),
                "linewidth": 4,
                "edge_color": color_to_rgb(svg_settings.get("stroke", "white")),
            },
            "viewport": {"background_color": svg_settings.get("background", "white")},
            "svg": svg_settings,
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


def save_configuration(plotter, mesh, config, animation_state, scene_path=None):
    camera = plotter.camera
    save_path = scene_path or SHARED_SCENE_PATH

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

    # Use SVG settings from config
    svg_settings = (
        config["svg"]
        if "svg" in config
        else {
            "stroke_width": DEFAULT_SVG_STROKE_WIDTH,
            "background": DEFAULT_BACKGROUND_COLOR,
            "fill": DEFAULT_SVG_FILL,
            "stroke": DEFAULT_SVG_STROKE,
            "stroke_linecap": "round",
            "stroke_linejoin": "round",
        }
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
        "svg": svg_settings,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(scene_data, f, indent=2)

    print(f"Saved scene to {save_path}")
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


def setup_scene_components(use_fresh=False, scene_path=None, style=None):
    config = load_configuration(
        ignore_saved=use_fresh, scene_path=scene_path, style=style
    )
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


def run_interactive_viewer(
    plotter, config, use_fresh, mode="positioning", scene_path=None
):
    config_file = scene_path if scene_path else get_scene_config_path(None)
    setup_camera(plotter, config)
    if not use_fresh and os.path.exists(config_file) and config["camera"]:
        plotter.show(axes=0, interactive=False, resetcam=False)
    else:
        plotter.show(axes=0, interactive=False, resetcam=True)

    run_interactive_session(plotter)


def configure_scene_in_viewer(
    use_fresh=False, mode="positioning", scene_path=None, style=None
):
    config, mesh, plotter = setup_scene_components(
        use_fresh, scene_path=scene_path, style=style
    )
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

    run_interactive_viewer(plotter, config, use_fresh, mode, scene_path)

    if mode == "positioning":
        was_saved = save_configuration(
            plotter, mesh, config, animation_state, scene_path=scene_path
        )
    else:
        was_saved = False

    plotter.close()
    return was_saved


def load_scene_data(scene_path=None):
    path = scene_path or SHARED_SCENE_PATH
    with open(path, "r") as f:
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


def setup_svg_renderer(json_path, context, style=None):
    with open(json_path, "r") as f:
        config = json.load(f)

    # If style is provided, merge it with the config
    if style:
        config["svg"] = style
        config["mesh"]["color"] = style.get("fill", "black")
        config["mesh"]["linewidth"] = 4
        config["mesh"]["edge_color"] = color_to_rgb(style.get("stroke", "white"))
        config["viewport"]["background_color"] = style.get("background", "white")

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


def create_svg_from_json(json_path, svg_path, context=None, style=None):
    if context is None:
        context = SvgGenerationContext()

    config, projected_edges = setup_svg_renderer(json_path, context, style)
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


def create_svg_from_scene(style_name=None):
    print(f"Generating SVG{' for style: ' + style_name if style_name else ''}...")

    context = SvgGenerationContext()

    scene_path = get_scene_config_path(style_name) if style_name else SHARED_SCENE_PATH
    svg_path = get_svg_path(style_name) if style_name else "build/dodecahedron.svg"

    # Load style information from scene.json if style_name is provided
    style = None
    if style_name:
        with open(scene_path, "r") as f:
            scene_config = json.load(f)
            style = scene_config.get("svg")

    # Generate the static SVG
    svg_output = create_svg_from_json(scene_path, svg_path, context)
    print(f"SVG saved as {svg_output}")

    # Always use shared frames directory
    shared_frames_dir = "build/frames"
    frame_json_files = sorted(
        glob.glob(os.path.join(shared_frames_dir, FRAME_JSON_PATTERN))
    )

    if frame_json_files and style_name:
        print(
            f"\nConverting {len(frame_json_files)} frame JSON files to SVG for style '{style_name}'..."
        )

        # Create style-specific frames directory for temporary SVG/PNG files
        style_frames_dir = get_frames_dir(style_name)
        os.makedirs(style_frames_dir, exist_ok=True)

        svg_paths = []
        for json_path in frame_json_files:
            frame_name = os.path.basename(json_path).replace(".json", ".svg")
            svg_path = os.path.join(style_frames_dir, frame_name)
            create_svg_from_json(json_path, svg_path, context, style)
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
                gif_path = get_gif_path(style_name)
                if create_animated_gif(
                    png_paths, gif_path, config["capture_fps"], config
                ):
                    # Clean up temporary frame SVGs and PNGs
                    print("\nCleaning up temporary frame files...")
                    for svg_path in svg_paths:
                        if os.path.exists(svg_path):
                            os.remove(svg_path)
                    for png_path in png_paths:
                        if os.path.exists(png_path):
                            os.remove(png_path)
                    print("Cleanup complete!")


def run_headless_recording(config):
    """Run a headless animation cycle to capture frames"""
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

    # Pre-fetch themes before anything else to avoid network calls during animation
    print("Fetching themes from registry...")
    load_themes()

    config = read_config_file()
    if not config:
        print("Error: Could not read config file")
        return

    capture_enabled = config.get("capture_fps", 0) > 0
    styles = config.get("styles", [])

    if not styles:
        print("Error: No themes loaded from configuration")
        return

    # Use first style for interactive positioning
    first_style = styles[0]

    print(f"\nConfigured themes: {', '.join(s['name'] for s in styles)}")
    print(f"Using '{first_style['name']}' theme for positioning\n")

    if capture_enabled:
        print("=" * 60)
        print("MULTI-STYLE WORKFLOW WITH ANIMATION")
        print("=" * 60)
        print("Phase 1: Camera positioning (once)")
        print("Phase 2: Animation preview (once)")
        print("Phase 3: Generate outputs for all themes")
        print("=" * 60 + "\n")
    else:
        print("=" * 60)
        print("MULTI-STYLE WORKFLOW")
        print("=" * 60)
        print("Phase 1: Camera positioning (once)")
        print("Phase 2: Generate outputs for all themes")
        print("=" * 60 + "\n")

    # Phase 1: Interactive positioning with first style
    was_saved = configure_scene_in_viewer(
        use_fresh=args.fresh,
        mode="positioning",
        scene_path=SHARED_SCENE_PATH,
        style=first_style,
    )

    if not was_saved:
        print("\nPositioning cancelled - no outputs will be generated.")
        return

    # Phase 2: Animation preview (if enabled)
    if capture_enabled:
        print("\n" + "=" * 60)
        print("Positioning complete!")
        print("Now opening animation preview...")
        print("=" * 60 + "\n")

        configure_scene_in_viewer(
            use_fresh=False,
            mode="animation",
            scene_path=SHARED_SCENE_PATH,
            style=first_style,
        )

    # Phase 3: Generate outputs for all themes
    print("\n" + "=" * 60)
    print("Generating outputs for all themes...")
    print("=" * 60 + "\n")

    # Load the shared positioning
    shared_config = load_scene_data(SHARED_SCENE_PATH)

    # Run headless recording once if animation is enabled
    if capture_enabled:
        frames_recorded = run_headless_recording(shared_config)
        if not frames_recorded:
            print("\nNo animation frames captured")
            capture_enabled = False  # Disable animation for styles if no frames

    for style in styles:
        print(f"\n{'=' * 40}")
        print(f"Processing theme: {style['name']}")
        print(f"{'=' * 40}\n")

        # Create style-specific config by merging positioning with style colors
        style_config = copy.deepcopy(shared_config)
        style_config["svg"] = style
        style_config["mesh"]["color"] = style.get("fill", "black")
        style_config["mesh"]["edge_color"] = color_to_rgb(style.get("stroke", "white"))
        style_config["viewport"]["background_color"] = style.get("background", "white")

        # Save style-specific scene.json
        style_dir = get_build_dir(style["name"])
        os.makedirs(style_dir, exist_ok=True)
        style_scene_path = get_scene_config_path(style["name"])
        with open(style_scene_path, "w") as f:
            json.dump(style_config, f, indent=2)
        print(f"Saved style config to {style_scene_path}")

        # Generate SVG outputs
        create_svg_from_scene(style["name"])

    print("\n" + "=" * 60)
    print("ALL THEMES COMPLETE!")
    print("=" * 60)
    print("\nOutput directories:")
    for style in styles:
        print(f"  - build/{style['name']}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
