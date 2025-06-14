# Configuration management, theme loading, and color utilities

import os
import json
import yaml
import re
import math
import requests

# File paths
MODEL_PATH = "resources/dodecahedron.obj"
CONFIG_FILE_PATH = "config.yaml"
SHARED_SCENE_PATH = "build/shared_scene.json"

# Default values
DEFAULT_SVG_STROKE_WIDTH = 12
DEFAULT_MESH_LINE_WIDTH = 4
DEFAULT_BACKGROUND_COLOR = "white"
DEFAULT_MESH_COLOR = "black"
DEFAULT_EDGE_COLOR = [1, 1, 1]
DEFAULT_SVG_FILL = "black"
DEFAULT_SVG_STROKE = "white"

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

# Cache for themes to avoid fetching during animation
_themes_cache = None


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


def load_scene_data(scene_path=None):
    path = scene_path or SHARED_SCENE_PATH
    with open(path, "r") as f:
        return json.load(f)
