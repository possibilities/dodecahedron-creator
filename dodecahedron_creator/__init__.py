"""Dodecahedron Creator - Interactive 3D viewer and SVG generator"""

from .config import read_config_file, load_themes
from .viewer import configure_scene_in_viewer
from .animation import run_headless_recording
from .svg import create_svg_from_scene

__version__ = "0.1.0"

__all__ = [
    "read_config_file",
    "load_themes",
    "configure_scene_in_viewer",
    "run_headless_recording",
    "create_svg_from_scene",
    "__version__",
]
