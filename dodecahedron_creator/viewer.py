# Interactive 3D visualization and user controls

import os
import time
from vedo import Plotter, Text2D

from .config import (
    save_configuration,
    get_scene_config_path,
)
from .animation import (
    setup_animation_state,
    handle_timer,
)
from .utils import setup_camera

# Viewer constants
FOV_ADJUSTMENT_STEP = 5
MAX_FOV_DEGREES = 120
MIN_FOV_DEGREES = 5
DEFAULT_VIEWER_SIZE = (1400, 900)
DEGREES_PER_ROTATION = 360.0
MS_PER_SECOND = 1000.0


def create_viewer(config):
    plotter = Plotter(size=(1400, 900), bg=config["viewport"]["background_color"])
    return plotter


def run_interactive_session(plotter):
    plotter.interactive()


def handle_key_press(
    evt, plotter, animation_state, mesh, config, mode="positioning", model_name=None
):
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


def register_event_handlers(
    plotter, mesh, config, animation_state, mode="positioning", model_name=None
):
    plotter.add_callback(
        "on key press",
        lambda evt: handle_key_press(
            evt, plotter, animation_state, mesh, config, mode, model_name
        ),
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
    use_fresh=False, mode="positioning", scene_path=None, style=None, model_name=None
):
    # Import here to avoid circular dependency
    from .utils import setup_scene_components

    config, mesh, plotter = setup_scene_components(
        use_fresh, scene_path=scene_path, style=style, model_name=model_name
    )
    animation_state = setup_animation_state()
    register_event_handlers(plotter, mesh, config, animation_state, mode, model_name)

    if mode == "animation":
        print("\n" + "=" * 60)
        print("ANIMATION PREVIEW MODE")
        print("=" * 60)
        print("Animation will start automatically")
        print("Close the window to proceed to recording")
        print("=" * 60 + "\n")

        mode_text = Text2D(
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

        mode_text = Text2D(
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
            plotter,
            mesh,
            config,
            animation_state,
            scene_path=scene_path,
            model_name=model_name,
        )
    else:
        was_saved = False

    plotter.close()
    return was_saved
