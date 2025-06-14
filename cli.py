# Command-line interface and workflow orchestration

import argparse
import os
import json
import copy

from config import (
    read_config_file,
    load_themes,
    load_scene_data,
    color_to_rgb,
    get_build_dir,
    get_scene_config_path,
    SHARED_SCENE_PATH,
)
from viewer import configure_scene_in_viewer
from animation import run_headless_recording
from svg_generator import create_svg_from_scene


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D dodecahedron viewer and SVG generator"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with default view, ignoring saved scene.json",
    )
    parser.add_argument(
        "--only-theme",
        action="append",
        dest="only_themes",
        help="Generate only specific theme(s). Can be used multiple times.",
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

    # Filter themes if --only-theme is used
    if args.only_themes:
        # Create a mapping of theme names to style objects
        available_themes = {style["name"]: style for style in styles}

        # Validate all requested themes exist
        invalid_themes = [
            theme for theme in args.only_themes if theme not in available_themes
        ]
        if invalid_themes:
            print(
                f"Error: The following themes do not exist: {', '.join(invalid_themes)}"
            )
            print(f"Available themes: {', '.join(available_themes.keys())}")
            return

        # Filter styles to only include requested themes
        styles = [available_themes[theme] for theme in args.only_themes]
        print(f"\nProcessing only selected themes: {', '.join(args.only_themes)}")
    else:
        print(f"\nConfigured themes: {', '.join(s['name'] for s in styles)}")

    # Use first style for interactive positioning
    first_style = styles[0]

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
