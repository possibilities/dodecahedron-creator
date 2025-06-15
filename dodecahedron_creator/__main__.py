# Command-line interface and workflow orchestration

import argparse
import os
import json
import copy

from .config import (
    read_config_file,
    load_themes,
    load_scene_data,
    color_to_rgb,
    get_build_dir,
    get_scene_config_path,
    get_shared_scene_path,
)
from .viewer import configure_scene_in_viewer
from .animation import run_headless_recording
from .svg import create_svg_from_scene
from .models import cleanup_temp_models


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D polyhedron viewer and SVG generator"
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
    parser.add_argument(
        "--only-model",
        action="append",
        dest="only_models",
        help="Generate only specific model(s). Can be used multiple times.",
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
    models = config.get("models", [])

    if not styles:
        print("Error: No themes loaded from configuration")
        return

    if not models:
        print("Error: No models loaded from configuration")
        return

    # Filter models if --only-model is used
    if args.only_models:
        # Validate all requested models exist
        invalid_models = [model for model in args.only_models if model not in models]
        if invalid_models:
            print(
                f"Error: The following models do not exist: {', '.join(invalid_models)}"
            )
            print(f"Available models: {', '.join(models)}")
            return

        # Filter models to only include requested ones
        models = [model for model in models if model in args.only_models]
        print(f"\nProcessing only selected models: {', '.join(args.only_models)}")
    else:
        print(f"\nConfigured models: {', '.join(models)}")

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

    # Use first model and style for interactive positioning
    first_model = models[0]
    first_style = styles[0]

    print(
        f"Using '{first_model}' model with '{first_style['name']}' theme for positioning\n"
    )

    total_combinations = len(models) * len(styles)

    if capture_enabled:
        print("=" * 60)
        print("MULTI-MODEL & MULTI-STYLE WORKFLOW WITH ANIMATION")
        print("=" * 60)
        print("Phase 1: Camera positioning (once per model)")
        print("Phase 2: Animation preview (once per model)")
        print(f"Phase 3: Generate outputs for {total_combinations} combinations")
        print(f"         ({len(models)} models × {len(styles)} themes)")
        print("=" * 60 + "\n")
    else:
        print("=" * 60)
        print("MULTI-MODEL & MULTI-STYLE WORKFLOW")
        print("=" * 60)
        print("Phase 1: Camera positioning (once per model)")
        print(f"Phase 2: Generate outputs for {total_combinations} combinations")
        print(f"         ({len(models)} models × {len(styles)} themes)")
        print("=" * 60 + "\n")

    # Process each model
    for model_idx, model_name in enumerate(models):
        print(f"\n{'#' * 60}")
        print(f"PROCESSING MODEL {model_idx + 1}/{len(models)}: {model_name}")
        print(f"{'#' * 60}\n")

        # Phase 1: Interactive positioning for this model
        model_scene_path = get_shared_scene_path(model_name)

        was_saved = configure_scene_in_viewer(
            use_fresh=args.fresh,
            mode="positioning",
            scene_path=model_scene_path,
            style=first_style,
            model_name=model_name,
        )

        if not was_saved:
            print(f"\nPositioning cancelled for {model_name} - skipping this model.")
            continue

        # Phase 2: Animation preview (if enabled)
        if capture_enabled:
            print("\n" + "=" * 60)
            print(f"Positioning complete for {model_name}!")
            print("Now opening animation preview...")
            print("=" * 60 + "\n")

            configure_scene_in_viewer(
                use_fresh=False,
                mode="animation",
                scene_path=model_scene_path,
                style=first_style,
                model_name=model_name,
            )

        # Phase 3: Generate outputs for all themes for this model
        print("\n" + "=" * 60)
        print(f"Generating outputs for {model_name} with all themes...")
        print("=" * 60 + "\n")

        # Load the model-specific positioning
        model_config = load_scene_data(model_scene_path)

        # Run headless recording once per model if animation is enabled
        if capture_enabled:
            frames_recorded = run_headless_recording(model_config, model_name)
            if not frames_recorded:
                print(f"\nNo animation frames captured for {model_name}")

        for style in styles:
            print(f"\n{'=' * 40}")
            print(f"Processing: {model_name} + {style['name']}")
            print(f"{'=' * 40}\n")

            # Create style-specific config by merging positioning with style colors
            style_config = copy.deepcopy(model_config)
            style_config["svg"] = style
            style_config["mesh"]["color"] = style.get("fill", "black")
            style_config["mesh"]["edge_color"] = color_to_rgb(
                style.get("stroke", "white")
            )
            style_config["viewport"]["background_color"] = style.get(
                "background", "white"
            )

            # Save style-specific scene.json
            style_dir = get_build_dir(model_name, style["name"])
            os.makedirs(style_dir, exist_ok=True)
            style_scene_path = get_scene_config_path(model_name, style["name"])
            with open(style_scene_path, "w") as f:
                json.dump(style_config, f, indent=2)
            print(f"Saved style config to {style_scene_path}")

            # Generate SVG outputs
            create_svg_from_scene(model_name, style["name"])

    print("\n" + "=" * 60)
    print("ALL MODELS AND THEMES COMPLETE!")
    print("=" * 60)
    print("\nOutput directories:")
    for model in models:
        for style in styles:
            print(f"  - build/{model}/{style['name']}/")
    print("=" * 60 + "\n")

    # Clean up temporary OBJ files
    cleanup_temp_models()


if __name__ == "__main__":
    main()
