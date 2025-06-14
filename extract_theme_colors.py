#!/usr/bin/env python3

import json
import requests
from typing import Dict, Any
import re
import math


def fetch_registry() -> Dict[str, Any]:
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


def extract_theme_colors(theme: Dict[str, Any]) -> Dict[str, Any]:
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


def generate_config_yaml_snippet(colors: Dict[str, Any]) -> str:
    """Generate a config.yaml style snippet for the theme."""
    snippet_lines = []

    # Light mode style
    if colors["light"]:
        snippet_lines.append(f'  - name: "{colors["name"]}-light"')
        snippet_lines.append(f'    background: "{colors["light"]["background"]}"')
        snippet_lines.append(f'    foreground: "{colors["light"]["foreground"]}"')

    # Dark mode style
    if colors["dark"]:
        if snippet_lines:  # Add separator if light mode exists
            snippet_lines.append("")
        snippet_lines.append(f'  - name: "{colors["name"]}-dark"')
        snippet_lines.append(f'    background: "{colors["dark"]["background"]}"')
        snippet_lines.append(f'    foreground: "{colors["dark"]["foreground"]}"')

    return "\n".join(snippet_lines)


def main():
    try:
        registry = fetch_registry()
        themes = registry.get("items", [])

        print(f"Found {len(themes)} themes in the registry\n")
        print("=" * 80)

        all_configs = []
        themes_json = {}

        for theme in themes:
            if theme.get("type") == "registry:style":
                colors = extract_theme_colors(theme)

                print(f"\nTheme: {colors['title']} ({colors['name']})")
                print("-" * 40)

                # Add light mode to JSON
                if colors["light"]:
                    theme_name = f"{colors['name']}-light"
                    themes_json[theme_name] = {
                        "foreground": colors["light"]["foreground"],
                        "background": colors["light"]["background"],
                    }
                    print("Light Mode:")
                    print(
                        f"  Background: {colors['light']['background']} (from: {colors['light']['background_raw']})"
                    )
                    print(
                        f"  Foreground: {colors['light']['foreground']} (from: {colors['light']['foreground_raw']})"
                    )

                # Add dark mode to JSON
                if colors["dark"]:
                    theme_name = f"{colors['name']}-dark"
                    themes_json[theme_name] = {
                        "foreground": colors["dark"]["foreground"],
                        "background": colors["dark"]["background"],
                    }
                    print("Dark Mode:")
                    print(
                        f"  Background: {colors['dark']['background']} (from: {colors['dark']['background_raw']})"
                    )
                    print(
                        f"  Foreground: {colors['dark']['foreground']} (from: {colors['dark']['foreground_raw']})"
                    )

                # Generate config snippet
                config_snippet = generate_config_yaml_snippet(colors)
                if config_snippet:
                    all_configs.append(config_snippet)

        # Save themes to JSON file
        with open("themes.json", "w") as f:
            json.dump(themes_json, f, indent=2)
        print(f"\nSaved {len(themes_json)} theme variations to themes.json")

        print("\n" + "=" * 80)
        print("\nCONFIG.YAML STRUCTURE:")
        print("=" * 80)
        print("\n# Common stroke settings for all styles")
        print("stroke_width: 12")
        print('stroke_linecap: "round"')
        print('stroke_linejoin: "round"')
        print("\n# Style variations for output")
        print("styles:")
        print("\n".join(all_configs))
        print("\n" + "=" * 80)

    except requests.RequestException as e:
        print(f"Error fetching registry: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
