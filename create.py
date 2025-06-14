#!/usr/bin/env python3
"""
Dodecahedron Creator - Interactive 3D viewer and SVG generator

This module has been refactored into multiple modules:
- cli.py: Command-line interface and workflow orchestration
- config.py: Configuration management and theme loading
- viewer.py: Interactive 3D visualization
- animation.py: Animation logic and timing
- svg_generator.py: SVG generation from 3D geometry
- utils.py: Shared utilities and mesh setup
"""

from cli import main

if __name__ == "__main__":
    main()
