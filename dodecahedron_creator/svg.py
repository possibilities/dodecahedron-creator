# SVG generation from 3D geometry

import os
import json
import glob
import re
import numpy as np
import svgwrite
from collections import defaultdict
import cairosvg
from PIL import Image

from .config import (
    get_scene_config_path,
    get_svg_path,
    get_frames_dir,
    get_gif_path,
    color_to_rgb,
    read_config_file,
)

# SVG-related constants
FRAME_JSON_PATTERN = "frame_*.json"
FRAME_SVG_PATTERN = "frame_*.svg"
BBOX_PADDING_MULTIPLIER = 2
VIEWER_FPS = 50
DEGREES_PER_ROTATION = 360.0
MS_PER_SECOND = 1000.0
MIN_GIF_FRAME_DURATION_MS = 20


class SvgGenerationContext:
    def __init__(self):
        self.global_bbox = None

    def reset(self):
        self.global_bbox = None

    def get_or_calculate_bbox(self, config, model_name=None):
        if self.global_bbox is None:
            self.global_bbox = get_global_bounding_box(config, model_name)
        return self.global_bbox


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


def get_global_bounding_box(config, model_name=None):
    # Import here to avoid circular dependency
    from .utils import setup_offscreen_renderer

    plotter, mesh, camera = setup_offscreen_renderer(config, model_name)

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


def setup_svg_renderer(json_path, context, style=None, model_name=None):
    # Import here to avoid circular dependency
    from .utils import setup_offscreen_renderer

    with open(json_path, "r") as f:
        config = json.load(f)

    # If style is provided, merge it with the config
    if style:
        config["svg"] = style
        config["mesh"]["color"] = style.get("fill", "black")
        config["mesh"]["linewidth"] = 4
        config["mesh"]["edge_color"] = color_to_rgb(style.get("stroke", "white"))
        config["viewport"]["background_color"] = style.get("background", "white")

    context.get_or_calculate_bbox(config, model_name)

    plotter, mesh, camera = setup_offscreen_renderer(config, model_name)
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


def create_svg_from_json(
    json_path, svg_path, context=None, style=None, model_name=None
):
    if context is None:
        context = SvgGenerationContext()

    config, projected_edges = setup_svg_renderer(json_path, context, style, model_name)
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


def create_svg_from_scene(model_name, style_name):
    print(f"Generating SVG for {model_name} with style: {style_name}...")

    context = SvgGenerationContext()

    scene_path = get_scene_config_path(model_name, style_name)
    svg_path = get_svg_path(model_name, style_name)

    # Load style information from scene.json if style_name is provided
    style = None
    if style_name:
        with open(scene_path, "r") as f:
            scene_config = json.load(f)
            style = scene_config.get("svg")

    # Generate the static SVG
    svg_output = create_svg_from_json(
        scene_path, svg_path, context, model_name=model_name
    )
    print(f"SVG saved as {svg_output}")

    # Use model-specific frames directory
    model_frames_dir = f"build/{model_name}/frames"
    frame_json_files = sorted(
        glob.glob(os.path.join(model_frames_dir, FRAME_JSON_PATTERN))
    )

    if frame_json_files:
        print(
            f"\nConverting {len(frame_json_files)} frame JSON files to SVG for style '{style_name}'..."
        )

        # Create model/style-specific frames directory for temporary SVG/PNG files
        style_frames_dir = get_frames_dir(model_name, style_name)
        os.makedirs(style_frames_dir, exist_ok=True)

        svg_paths = []
        for json_path in frame_json_files:
            frame_name = os.path.basename(json_path).replace(".json", ".svg")
            svg_path = os.path.join(style_frames_dir, frame_name)
            create_svg_from_json(json_path, svg_path, context, style, model_name)
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
                gif_path = get_gif_path(model_name, style_name)
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
