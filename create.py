from vedo import Mesh, Plotter
import numpy as np
import svgwrite
from collections import defaultdict
import json
import os


def load_configuration():
    config_file = "scene.json"

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        print("Loaded scene from scene.json")
        return config
    else:
        print("Using default scene configuration")
        return {
            "camera": {
                "position": [2.8228959321975706, -4.437200307846069, 5.200783538818359],
                "focal_point": [-0.09484875202178955, -0.07701563835144043, 0.0],
                "view_up": [0.0, 0.0, 1.0],
                "view_angle": 30.0,
                "clipping_range": [3.696141151662736, 12.051587187261307],
            },
            "mesh": {
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "position": [0.0, 0.0, 0.0],
                "color": "black",
                "linewidth": 4,
                "edge_color": [1, 1, 1],
            },
            "viewport": {"background_color": "white"},
        }


def setup_mesh(config):
    mesh = Mesh("resources/dodecahedron.obj")

    transform_matrix = np.array(config["mesh"]["transform_matrix"])
    mesh.apply_transform(transform_matrix)
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
    camera.SetPosition(config["camera"]["position"])
    camera.SetFocalPoint(config["camera"]["focal_point"])
    camera.SetViewUp(config["camera"]["view_up"])
    camera.SetViewAngle(config["camera"]["view_angle"])
    camera.SetClippingRange(config["camera"]["clipping_range"])
    return camera


def run_interactive_session(plotter):
    plotter.interactive()


def save_configuration(plotter, mesh, config):
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
        "svg": {
            "stroke_width": 12,
            "background": "white",
            "fill": "black",
            "stroke": "white",
        },
    }

    with open("scene.json", "w") as f:
        json.dump(scene_data, f, indent=2)

    print("Saved scene to scene.json")


def configure_scene_in_viewer():
    config = load_configuration()

    mesh = setup_mesh(config)

    plotter = create_viewer(config)
    plotter.add(mesh)

    def handle_key_press(evt):
        if evt.keypress == "Up":
            cam = plotter.camera
            current_fov = cam.GetViewAngle()
            new_fov = min(current_fov + 5, 120)
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

    plotter.add_callback("on key press", handle_key_press)

    print("\n=== Keybindings ===")
    print("Camera Controls:")
    print("  Mouse Left   : Rotate camera")
    print("  Mouse Right  : Zoom/dolly")
    print("  Mouse Middle : Pan/translate")
    print("  Up Arrow     : Increase FOV by 5°")
    print("  Down Arrow   : Decrease FOV by 5°")
    print("\nUtility:")
    print("  q            : Quit and save scene")
    print("  r            : Reset camera")
    print("  s            : Screenshot")
    print("==================\n")

    config_file = "scene.json"
    if os.path.exists(config_file):
        setup_camera(plotter, config)
        plotter.show(axes=0, interactive=False, resetcam=False)
    else:
        plotter.show(axes=0, interactive=False, resetcam=True)

    run_interactive_session(plotter)

    save_configuration(plotter, mesh, config)

    plotter.close()


def load_scene_data():
    """Load the saved scene configuration."""
    with open("scene.json", "r") as f:
        return json.load(f)


def setup_offscreen_renderer(config):
    """Setup the offscreen renderer for SVG generation."""
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
    """Calculate which faces and edges are visible from the camera."""
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
    """Project 3D edges to 2D screen coordinates."""
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
    """Calculate 2D cross product for convex hull."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Calculate convex hull of 2D points."""
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


def generate_svg(projected_edges, config):
    """Generate SVG file from projected edges."""
    viewport_size = tuple(config["viewport"]["size"])
    svg_config = config["svg"]

    all_vertices = set()
    for edge in projected_edges:
        start = (edge["start_2d"][0], viewport_size[1] - edge["start_2d"][1])
        end = (edge["end_2d"][0], viewport_size[1] - edge["end_2d"][1])
        all_vertices.add(start)
        all_vertices.add(end)

    vertices_list = list(all_vertices)

    min_x = min(v[0] for v in vertices_list)
    max_x = max(v[0] for v in vertices_list)
    min_y = min(v[1] for v in vertices_list)
    max_y = max(v[1] for v in vertices_list)

    stroke_inset = svg_config["stroke_width"] / 2
    min_x += stroke_inset
    max_x -= stroke_inset
    min_y += stroke_inset
    max_y -= stroke_inset

    svg_width = max_x - min_x
    svg_height = max_y - min_y

    dwg = svgwrite.Drawing("dodecahedron.svg", size=(svg_width, svg_height))

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
                stroke_linecap="round",
                stroke_linejoin="round",
            )
        )

    dwg.save()
    print("SVG saved as dodecahedron.svg")


def create_svg_from_scene():
    """Main function to generate SVG from saved scene configuration."""
    print("Generating SVG...")

    config = load_scene_data()

    plotter, mesh, camera = setup_offscreen_renderer(config)

    visible_edges, vertices = calculate_visible_geometry(mesh, camera)

    projected_edges = project_to_2d(plotter, visible_edges, vertices)

    plotter.close()

    generate_svg(projected_edges, config)


def main():
    """Main entry point - reads like a recipe."""
    configure_scene_in_viewer()
    create_svg_from_scene()


if __name__ == "__main__":
    main()
