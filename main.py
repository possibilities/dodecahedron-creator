from vedo import Mesh, Plotter, Text2D, Box
import numpy as np
import svgwrite
from collections import defaultdict
import math
import json
import os

config_file = "scene.json"
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    CAMERA_POSITION = config["camera"]["position"]
    CAMERA_FOCAL_POINT = config["camera"]["focal_point"]
    CAMERA_VIEW_UP = config["camera"]["view_up"]
    CAMERA_VIEW_ANGLE = config["camera"]["view_angle"]
    CAMERA_CLIPPING_RANGE = config["camera"]["clipping_range"]

    TRANSFORM_MATRIX = np.array(config["mesh"]["transform_matrix"])
    MESH_POSITION = config["mesh"]["position"]
    MESH_COLOR = config["mesh"]["color"]
    MESH_LINEWIDTH = config["mesh"]["linewidth"]
    EDGE_COLOR = tuple(config["mesh"]["edge_color"])
    BACKGROUND_COLOR = config["viewport"]["background_color"]

    print("Loaded existing scene configuration from scene.json")
else:
    CAMERA_POSITION = [2.8228959321975706, -4.437200307846069, 5.200783538818359]
    CAMERA_FOCAL_POINT = [-0.09484875202178955, -0.07701563835144043, 0.0]
    CAMERA_VIEW_UP = [0.0, 0.0, 1.0]
    CAMERA_VIEW_ANGLE = 30.0
    CAMERA_CLIPPING_RANGE = [3.696141151662736, 12.051587187261307]

    TRANSFORM_MATRIX = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    MESH_POSITION = [np.float64(0.0), np.float64(0.0), np.float64(0.0)]

    MESH_COLOR = "black"
    MESH_LINEWIDTH = 4
    EDGE_COLOR = (1, 1, 1)
    BACKGROUND_COLOR = "white"

    print("Using default scene configuration")

mesh = Mesh("resources/dodecahedron.obj")
mesh.apply_transform(TRANSFORM_MATRIX)
mesh.pos(MESH_POSITION)

mesh.color("black")
mesh.lighting("off")
mesh.flat()

mesh.linewidth(MESH_LINEWIDTH)
mesh.properties.EdgeVisibilityOn()
mesh.properties.SetEdgeColor(1, 1, 1)
mesh.properties.SetLineWidth(MESH_LINEWIDTH)

plotter = Plotter(size=(1400, 900), bg=BACKGROUND_COLOR)

plotter.add(mesh)

camera = plotter.camera
camera.SetPosition(CAMERA_POSITION)
camera.SetFocalPoint(CAMERA_FOCAL_POINT)
camera.SetViewUp(CAMERA_VIEW_UP)
camera.SetViewAngle(CAMERA_VIEW_ANGLE)
camera.SetClippingRange(CAMERA_CLIPPING_RANGE)

plotter.show(axes=0, interactive=False, resetcam=False)

control_bg = Box(pos=(0.25, 0.5, 0), width=0.48, height=0.95, length=0.001)
control_bg.color("lightgray").alpha(0.9)
plotter.add(control_bg)

title = Text2D(
    "CAMERA CONTROLS", pos=(0.25, 0.92), s=1.2, c="black", bold=True, justify="center"
)
plotter.add(title)

slider_width = 0.35
slider_x_start = 0.08
slider_x_end = slider_x_start + slider_width
slider_y = 0.82
slider_spacing = 0.09

slider_values = []


def fov_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    camera.SetViewAngle(value)
    slider_values[0].text(f"FOV: {value:.1f}°")
    plotter.render()


fov_value = Text2D(
    f"FOV: {camera.GetViewAngle():.1f}°",
    pos=(0.25, slider_y + 0.03),
    s=0.9,
    c="black",
    justify="center",
)
plotter.add(fov_value)
slider_values.append(fov_value)

plotter.add_slider(
    fov_slider,
    10,
    120,
    camera.GetViewAngle(),
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)

slider_y -= slider_spacing


def distance_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    cam_pos = np.array(camera.GetPosition())
    focal_point = np.array(camera.GetFocalPoint())
    direction = cam_pos - focal_point
    direction = direction / np.linalg.norm(direction)
    new_pos = focal_point + direction * value
    camera.SetPosition(new_pos)
    slider_values[1].text(f"Distance: {value:.2f}")
    plotter.render()


current_distance = np.linalg.norm(
    np.array(camera.GetPosition()) - np.array(camera.GetFocalPoint())
)
dist_value = Text2D(
    f"Distance: {current_distance:.2f}",
    pos=(0.25, slider_y + 0.03),
    s=0.9,
    c="black",
    justify="center",
)
plotter.add(dist_value)
slider_values.append(dist_value)

plotter.add_slider(
    distance_slider,
    0.5,
    20.0,
    current_distance,
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)

slider_y -= slider_spacing


def azimuth_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    camera.Azimuth(value - widget.angle_last)
    widget.angle_last = value
    slider_values[2].text(f"Azimuth: {value:.1f}°")
    plotter.render()


azimuth_value = Text2D(
    "Azimuth: 0.0°", pos=(0.25, slider_y + 0.03), s=0.9, c="black", justify="center"
)
plotter.add(azimuth_value)
slider_values.append(azimuth_value)

azimuth_widget = plotter.add_slider(
    azimuth_slider,
    -180,
    180,
    0,
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)
azimuth_widget.angle_last = 0

slider_y -= slider_spacing


def elevation_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    camera.Elevation(value - widget.angle_last)
    widget.angle_last = value
    slider_values[3].text(f"Elevation: {value:.1f}°")
    plotter.render()


elevation_value = Text2D(
    "Elevation: 0.0°", pos=(0.25, slider_y + 0.03), s=0.9, c="black", justify="center"
)
plotter.add(elevation_value)
slider_values.append(elevation_value)

elevation_widget = plotter.add_slider(
    elevation_slider,
    -90,
    90,
    0,
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)
elevation_widget.angle_last = 0

slider_y -= slider_spacing


def roll_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    camera.Roll(value - widget.angle_last)
    widget.angle_last = value
    slider_values[4].text(f"Roll: {value:.1f}°")
    plotter.render()


roll_value = Text2D(
    "Roll: 0.0°", pos=(0.25, slider_y + 0.03), s=0.9, c="black", justify="center"
)
plotter.add(roll_value)
slider_values.append(roll_value)

roll_widget = plotter.add_slider(
    roll_slider,
    -180,
    180,
    0,
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)
roll_widget.angle_last = 0

slider_y -= slider_spacing * 1.5
focal_title = Text2D(
    "FOCAL POINT", pos=(0.25, slider_y), s=1.2, c="black", bold=True, justify="center"
)
plotter.add(focal_title)

slider_y -= slider_spacing * 0.7


def focal_x_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    focal = list(camera.GetFocalPoint())
    focal[0] = value
    camera.SetFocalPoint(focal)
    slider_values[5].text(f"X: {value:.3f}")
    plotter.render()


focal_x_value = Text2D(
    f"X: {camera.GetFocalPoint()[0]:.3f}",
    pos=(0.25, slider_y + 0.03),
    s=0.9,
    c="black",
    justify="center",
)
plotter.add(focal_x_value)
slider_values.append(focal_x_value)

plotter.add_slider(
    focal_x_slider,
    -5.0,
    5.0,
    camera.GetFocalPoint()[0],
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)

slider_y -= slider_spacing


def focal_y_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    focal = list(camera.GetFocalPoint())
    focal[1] = value
    camera.SetFocalPoint(focal)
    slider_values[6].text(f"Y: {value:.3f}")
    plotter.render()


focal_y_value = Text2D(
    f"Y: {camera.GetFocalPoint()[1]:.3f}",
    pos=(0.25, slider_y + 0.03),
    s=0.9,
    c="black",
    justify="center",
)
plotter.add(focal_y_value)
slider_values.append(focal_y_value)

plotter.add_slider(
    focal_y_slider,
    -5.0,
    5.0,
    camera.GetFocalPoint()[1],
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)

slider_y -= slider_spacing


def focal_z_slider(widget, event):
    value = widget.GetRepresentation().GetValue()
    focal = list(camera.GetFocalPoint())
    focal[2] = value
    camera.SetFocalPoint(focal)
    slider_values[7].text(f"Z: {value:.3f}")
    plotter.render()


focal_z_value = Text2D(
    f"Z: {camera.GetFocalPoint()[2]:.3f}",
    pos=(0.25, slider_y + 0.03),
    s=0.9,
    c="black",
    justify="center",
)
plotter.add(focal_z_value)
slider_values.append(focal_z_value)

plotter.add_slider(
    focal_z_slider,
    -5.0,
    5.0,
    camera.GetFocalPoint()[2],
    pos=([slider_x_start, slider_y], [slider_x_end, slider_y]),
    title="",
    c="darkgray",
    show_value=False,
)

help_text = Text2D(
    "Mouse: Rotate/Zoom | Shift+Mouse: Pan\nClose window to generate SVG",
    pos=(0.25, 0.08),
    s=0.7,
    c="dimgray",
    justify="center",
)
plotter.add(help_text)

plotter.interactive()

final_position = list(camera.GetPosition())
final_focal_point = list(camera.GetFocalPoint())
final_view_up = list(camera.GetViewUp())
final_view_angle = camera.GetViewAngle()
final_clipping_range = list(camera.GetClippingRange())

final_mesh_position = list(mesh.pos())

transform_matrix = (
    mesh.transform.matrix.tolist()
    if hasattr(mesh.transform, "matrix")
    else TRANSFORM_MATRIX.tolist()
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
        "color": MESH_COLOR,
        "linewidth": MESH_LINEWIDTH,
        "edge_color": list(EDGE_COLOR),
    },
    "viewport": {
        "size": list(plotter.window.GetSize()),
        "background_color": BACKGROUND_COLOR,
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

print("\nScene configuration saved to scene.json")

plotter.close()

print("Generating SVG...")

with open("scene.json", "r") as f:
    config = json.load(f)

CAMERA_POSITION = config["camera"]["position"]
CAMERA_FOCAL_POINT = config["camera"]["focal_point"]
CAMERA_VIEW_UP = config["camera"]["view_up"]
CAMERA_VIEW_ANGLE = config["camera"]["view_angle"]
CAMERA_CLIPPING_RANGE = config["camera"]["clipping_range"]

TRANSFORM_MATRIX = np.array(config["mesh"]["transform_matrix"])
MESH_POSITION = config["mesh"]["position"]
MESH_COLOR = config["mesh"]["color"]
MESH_LINEWIDTH = config["mesh"]["linewidth"]
EDGE_COLOR = tuple(config["mesh"]["edge_color"])

BACKGROUND_COLOR = config["viewport"]["background_color"]
VIEWPORT_SIZE = tuple(config["viewport"]["size"])

SVG_STROKE_WIDTH = config["svg"]["stroke_width"]
SVG_BACKGROUND = config["svg"]["background"]
SVG_FILL = config["svg"]["fill"]
SVG_STROKE = config["svg"]["stroke"]

mesh = Mesh("resources/dodecahedron.obj")

mesh.apply_transform(TRANSFORM_MATRIX)
mesh.pos(MESH_POSITION)

mesh.color(MESH_COLOR)
mesh.linewidth(MESH_LINEWIDTH)
mesh.properties.EdgeVisibilityOn()
mesh.properties.SetEdgeColor(*EDGE_COLOR)
mesh.properties.SetLineWidth(MESH_LINEWIDTH)

plotter = Plotter(bg=BACKGROUND_COLOR, offscreen=True, size=VIEWPORT_SIZE)

plotter.add(mesh)

camera = plotter.camera
camera.SetPosition(CAMERA_POSITION)
camera.SetFocalPoint(CAMERA_FOCAL_POINT)
camera.SetViewUp(CAMERA_VIEW_UP)
camera.SetViewAngle(CAMERA_VIEW_ANGLE)
camera.SetClippingRange(CAMERA_CLIPPING_RANGE)

plotter.show(axes=0, interactive=False, resetcam=False)

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

renderer = plotter.renderer

projected_edges = []
for edge in final_edges:
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

plotter.close()

viewport_size = VIEWPORT_SIZE

edges = projected_edges

all_vertices = set()
for edge in edges:
    start = (edge["start_2d"][0], viewport_size[1] - edge["start_2d"][1])
    end = (edge["end_2d"][0], viewport_size[1] - edge["end_2d"][1])
    all_vertices.add(start)
    all_vertices.add(end)

vertices_list = list(all_vertices)

min_x = min(v[0] for v in vertices_list)
max_x = max(v[0] for v in vertices_list)
min_y = min(v[1] for v in vertices_list)
max_y = max(v[1] for v in vertices_list)

stroke_inset = SVG_STROKE_WIDTH / 2
min_x += stroke_inset
max_x -= stroke_inset
min_y += stroke_inset
max_y -= stroke_inset

svg_width = max_x - min_x
svg_height = max_y - min_y

dwg = svgwrite.Drawing("dodecahedron.svg", size=(svg_width, svg_height))

dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill=SVG_BACKGROUND))
center_x = sum(v[0] for v in vertices_list) / len(vertices_list)
center_y = sum(v[1] for v in vertices_list) / len(vertices_list)


def angle_from_center(point):
    return math.atan2(point[1] - center_y, point[0] - center_x)


vertices_sorted = sorted(vertices_list, key=angle_from_center)


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


hull_vertices = convex_hull(vertices_list)

translated_hull = [(x - min_x, y - min_y) for x, y in hull_vertices]

dwg.add(dwg.polygon(points=translated_hull, fill=SVG_FILL, stroke="none"))

vertex_map = defaultdict(list)
edge_used = [False] * len(edges)

for i, edge in enumerate(edges):
    start = (edge["start_2d"][0], edge["start_2d"][1])
    end = (edge["end_2d"][0], edge["end_2d"][1])
    vertex_map[start].append((i, end))
    vertex_map[end].append((i, start))

paths = []
for i, edge in enumerate(edges):
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
            stroke=SVG_STROKE,
            stroke_width=SVG_STROKE_WIDTH,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )

dwg.save()

print(
    f"SVG saved as dodecahedron.svg with {SVG_BACKGROUND} background and {SVG_FILL} fill"
)
