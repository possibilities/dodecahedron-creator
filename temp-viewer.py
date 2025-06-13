from vedo import Mesh, Plotter
import numpy as np
import json
import os

# Load scene configuration
config_file = "scene.json"
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    
    # Extract camera settings
    CAMERA_POSITION = config["camera"]["position"]
    CAMERA_FOCAL_POINT = config["camera"]["focal_point"]
    CAMERA_VIEW_UP = config["camera"]["view_up"]
    CAMERA_VIEW_ANGLE = config["camera"]["view_angle"]
    CAMERA_CLIPPING_RANGE = config["camera"]["clipping_range"]
    
    # Extract mesh settings
    TRANSFORM_MATRIX = np.array(config["mesh"]["transform_matrix"])
    MESH_POSITION = config["mesh"]["position"]
    MESH_COLOR = config["mesh"]["color"]
    MESH_LINEWIDTH = config["mesh"]["linewidth"]
    EDGE_COLOR = tuple(config["mesh"]["edge_color"])
    
    # Extract viewport settings
    BACKGROUND_COLOR = config["viewport"]["background_color"]
    VIEWPORT_SIZE = tuple(config["viewport"]["size"])
    
    print("Loaded scene configuration from scene.json")
else:
    print("Error: scene.json not found!")
    exit(1)

# Load and configure the mesh
mesh = Mesh("resources/dodecahedron.obj")
mesh.apply_transform(TRANSFORM_MATRIX)
mesh.pos(MESH_POSITION)

mesh.color(MESH_COLOR)
mesh.lighting("off")
mesh.flat()

mesh.linewidth(MESH_LINEWIDTH)
mesh.properties.EdgeVisibilityOn()
mesh.properties.SetEdgeColor(*EDGE_COLOR)
mesh.properties.SetLineWidth(MESH_LINEWIDTH)

# Create plotter
plotter = Plotter(size=VIEWPORT_SIZE, bg=BACKGROUND_COLOR)
plotter.add(mesh)

# Configure camera
camera = plotter.camera
camera.SetPosition(CAMERA_POSITION)
camera.SetFocalPoint(CAMERA_FOCAL_POINT)
camera.SetViewUp(CAMERA_VIEW_UP)
camera.SetViewAngle(CAMERA_VIEW_ANGLE)
camera.SetClippingRange(CAMERA_CLIPPING_RANGE)

# Show the scene
plotter.show(axes=0, interactive=True, resetcam=False)
plotter.close()