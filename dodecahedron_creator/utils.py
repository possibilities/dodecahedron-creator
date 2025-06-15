# Shared utilities, mesh setup, and media processing

import numpy as np
from vedo import Mesh, Plotter

from .config import (
    load_configuration,
)
from .models import get_model_obj_path


# Import needed for circular dependency avoidance
def setup_mesh(config, model_name=None):
    if model_name:
        model_path = get_model_obj_path(model_name)
    else:
        # Fallback to dodecahedron if no model specified
        model_path = get_model_obj_path("dodecahedron")

    mesh = Mesh(model_path)

    if "transform_matrix" in config["mesh"]:
        transform_matrix = np.array(config["mesh"]["transform_matrix"])
        mesh.apply_transform(transform_matrix)

    if "position" in config["mesh"]:
        mesh.pos(config["mesh"]["position"])

    mesh.color(config["mesh"]["color"])
    mesh.lighting("off")
    mesh.flat()

    mesh.linewidth(config["mesh"]["linewidth"])
    mesh.properties.EdgeVisibilityOn()
    mesh.properties.SetEdgeColor(*config["mesh"]["edge_color"])
    mesh.properties.SetLineWidth(config["mesh"]["linewidth"])

    return mesh


def setup_camera(plotter, config):
    camera = plotter.camera
    if "position" in config["camera"]:
        camera.SetPosition(config["camera"]["position"])
    if "focal_point" in config["camera"]:
        camera.SetFocalPoint(config["camera"]["focal_point"])
    if "view_up" in config["camera"]:
        camera.SetViewUp(config["camera"]["view_up"])
    if "view_angle" in config["camera"]:
        camera.SetViewAngle(config["camera"]["view_angle"])
    if "clipping_range" in config["camera"]:
        camera.SetClippingRange(config["camera"]["clipping_range"])
    return camera


def setup_scene_components(
    use_fresh=False, scene_path=None, style=None, model_name=None
):
    # Import here to avoid circular dependency
    from .viewer import create_viewer

    config = load_configuration(
        ignore_saved=use_fresh,
        scene_path=scene_path,
        style=style,
        model_name=model_name,
    )
    mesh = setup_mesh(config, model_name=model_name)
    plotter = create_viewer(config)
    plotter.add(mesh)
    return config, mesh, plotter


def setup_offscreen_renderer(config, model_name=None):
    mesh = setup_mesh(config, model_name=model_name)

    plotter = Plotter(
        bg=config["viewport"]["background_color"],
        offscreen=True,
        size=tuple(config["viewport"]["size"]),
    )
    plotter.add(mesh)

    camera = setup_camera(plotter, config)
    plotter.show(axes=0, interactive=False, resetcam=False)

    return plotter, mesh, camera
