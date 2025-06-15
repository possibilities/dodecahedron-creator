# Dynamic model loading from polyhedra-viewer project

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile

# Path to polyhedra-viewer data
POLYHEDRA_DATA_PATH = (
    Path.home() / "src" / "polyhedra-viewer" / "src" / "data" / "polyhedra"
)

# Cache for loaded models
_models_cache: Dict[str, str] = {}


def format_decimal(number: float) -> str:
    """Format number as decimal, ensuring .0 for integers"""
    return f"{number:.1f}" if number == int(number) else str(number)


def vertex_to_obj(vertex: List[float]) -> str:
    """Convert vertex coordinates to OBJ vertex line"""
    coords = " ".join(format_decimal(coord) for coord in vertex)
    return f"v {coords}"


def face_to_obj(face: List[int]) -> str:
    """Convert face indices to OBJ face line (1-indexed)"""
    indices = " ".join(str(i + 1) for i in face)
    return f"f {indices}"


def polyhedron_to_obj(polyhedron_data: Dict) -> str:
    """Convert polyhedron data to OBJ format string"""
    vertices = polyhedron_data.get("vertices", [])
    faces = polyhedron_data.get("faces", [])

    # Convert vertices
    vertex_lines = [vertex_to_obj(vertex) for vertex in vertices]

    # Convert faces
    face_lines = [face_to_obj(face) for face in faces]

    # Combine all lines
    obj_lines = vertex_lines + face_lines
    return "\n".join(obj_lines)


def get_available_models() -> List[str]:
    """Get list of all available model names from polyhedra-viewer"""
    if not POLYHEDRA_DATA_PATH.exists():
        raise ValueError(f"Polyhedra data directory not found: {POLYHEDRA_DATA_PATH}")

    json_files = list(POLYHEDRA_DATA_PATH.glob("*.json"))
    model_names = [f.stem for f in json_files]

    return sorted(model_names)


def validate_model_names(model_names: List[str]) -> Tuple[List[str], List[str]]:
    """Validate model names against available models.

    Returns:
        Tuple of (valid_names, invalid_names)
    """
    available = set(get_available_models())
    valid = []
    invalid = []

    for name in model_names:
        if name in available:
            valid.append(name)
        else:
            invalid.append(name)

    return valid, invalid


def load_model_data(model_name: str) -> Dict:
    """Load polyhedron data from JSON file"""
    json_path = POLYHEDRA_DATA_PATH / f"{model_name}.json"

    if not json_path.exists():
        available = get_available_models()
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {', '.join(available[:10])}..."
        )

    with open(json_path, "r") as f:
        return json.load(f)


def get_model_obj_path(model_name: str) -> str:
    """Get OBJ file path for a model, generating it if needed.

    Returns path to a temporary OBJ file containing the model data.
    """
    # Check cache first
    if model_name in _models_cache:
        return _models_cache[model_name]

    # Load and convert model data
    model_data = load_model_data(model_name)
    obj_content = polyhedron_to_obj(model_data)

    # Create temporary file for this session
    # Using NamedTemporaryFile with delete=False to keep file available
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{model_name}.obj", delete=False
    ) as tmp_file:
        tmp_file.write(obj_content)
        obj_path = tmp_file.name

    # Cache the path
    _models_cache[model_name] = obj_path

    return obj_path


def cleanup_temp_models():
    """Clean up temporary OBJ files"""
    for obj_path in _models_cache.values():
        try:
            if os.path.exists(obj_path):
                os.unlink(obj_path)
        except Exception:
            pass
    _models_cache.clear()
