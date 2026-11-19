import numpy as np
import pytest

from backend.simulation.obj_terrain import ObjSurface


@pytest.mark.parametrize(
    "faces, expected",
    [
        ("f 1 2 3\nf 2 3 4\n", 0),
        ("f 1 2 4\nf 1 3 4\n", 5),
    ],
)
def test_obj_samples_original_face_diagonal(tmp_path, faces, expected):
    path = tmp_path / "surface.obj"
    path.write_text("v 0 0 0\nv 10 0 0\nv 0 10 0\nv 10 10 10\n" + faces)
    surface = ObjSurface.read(path)
    assert surface.evaluate(5, 5) == expected
    assert np.isnan(surface.evaluate(-1, 5))
