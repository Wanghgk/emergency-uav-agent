import numpy as np

from backend.cerebellum.pose_person import supports_person


def test_pose_requires_separated_body_parts_inside_detection():
    points = np.zeros((17, 3))
    points[5] = (25, 24, 0.8)
    points[6] = (36, 25, 0.9)
    points[13] = (30, 66, 0.8)
    assert supports_person(points, (10, 10, 50, 80))
    points[13, 2] = 0.1
    assert not supports_person(points, (10, 10, 50, 80))
    points[13] = (30, 140, 0.9)
    assert not supports_person(points, (10, 10, 50, 80))
    points[13] = (31, 28, 0.9)
    assert not supports_person(points, (10, 10, 50, 80))
