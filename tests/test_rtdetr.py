from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.cerebellum.rtdetr import RTDETRPersonDetector


def test_rtdetr_decodes_only_person_boxes_in_rgb_coordinates():
    class Session:
        def run(self, names, inputs):
            assert names == ["logits", "pred_boxes"]
            tensor = inputs["pixel_values"]
            assert tensor.shape == (1, 3, 640, 640)
            np.testing.assert_allclose(tensor[0, :, 0, 0], [1, 64 / 255, 0])
            logits = np.array([[[4, 1], [2, 3], [-4, -9]]])
            boxes = np.tile([0.5, 0.5, 0.5, 0.5], (1, 3, 1))
            return logits, boxes

    detector = RTDETRPersonDetector.__new__(RTDETRPersonDetector)
    detector.session = Session()
    detector.input = SimpleNamespace(name="pixel_values")
    detector.width = detector.height = 640
    detector.threshold = 0.4
    boxes, _ = detector.predict(Image.new("RGB", (200, 100), (255, 64, 0)))
    [person] = boxes
    assert person.bbox == (50, 25, 150, 75)
