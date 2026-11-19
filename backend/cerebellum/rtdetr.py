"""Run the pinned RT-DETRv2 R18 ONNX export on ESAR RGB frames."""

from time import perf_counter

import numpy as np
from PIL import Image

from .person_detector import Detection, PersonDetector


class RTDETRPersonDetector(PersonDetector):
    model_name = "RT-DETRv2 R18"
    model_version = "PekingU COCO / ONNX 1e28de7"

    def predict(self, rgb):
        started = perf_counter()
        size = (
            self.width if isinstance(self.width, int) else 640,
            self.height if isinstance(self.height, int) else 640,
        )
        image = rgb.resize(size, Image.Resampling.BILINEAR)
        tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        logits, boxes = self.session.run(
            ["logits", "pred_boxes"], {self.input.name: np.ascontiguousarray(tensor)}
        )
        person = 1 / (1 + np.exp(-np.clip(logits[0, :, 0], -30, 30)))
        accepted = (person >= self.threshold) & (logits[0].argmax(axis=-1) == 0)
        centers = boxes[0, accepted, :2]
        half_size = boxes[0, accepted, 2:] / 2
        corners = np.concatenate((centers - half_size, centers + half_size), axis=-1)
        corners = corners.clip(0, 1) * [rgb.width, rgb.height, rgb.width, rgb.height]
        detections = [
            Detection(confidence=float(score), bbox=tuple(box))
            for score, box in zip(person[accepted], corners, strict=True)
            if box[2] > box[0] and box[3] > box[1]
        ]
        return detections, round((perf_counter() - started) * 1000, 2)
