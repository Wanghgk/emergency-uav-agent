"""Confirm aerial person proposals with a dedicated human-pose model."""

import math
from time import perf_counter

import cv2
import numpy as np

from .person_detector import Detection, PersonDetector
from .rtdetr import RTDETRPersonDetector

BODY_GROUPS = ((0, 1, 2, 3, 4), (5, 6, 11, 12), (7, 8, 9, 10, 13, 14, 15, 16))


def supports_person(keypoints, box):
    left, top, right, bottom = box
    x, y, scores = keypoints.T
    visible = (scores >= 0.45) & (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
    groups = [visible[list(group)].any() for group in BODY_GROUPS]
    if visible.sum() < 3 or not groups[1] or sum(groups) < 2:
        return False
    spread = np.ptp(keypoints[visible, :2], axis=0)
    return bool(np.linalg.norm(spread) >= 0.18 * math.hypot(right - left, bottom - top))


def overlap(first, second):
    left, top = np.maximum(first[:2], second[:2])
    right, bottom = np.minimum(first[2:], second[2:])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = (first[2] - first[0]) * (first[3] - first[1])
    other = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area + other - intersection) if intersection else 0


class PosePersonDetector(PersonDetector):
    model_name = "YOLO11s-Pose + RT-DETRv2 proposals"
    model_version = "YOLO11s Pose ONNX 307d2ad / RT-DETRv2 ONNX 1e28de7"

    def __init__(self, weights, proposal_weights):
        super().__init__(weights, threshold=0.32)
        self.proposer = RTDETRPersonDetector(proposal_weights, threshold=0.40)

    def predict(self, rgb):
        started = perf_counter()
        proposals, _ = self.proposer.predict(rgb)
        detections = []
        for candidate in sorted(
            proposals, key=lambda item: item.confidence, reverse=True
        )[:3]:
            detection = self.confirm(rgb, candidate.bbox)
            if detection and all(
                overlap(detection.bbox, item.bbox) < 0.5 for item in detections
            ):
                detections.append(detection)
        return detections, round((perf_counter() - started) * 1000, 2)

    def confirm(self, rgb, proposal):
        left, top, right, bottom = proposal
        padding = max(10, 0.65 * max(right - left, bottom - top))
        x0, y0 = max(0, int(left - padding)), max(0, int(top - padding))
        x1 = min(rgb.width, math.ceil(right + padding))
        y1 = min(rgb.height, math.ceil(bottom + padding))
        crop = np.asarray(rgb.crop((x0, y0, x1, y1)))
        scale = min(self.width / crop.shape[1], self.height / crop.shape[0])
        width, height = (
            max(1, round(crop.shape[1] * scale)),
            max(1, round(crop.shape[0] * scale)),
        )
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[:height, :width] = cv2.resize(crop, (width, height))
        tensor = (
            np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
            / 255
        )
        rows = self.session.run(None, {self.input.name: tensor})[0][0].T
        strong = rows[rows[:, 4] >= self.threshold]
        for row in sorted(strong, key=lambda item: item[4], reverse=True):
            canvas_box = np.concatenate(
                (row[:2] - row[2:4] / 2, row[:2] + row[2:4] / 2)
            )
            box = (canvas_box / scale + [x0, y0, x0, y0]).clip(
                0, [rgb.width, rgb.height, rgb.width, rgb.height]
            )
            if overlap(box, proposal) >= 0.18 and supports_person(
                row[5:].reshape(17, 3), canvas_box
            ):
                return Detection(confidence=float(row[4]), bbox=tuple(box))
        return None
