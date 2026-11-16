"""YOLOX-Tiny COCO person detection on RGB observations."""

from time import perf_counter

import cv2
import numpy as np
from pydantic import BaseModel, Field

from backend.simulation.coordinate_transform import Point
from backend.simulation.observation_collector import map_point


class Detection(BaseModel):
    category: str = "person"
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]
    ned_m: Point | None = None
    ue_cm: Point | None = None
    depth_spread_m: float | None = None


class PersonDetector:
    model_name = "YOLOX-Tiny"
    model_version = "0.1.1rc0 / COCO ONNX"

    def __init__(self, weights, threshold=0.30):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(weights), options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.height, self.width = self.input.shape[-2:]
        self.threshold = threshold

    def detect(self, observation, rgb, depth):
        detections, latency = self.predict(rgb)
        return [
            estimate(detection, observation, depth) for detection in detections
        ], latency

    def predict(self, rgb):
        start = perf_counter()
        scale = min(self.width / rgb.width, self.height / rgb.height)
        data = np.asarray(rgb)[:, :, ::-1]
        resized = cv2.resize(data, (int(rgb.width * scale), int(rgb.height * scale)))
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        batch = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        rows = self.session.run(None, {self.input.name: batch})[0][0]
        # The official export has raw offsets on the stride 8, 16 and 32 grids.
        offset = 0
        for stride in (8, 16, 32):
            y, x = np.indices((self.height // stride, self.width // stride))
            layer = rows[offset : offset + x.size]
            layer[:, :2] = (
                layer[:, :2] + np.column_stack((x.ravel(), y.ravel()))
            ) * stride
            layer[:, 2:4] = np.exp(layer[:, 2:4]) * stride
            offset += x.size
        scores = rows[:, 4] * rows[:, 5]
        accepted = (scores >= self.threshold) & (rows[:, 5:].argmax(axis=1) == 0)
        xy, size = rows[accepted, :2], rows[accepted, 2:4]
        first = np.clip((xy - size / 2) / scale, 0, [rgb.width, rgb.height])
        last = np.clip((xy + size / 2) / scale, 0, [rgb.width, rgb.height])
        boxes = np.column_stack((first, last - first))
        scores = scores[accepted]
        detections = []
        for index in cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.threshold, 0.45
        ):
            bbox = tuple(float(value) for value in (*first[index], *last[index]))
            detections.append(Detection(confidence=float(scores[index]), bbox=bbox))
        return detections, round((perf_counter() - start) * 1000, 2)


def estimate(detection, observation, depth):
    left, top, right, bottom = detection.bbox
    rgb = observation.camera_intrinsics
    camera = observation.depth_camera_intrinsics
    # Keep samples near the centre of the detection and use the median depth.
    u0 = max(0, int((0.7 * left + 0.3 * right) / rgb.width * camera.width))
    u1 = min(
        camera.width, int((0.3 * left + 0.7 * right) / rgb.width * camera.width) + 1
    )
    v0 = max(0, int((0.7 * top + 0.3 * bottom) / rgb.height * camera.height))
    v1 = min(
        camera.height, int((0.3 * top + 0.7 * bottom) / rgb.height * camera.height) + 1
    )
    window = depth[v0:v1, u0:u1]
    valid = np.argwhere((window >= 0.5) & (window <= 200))
    if len(valid) < 3:
        return detection
    samples = window[tuple(valid.T)]
    median = np.median(samples)
    spread = float(np.percentile(samples, 75) - np.percentile(samples, 25))
    if spread > 2:
        return detection
    v, u = valid[np.argmin(np.abs(samples - median))]
    try:
        location = map_point(observation, depth, u0 + int(u), v0 + int(v))
    except ValueError:
        return detection
    return detection.model_copy(
        update={
            "ned_m": location["ned_m"],
            "ue_cm": location["ue_cm"],
            "depth_spread_m": spread,
        }
    )
