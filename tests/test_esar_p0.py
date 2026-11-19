from datetime import UTC, datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.simulation.coordinate_transform import (
    Intrinsics,
    Pose,
    ned_to_ue,
    pixel_to_ned,
)
from backend.simulation.esar_adapter import launch_args, load_task
from backend.simulation.observation_collector import (
    Observation,
    compare_static_depth,
    load_observation,
    map_point,
    save_observation,
)


@pytest.fixture
def record():
    pose = Pose(position=(10, 20, -30), orientation_xyzw=(0, 0, 0, 1))
    return Observation(
        map_id="aotai2800",
        task_id=1,
        frame_id="test",
        timestamp=datetime.now(UTC),
        rgb_timestamp_ns=100,
        depth_timestamp_ns=100,
        drone_state_timestamp_ns=100,
        drone_position=pose.position,
        drone_orientation=pose.orientation_xyzw,
        camera_name="front_camera",
        camera_pose=pose,
        depth_camera_pose=pose.model_copy(),
        camera_intrinsics=Intrinsics.from_horizontal_fov(512, 512, 90),
        depth_camera_intrinsics=Intrinsics.from_horizontal_fov(128, 128, 90),
        map_origin_ue_cm=(0, 0, -16000),
    )


def test_tracking_off_axis_pixel_between_frames(record):
    depth = np.full((128, 128), 25, dtype=np.float32)
    result = map_point(record, depth, 91, 40)
    assert result["rgb_pixel"] == pytest.approx((365.5, 161.5))
    assert result["ned_m"] == pytest.approx((35, 30.7421875, -39.1796875))
    assert ned_to_ue((120, 140, -60), record.map_origin_ue_cm) == (12000, 14000, -10000)


def test_camera_yaw_rotates_forward_to_east(record):
    pose = Pose(position=(10, 20, -30), orientation_xyzw=(0, 0, sqrt(0.5), sqrt(0.5)))
    assert pixel_to_ned(
        63.5, 63.5, 10, record.depth_camera_intrinsics, pose
    ) == pytest.approx((10, 30, -30))


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf, 65504])
def test_failed_depth_is_never_given_a_location(record, invalid):
    depth = np.full((128, 128), invalid, dtype=np.float32)
    with pytest.raises(ValueError, match="positive planar depth"):
        map_point(record, depth, 0, 0)


def test_replay_with_unequal_rgb_depth_sizes(record, tmp_path):
    depth = np.full((128, 128), 37, dtype=np.float32)
    rgb = Image.new("RGB", (512, 512), color=(16, 32, 64))
    manifest = save_observation(tmp_path, record, rgb, depth)
    restored, image, values = load_observation(manifest)
    assert image.getpixel((100, 100)) == (16, 32, 64)
    assert np.array_equal(depth, values)
    assert map_point(restored, values, 60, 70) == map_point(record, depth, 60, 70)
    with pytest.raises(ValueError, match="outside"):
        map_point(restored, values, -1, 0)


def test_disagreeing_sensor_poses_are_rejected(record):
    record.depth_camera_pose.position = (11, 20, -30)
    with pytest.raises(ValueError, match="poses differ"):
        map_point(record, np.ones((128, 128)), 10, 10)


def test_cross_view_checks_static_depth_against_actual_measurements(record):
    moved = record.model_copy(deep=True)
    moved.depth_camera_pose.position = (10, 20, -33)
    depth = np.full((128, 128), 25, dtype=np.float32)
    assert (
        compare_static_depth(record, depth, moved, depth)["median_depth_residual_m"]
        == 0
    )
    with pytest.raises(ValueError, match="median depth residual"):
        compare_static_depth(record, depth, moved, depth + 3)


def test_official_lopnur_task_and_launch_config(tmp_path):
    upstream = Path(".external/ESAR-pinned")
    if not upstream.is_dir():
        pytest.skip(
            "download the pinned official ESAR source to exercise this integration test"
        )
    task, start, origin = load_task(upstream, "task_data.json", 290)
    assert start == (1144.1, 1332.5, -16)
    assert origin == (0, 0, 0)
    assert not {"targets", "cues", "fire"} & task.model_dump().keys()
    executable = tmp_path / "lopnur" / "Windows" / "lopnur.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    command = launch_args(tmp_path, upstream / "settings" / "settings.json", task)
    assert command[0] == str(executable)
    assert command[1:4] == ["-TaskNum=14", "-Weather=4", "-Time=1600"]
    assert task.original_time_of_day is None


def test_official_night_task_launches_at_noon(tmp_path):
    upstream = Path(".external/ESAR-pinned")
    if not upstream.is_dir():
        pytest.skip("download the pinned official ESAR source to test K2 task 372")
    task, _, _ = load_task(upstream, "task_data.json", 372)
    assert task.original_time_of_day == 180
    assert task.time_of_day == 1200
    executable = tmp_path / "k2" / "k2.exe"
    executable.parent.mkdir()
    executable.touch()
    command = launch_args(tmp_path, upstream / "settings" / "settings.json", task)
    assert command[1:4] == [
        f"-TaskNum={task.snapshot_id % 15}",
        "-Weather=1",
        "-Time=1200",
    ]
