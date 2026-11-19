from math import dist

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from backend.brain.planning import (
    DeepSeekProvider,
    MockLLMProvider,
    SearchPlan,
    select_plan,
)
from backend.simulation.basemap import Basemap, terrain_relief, visible_rgb
from backend.simulation.coordinate_transform import Pose
from scripts.run_mission import sample_legs, sector_route


def test_basemap_bounds_in_air_sim_metres():
    basemap = Basemap(
        map_id="lopnur",
        task_id=290,
        camera_pose=Pose(
            position=(1144.1, 1332.5, -416), orientation_xyzw=(0, 0, 0, 1)
        ),
        width_px=1536,
        height_px=1536,
        width_m=1200,
    )
    assert basemap.bounds() == pytest.approx(
        {
            "north": 1744.1,
            "south": 544.1,
            "west": 732.5,
            "east": 1932.5,
        }
    )


def test_dark_basemap_preserves_image_detail_or_uses_depth():
    dim = Image.fromarray(np.tile(np.arange(7, 47, dtype=np.uint8), (40, 1)))
    corrected, kind = visible_rgb(dim.convert("RGB"))
    assert kind == "rgb_enhanced"
    assert corrected.size == dim.size
    assert np.median(np.asarray(corrected)) >= 100
    assert np.asarray(corrected).std() > 3
    assert visible_rgb(Image.new("RGB", dim.size)) == (None, "depth_relief")

    meta = Basemap(
        map_id="k2",
        task_id=372,
        camera_pose=Pose(position=(0, 0, -50), orientation_xyzw=(0, 0, 0, 1)),
        width_px=40,
        height_px=40,
        width_m=30,
    )
    y, x = np.indices((12, 12), dtype=float)
    relief = np.asarray(terrain_relief((x - 6) ** 2 + y, meta))
    assert relief.shape == (40, 40, 3)
    assert relief.min() >= 140 and relief.std() > 3


def test_replanning_uses_available_sector():
    provider = MockLLMProvider()
    targets = {"N": (45, 0, -24), "E": (0, 45, -24)}
    first = select_plan(provider, "Search near the start", targets, None)
    assert first.sector == "N"
    del targets[first.sector]
    replanned = select_plan(
        provider, "Search near the start", targets, {"person_detections": []}
    )
    assert replanned.sector == "E"


def test_model_cannot_supply_flight_parameters():
    with pytest.raises(ValidationError):
        SearchPlan.model_validate(
            {
                "sector": "N",
                "objective": "Find help",
                "rationale": "Survey",
                "speed": 1_000,
            }
        )


def test_deepseek_uses_dedicated_key_and_official_endpoint(monkeypatch):
    monkeypatch.setenv("DeepseekApikey", "test-only")
    monkeypatch.setenv("ESAR_LLM_API_KEY", "other-provider")
    monkeypatch.setenv("ESAR_LLM_BASE_URL", "https://example.invalid")
    monkeypatch.delenv("ESAR_DEEPSEEK_MODEL", raising=False)
    provider = DeepSeekProvider()
    assert provider.key == "test-only"
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.name == "deepseek-flash"


def test_extended_route_covers_two_lanes_with_bounded_sampling():
    start = (1144.1, 1332.5, -16)
    route = sector_route(start, "E", extended=True)
    cruise = (*start[:2], route[0][2])
    points = [cruise, *sample_legs(cruise, route, 22)]
    gaps = [dist(a, b) for a, b in zip(points, points[1:], strict=False)]
    assert len(points) > 10
    assert sum(gaps) > 235
    assert max(gaps) <= 22 + 1e-6
    assert points[-1] == route[-1]
