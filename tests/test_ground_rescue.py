import numpy as np
import pytest

from backend.brain.ground_rescue import configured_entry, ground_rescue
from backend.brain.planning import GroundDecision, MockLLMProvider
from backend.simulation.ground_terrain import SurfaceGrid, check_route

ENTRY = {"north": 54.0, "east": 6.0, "label": "edge"}
PERSON = {
    "id": 1,
    "status": "confirmed",
    "ned_m": [6.0, 6.0, -3.5],
    "horizontal_observation_spread_m": 1.2,
    "hidden_value": [100, 200],
}


def test_coarse_then_native_terrain_and_final_route():
    class InspectingProvider(MockLLMProvider):
        def inspect_ground(self, context):
            assert context["terrain_summary"]["row_step_m_south"] > 2
            overview = context["terrain_overview"]
            assert overview["cell_diagonal"] == "northwest_to_southeast"
            assert overview["vertex_north_m"][0] == 59
            assert overview["vertex_north_m"][-1] == 1
            assert overview["vertex_east_m"][-1] == 59
            assert context["horizontal_observation_spread_m"] == 1.2
            assert "terrain_details" not in context and "hidden_value" not in context
            return super().inspect_ground(context)

        def choose_ground(self, context):
            slices = context["terrain_details"]
            assert 1 <= len(slices) <= 2
            for window in slices.values():
                assert window["row_step_m_south"] == 2
                assert (
                    window["surface_function"]
                    == "piecewise affine interpolation on triangles"
                )
                assert len(window["surface_height_m"]) <= 9
            return super().choose_ground(context)

    grid = SurfaceGrid(np.zeros((30, 30)), 60, 0, 2, 2)
    result = ground_rescue(grid, "sample", ENTRY, [PERSON], InspectingProvider())
    [route] = result["routes"]
    assert route["status"] == "surface_checked"
    assert route["metrics"]["approach_distance_m"] <= 12
    assert route["waypoints_ned_m"][0][0] == 53
    assert route["inspected_windows"]


def test_missing_surface_prevents_path_and_candidate_alone_does_not_trigger():
    surface = np.zeros((30, 30))
    surface[15, :] = np.nan
    grid = SurfaceGrid(surface, 60, 0, 2, 2)
    result = ground_rescue(grid, "sample", ENTRY, [PERSON], object())
    assert result["routes"][0]["status"] == "no_route"
    result = ground_rescue(
        grid, "sample", ENTRY, [PERSON | {"status": "candidate"}], object()
    )
    assert result["status"] == "waiting_for_confirmation"
    assert not result["routes"]


def test_model_can_abstain_or_return_invalid_choice():
    class Provider(MockLLMProvider):
        choice = None

        def choose_ground(self, context):
            return GroundDecision(route_id=self.choice, rationale="Requires review")

    provider = Provider()
    grid = SurfaceGrid(np.zeros((30, 30)), 60, 0, 2, 2)
    assert (
        ground_rescue(grid, "sample", ENTRY, [PERSON], provider)["routes"][0]["status"]
        == "manual_review"
    )
    provider.choice = "invented"
    route = ground_rescue(grid, "sample", ENTRY, [PERSON], provider)["routes"][0]
    assert route["model"] == "local_fallback"
    assert route["route_id"] != "invented"


def test_validator_rejects_a_straight_line_skipping_original_cells():
    grid = SurfaceGrid(np.zeros((30, 30)), 60, 0, 2, 2)
    start = grid.cell((54, 6))
    end = grid.cell((6, 6))
    with pytest.raises(ValueError, match="skips"):
        check_route(grid, [start, end], start, (6, 6), 30, 12)


def test_flat_raised_rock_and_its_buffer_are_excluded():
    surface = np.zeros((50, 60))
    surface[20:29, 25:36] = -3
    grid = SurfaceGrid(surface, 50, 0, 1, 1)
    assert grid.slope[24, 30] == 0
    assert grid.blocked[24, 30] and grid.blocked[24, 24]
    with pytest.raises(ValueError, match="blocked"):
        check_route(grid, [(24, 30)], (24, 30), grid.point((24, 30))[:2], 30, 12)
    assert grid.packet((18, 32, 20, 39), stride=4)["contains_blocked_cells"][1][1]


def test_each_current_map_has_a_configured_entry():
    for name in ("lopnur", "aotai2800", "k2"):
        entry = configured_entry(name)
        assert set(entry) == {"north", "east", "reference_down_m", "label"}


def test_entry_rejects_surface_obstruction_and_raster_checks_task(tmp_path):
    path = tmp_path / "mission.npz"
    np.savez_compressed(
        path,
        down=np.zeros((10, 10)),
        north=30,
        south=0,
        west=0,
        east=30,
        task_id=290,
        map_id="lopnur",
    )
    with pytest.raises(ValueError, match="different ESAR task"):
        SurfaceGrid.from_file(path, task={"task_id": 28, "map_name": "aotai2800"})
    grid = SurfaceGrid.from_file(path, task={"task_id": 290, "map_name": "lopnur"})
    with pytest.raises(ValueError, match="surveyed elevation"):
        grid.entry_point(15, 15, expected_down=40)
