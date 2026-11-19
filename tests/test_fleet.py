from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from backend.simulation.fleet import (
    distances_and_clearance,
    fleet_roles,
    network_links,
    relay_choices,
    scout_route,
)
from backend.simulation.rescue_search import Alarm
from scripts import run_fleet
from scripts.compare_terrain import interpolate, slopes
from scripts.run_fleet import fly, search_route


def test_high_relief_climb_uses_bounded_air_sim_commands(monkeypatch):
    client = Mock()
    world_down = 480.4
    offset = (-700, -800, 640)
    steps = []

    def read_pose(**kwargs):
        xyz = (240, -110, world_down - offset[2])
        return SimpleNamespace(
            position=SimpleNamespace(
                **dict(zip(("x_val", "y_val", "z_val"), xyz, strict=True))
            )
        )

    def move(down, speed, timeout_sec, **kwargs):
        nonlocal world_down
        next_down = down + offset[2]
        assert abs(world_down - next_down) / speed < timeout_sec < 85
        world_down = next_down
        steps.append(next_down)
        return Mock()

    client.simGetVehiclePose.side_effect = read_pose
    client.simGetCollisionInfo.return_value.has_collided = False
    client.moveToZAsync.side_effect = move
    monkeypatch.setattr("scripts.run_fleet.time.sleep", lambda _: None)
    fly(Mock(), "Drone0", offset, (-460, -910, 106.15), 0, client=client)

    assert len(steps) == 4
    assert steps[-1] == pytest.approx(106.15)
    client.moveToPositionAsync.assert_not_called()


def test_second_search_round_climbs_only_for_its_local_terrain():
    first, second = (-347, -910), (-335, -910)
    current = (*first, 392.9)
    flown = []

    def move(point, **kwargs):
        nonlocal current
        flown.append(point)
        current = point

    terrain = SimpleNamespace(highest=lambda a, b: 418.4 if a == b else 408.9)
    assert search_route(
        terrain, [first, second], lambda: current, move, lambda: False, 19, None
    )
    assert min(point[2] for point in flown) == pytest.approx(389.9)
    assert flown[-1][:2] == second


def test_second_scout_starts_next_round_and_finishes_if_first_scout_fails(monkeypatch):
    first_failed = Event()
    second_started = Event()
    events = []

    def event(kind, **details):
        events.append((kind, details))
        if kind == "fleet_aircraft_error" and details["vehicle_name"] == "Drone0":
            first_failed.set()

    def route(terrain, waypoints, locate, move, stopped, clearance, departure):
        if clearance == 14:
            assert second_started.wait(10)
            raise RuntimeError("Drone0 waypoint failure")
        if departure is None:
            second_started.set()
            assert first_failed.wait(10)
            assert not stopped()
        return True

    def select(provider, instruction, choices, feedback):
        return {"sector": choices[0], "source": "test"}

    monkeypatch.setattr(run_fleet, "search_route", route)
    monkeypatch.setattr(run_fleet, "select", select)
    monkeypatch.setattr(run_fleet, "position", lambda client, name, offset: offset)
    monkeypatch.setattr(run_fleet, "observe", lambda *args: {"vehicle_name": args[-2]})
    args = SimpleNamespace(
        fleet=fleet_roles(2, 0),
        rounds=2,
        spacing=12,
        frame_interval=0.25,
        points_per_scout=2,
        radio_range=320,
    )
    ready = {"Drone0": (0, 0, -20), "Scout2": (0, 8, -33)}
    _, failures = run_fleet.coordinate(
        args,
        Mock(),
        Mock(),
        SimpleNamespace(prompt="Search"),
        (0, 0, 0),
        Alarm(10, 10, 2),
        Mock(),
        Mock(),
        "test",
        event,
        SimpleNamespace(profile=lambda a, b: np.zeros(2)),
        (0, 0, -8),
        {"Drone0": -20, "Scout2": -33},
        ready,
        dict.fromkeys(ready, 0),
        ready,
    )
    assert failures == {
        "Drone0": {"stage": "search", "error": "Drone0 waypoint failure"}
    }
    assert [
        details["round"]
        for kind, details in events
        if kind == "fleet_round_complete" and details["vehicle_name"] == "Scout2"
    ] == [1, 2]


class Ridge:
    def profile(self, a, b):
        return np.array([0, -20, 0])


def test_relay_margin_detects_terrain_occlusion_inside_radio_range():
    low = distances_and_clearance(Ridge(), (0, 0, -15), (20, 0, -15), 40)
    high = distances_and_clearance(Ridge(), (0, 0, -27), (20, 0, -27), 40)
    assert low[0] == high[0] == 20
    assert low[2] < 0 < high[2]
    assert relay_choices({"HOLD": low[2], "BALANCE": high[2], "LEAD": 1}) == ["BALANCE"]
    assert relay_choices({"HOLD": -10, "LEAD": -2}) == ["LEAD"]


def test_scout_paths_are_disjoint_and_cover_their_assigned_rectangle():
    alarm = Alarm(north=100, east=200, expected_people=1)
    for index in range(2):
        ew = list(scout_route(alarm, index, 2, "EW", 12))
        ns = list(scout_route(alarm, index, 2, "NS", 12))
        assert ew != ns
        assert set(ew) == set(ns)
        assert all(alarm.contains(point) for point in ew)
        assert all((north <= 97 if index == 0 else north >= 103) for north, _ in ew)


def test_three_scouts_two_relays_and_direct_link():
    class Flat:
        def profile(self, a, b):
            return np.zeros(2)

    assert list(fleet_roles(3, 2)) == ["Drone0", "Scout2", "Scout3", "Relay", "Relay2"]
    alarm = Alarm(north=100, east=200, expected_people=1)
    stripes = [set(scout_route(alarm, i, 3, "EW", 12)) for i in range(3)]
    assert all(stripes) and len(set.union(*stripes)) == sum(map(len, stripes))
    scouts = {"Drone0": (65, 0, -8), "Scout2": (0, 20, -8)}
    base = (0, 0, -8)
    relays = {"Relay": (0, 30, -8), "Relay2": (33, 0, -8)}
    links, connected = network_links(Flat(), base, scouts, relays, 40)
    assert links["Relay", "Drone0"][2] < 0 <= links["Relay2", "Drone0"][2]
    assert connected == {"Drone0": True, "Scout2": True}
    links, connected = network_links(Flat(), base, scouts, {}, 40)
    assert set(links) == {("base", name) for name in scouts}
    assert connected == {"Drone0": False, "Scout2": True}


def test_terrain_grid_reconstructs_plane_and_slope():
    yy, xx = np.indices((5, 5))
    plane = 10 + 0.5 * xx + yy
    y, x = np.indices((9, 9)) / 2
    expected = 10 + 0.5 * x + y
    bilinear, tin, cubic = interpolate(plane, y, x)
    np.testing.assert_allclose(bilinear, expected)
    np.testing.assert_allclose(tin, expected)
    np.testing.assert_allclose(cubic[2:-2, 2:-2], expected[2:-2, 2:-2])
    np.testing.assert_allclose(
        slopes(bilinear, 1, 1), np.degrees(np.arctan(np.hypot(0.25, 0.5)))
    )
