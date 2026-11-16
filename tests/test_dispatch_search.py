import math

from backend.brain.planning import MockLLMProvider, SearchDecision
from backend.simulation.rescue_search import (
    Alarm,
    match_people,
    person_results,
    review_search,
    sweep,
    track_detections,
)


def test_grid_search_starts_at_selected_corner_and_covers_alarm():
    square = Alarm(0, 0, 2)
    for cross in [True, False]:
        route = list(sweep(square, "NW", 20, cross))
        assert route[0] == (40, -40)
        assert len(set(route)) == 25
        assert all(square.contains(point) for point in route)
        assert all(
            math.dist(a, b) <= 20 for a, b in zip(route, route[1:], strict=False)
        )


def test_two_observations_not_two_boxes_confirm_a_person():
    square = Alarm(0, 0, 2)
    tracks = []
    hit = {"ned_m": (5, 4, -1), "confidence": 0.85}
    track_detections(tracks, "frame-1", [hit, hit], square)
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 1
    track_detections(tracks, "frame-2", [hit], square)
    assert len(tracks[0].observations) == 2


def test_scoring_does_not_count_duplicate_positions_as_distinct_people():
    assert len(match_people([(0, 0), (0.1, 0.1)], [(0, 0), (10, 10)])) == 1


def test_results_keep_measured_consensus_and_original_evidence():
    tracks = []
    alarm = Alarm(0, 0, 1)
    track_detections(
        tracks,
        "first",
        [{"ned_m": (0, 0, -2), "confidence": 0.5, "bbox": (1, 2, 4, 6)}],
        alarm,
    )
    track_detections(
        tracks,
        "second",
        [{"ned_m": (0, 1, -2), "confidence": 0.8, "bbox": (4, 5, 8, 9)}],
        alarm,
    )
    [result] = person_results(tracks, (100, 200, 300))
    assert result["status"] == "confirmed"
    assert result["ue_cm"] == (100, 250, 500)
    assert result["evidence"]["frame_id"] == "second"


def test_review_cannot_end_early_on_missing_people():
    class TooOptimistic(MockLLMProvider):
        def review_search(self, context):
            return SearchDecision(
                action="finish", rationale="Test premature recommendation."
            )

    early = review_search(TooOptimistic(), [], 2, can_continue=True)
    assert early["recommendation"] == "finish"
    assert early["action"] == "continue"
    exhausted = review_search(MockLLMProvider(), [], 2, can_continue=False)
    assert exhausted["source"] == "search_limit"
    assert exhausted["action"] == "finish"
    assert exhausted["confirmed_count"] == 0
