"""Bounded flight routes and a simulated line-of-sight relay policy."""

import math

import numpy as np


def fleet_roles(scout_count, relay_count):
    if not 1 <= scout_count <= 6 or not 0 <= relay_count <= 4:
        raise ValueError("configure 1–6 scouts and 0–4 relays")
    return {
        "Drone0": "scout",
        **{f"Scout{i}": "scout" for i in range(2, scout_count + 1)},
        **{
            "Relay" if i == 1 else f"Relay{i}": "relay"
            for i in range(1, relay_count + 1)
        },
    }


def scout_route(alarm, index, count, orientation, step):
    """Partition the alarm rectangle and leave separation at shared boundaries."""
    width = 2 * alarm.half_width / count
    gap = min(3, width / 4)
    lo = -alarm.half_width + index * width + (gap if index else 0)
    hi = -alarm.half_width + (index + 1) * width - (gap if index + 1 < count else 0)
    north = np.linspace(lo, hi, math.ceil((hi - lo) / step) + 1).tolist()
    east = np.linspace(
        -alarm.half_width, alarm.half_width, math.ceil(2 * alarm.half_width / step) + 1
    ).tolist()
    first, second = (east, north) if orientation == "NS" else (north, east)
    for row, outer in enumerate(first):
        for inner in second[:: 1 if row % 2 == 0 else -1]:
            n, e = (inner, outer) if orientation == "NS" else (outer, inner)
            yield alarm.north + n, alarm.east + e


def distances_and_clearance(terrain, a, b, max_range):
    distance = math.dist(a, b)
    ground_down = terrain.profile(a[:2], b[:2])
    ray_down = np.linspace(a[2], b[2], ground_down.size)
    clearance = float(np.min(ground_down - ray_down))
    return distance, clearance, min(max_range - distance, (clearance - 1) * 4)


def relay_links(terrain, base, relay, scouts, max_range):
    return {
        name: distances_and_clearance(terrain, relay, target, max_range)
        for name, target in {"base": base, **scouts}.items()
    }


def network_links(terrain, base, scouts, relays, max_range):
    links = {}
    connected = dict.fromkeys(scouts, False)
    if not relays:
        for name, target in scouts.items():
            result = distances_and_clearance(terrain, base, target, max_range)
            links["base", name] = result
            connected[name] = result[2] >= 0
    for relay, point in relays.items():
        coverage = relay_links(terrain, base, point, scouts, max_range)
        links.update({(relay, name): result for name, result in coverage.items()})
        for name in scouts:
            connected[name] |= coverage["base"][2] >= 0 and coverage[name][2] >= 0
    return links, connected


def relay_options(terrain, base, relay, scouts, lead, altitude, max_range):
    targets = list(scouts.values())
    centroid = np.mean([base[:2], *(target[:2] for target in targets)], axis=0)
    ahead = np.mean([base[:2], *(target[:2] for target in lead.values())], axis=0)
    mid = np.mean([target[:2] for target in targets], axis=0)
    options = {
        "HOLD": (*relay[:2], altitude),
        "BALANCE": (*centroid.tolist(), altitude),
        "LEAD": (*ahead.tolist(), altitude),
        "SCOUT_MID": (*mid.tolist(), altitude),
        "MIDPOINT": (*((mid + base[:2]) / 2).tolist(), altitude),
    }
    margins = {
        name: min(
            *(
                link[2]
                for link in relay_links(
                    terrain, base, point, scouts, max_range
                ).values()
            ),
            *(
                link[2]
                for link in relay_links(terrain, base, point, lead, max_range).values()
            ),
        )
        for name, point in options.items()
    }
    return options, margins


def relay_choices(margins):
    best = max(margins, key=margins.__getitem__)
    # Only offer options within the feasible set and near the best margin.
    return [
        name
        for name in margins
        if margins[name] >= 0 and margins[name] >= margins[best] - 10
    ] or [best]
