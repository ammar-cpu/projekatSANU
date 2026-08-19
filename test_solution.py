import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from qvnd.data import parse_vrp, compute_distance_matrix
from qvnd.local_search import (
    CROSS_SEGMENT_LENGTHS,
    SEGMENT_LENGTHS,
    cross_exchange,
    cross_exchange_delta,
    or_opt,
    or_opt_delta,
    relocate,
    relocate_delta,
    swap,
    swap_delta,
    two_opt,
    two_opt_delta,
)
from qvnd.experiments import (
    ARMS,
    format_comparisons,
    format_summary,
    make_selector,
    paired_comparison,
    run_batch,
    run_single,
    save_results_csv,
    summarize,
)
from qvnd.gvns import SHAKE_E, gvns, shake
from qvnd.qlearning import N_STATES, QAgent
from qvnd.vnd import (
    DEFAULT_NEIGHBORHOODS,
    REFERENCE_CALL_MS,
    REWARDS,
    improvement_per_second,
    improvement_reward,
    EARLY,
    LATE,
    MID,
    FixedSelector,
    RandomSelector,
    elapsed_progress,
    phase_of,
    vnd,
)
from qvnd.solution import (
    Solution,
    cost,
    is_feasible,
    initial_solution_sweep,
    initial_solution_nearest_neighbor,
)

INSTANCE = Path(__file__).parent / "qvnd" / "instances" / "A-n32-k5.vrp"


@contextmanager
def time_limit(seconds):
    # guards against a construction heuristic that fails to terminate
    def _timed_out(signum, frame):
        raise TimeoutError(f"call did not return within {seconds}s")

    previous = signal.signal(signal.SIGALRM, _timed_out)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

# hand-built fixture: depot at origin, customers on a 3-4-5 rectangle so every
# pairwise distance used below is an exact integer
FIXTURE_COORDS = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 3.0], [4.0, 0.0]])
FIXTURE_DIST = compute_distance_matrix(FIXTURE_COORDS)


def test_parse_reads_name_and_optimum():
    inst = parse_vrp(INSTANCE)
    assert inst["name"] == "A-n32-k5"
    assert inst["optimum"] == 784  # taken from the COMMENT line, not a separate table


def test_every_instance_parses_and_is_solvable():
    # the whole Augerat A set, not just the three the earlier tests use
    files = sorted((INSTANCE.parent).glob("*.vrp"))
    assert len(files) == 27

    for path in files:
        inst = parse_vrp(path)
        assert inst["optimum"] > 0
        assert len(inst["coords"]) == inst["dimension"] == len(inst["demands"])
        assert inst["demands"][inst["depot_id"]] == 0

        routes = initial_solution_nearest_neighbor(
            inst["coords"], inst["demands"], inst["capacity"], inst["depot_id"], inst["dist"]
        )
        assert is_feasible(routes, inst["demands"], inst["capacity"])
        # a construction has to be worse than the optimum, never better
        assert cost(routes, inst["dist"]) >= inst["optimum"]


def test_parse_node_count_and_capacity():
    inst = parse_vrp(INSTANCE)
    assert inst["dimension"] == 32
    assert inst["capacity"] == 100
    assert len(inst["coords"]) == 32
    assert len(inst["demands"]) == 32


def test_parse_depot_is_index_zero():
    inst = parse_vrp(INSTANCE)
    assert inst["depot_id"] == 0
    # node 1 in the file is the depot: coords (82, 76), demand 0
    assert tuple(inst["coords"][0]) == (82.0, 76.0)
    assert inst["demands"][0] == 0
    # customers keep file order after the depot: node 2 is (96, 44), demand 19
    assert tuple(inst["coords"][1]) == (96.0, 44.0)
    assert inst["demands"][1] == 19


def test_distance_matrix_is_symmetric():
    inst = parse_vrp(INSTANCE)
    dm = compute_distance_matrix(inst["coords"])
    assert dm.shape == (32, 32)
    assert np.allclose(dm, dm.T)
    assert np.allclose(np.diag(dm), 0.0)


def test_distance_matrix_known_values():
    # 3-4-5 triangle: depot (0,0) to customer 2 (4,3) is exactly 5
    assert FIXTURE_DIST[0, 2] == 5.0
    assert FIXTURE_DIST[0, 1] == 3.0
    assert FIXTURE_DIST[1, 2] == 4.0


def test_distance_matrix_uses_rounded_euc_2d():
    # pins the TSPLIB EUC_2D convention the published optima assume: every
    # distance is a whole number, not a raw floating-point Euclidean one
    inst = parse_vrp(INSTANCE)
    dm = compute_distance_matrix(inst["coords"])
    assert np.array_equal(dm, np.round(dm))

    # depot (82,76) -> node 2 (96,44): sqrt(14^2 + 32^2) = sqrt(1220) = 34.9285... -> 35
    raw = np.sqrt(1220.0)
    assert 34.9 < raw < 35.0  # the unrounded value is strictly below 35
    assert dm[0, 1] == 35.0


def test_cost_single_route_hand_computed():
    # depot -> (0,3) -> (4,3) -> (4,0) -> depot  =  3 + 4 + 3 + 4  =  14
    assert cost([[1, 2, 3]], FIXTURE_DIST) == 14.0


def test_cost_multiple_routes_hand_computed():
    # route A: depot -> (0,3) -> (4,3) -> depot  =  3 + 4 + 5  =  12
    # route B: depot -> (4,0) -> depot           =  4 + 4      =   8
    assert cost([[1, 2], [3]], FIXTURE_DIST) == 20.0


def test_cost_single_customer_route_is_there_and_back():
    assert cost([[2]], FIXTURE_DIST) == 10.0


def test_cost_ignores_empty_routes():
    assert cost([[1, 2], [], [3]], FIXTURE_DIST) == cost([[1, 2], [3]], FIXTURE_DIST)


def test_feasible_solution_accepted():
    demands = np.array([0, 10, 10, 10])
    assert is_feasible([[1, 2], [3]], demands, capacity=25)


def test_capacity_overflow_rejected():
    demands = np.array([0, 10, 10, 10])
    # all three customers on one route: load 30 > capacity 25
    assert not is_feasible([[1, 2, 3]], demands, capacity=25)


def test_capacity_boundary_is_inclusive():
    demands = np.array([0, 10, 10, 10])
    # load exactly equal to capacity must stay feasible
    assert is_feasible([[1, 2, 3]], demands, capacity=30)


def test_duplicate_customer_rejected():
    demands = np.array([0, 10, 10, 10])
    # capacity is generous on purpose, so only the duplication can cause failure
    assert not is_feasible([[1, 2], [2, 3]], demands, capacity=100)


def test_duplicate_within_same_route_rejected():
    demands = np.array([0, 10, 10, 10])
    assert not is_feasible([[1, 2, 2], [3]], demands, capacity=100)


def test_missing_customer_rejected():
    demands = np.array([0, 10, 10, 10])
    assert not is_feasible([[1, 2]], demands, capacity=100)


def test_nn_terminates_on_infeasible_demand():
    # capacity 10 against demands 9, 9, 9, 2: no two customers fit together, so
    # the heuristic is forced to open several routes
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    demands = np.array([0, 9, 9, 9, 2])
    capacity = 10
    dist = compute_distance_matrix(coords)

    with time_limit(3):
        routes = initial_solution_nearest_neighbor(coords, demands, capacity, 0, dist)

    placed = [c for route in routes for c in route]
    assert sorted(placed) == [1, 2, 3, 4]        # every customer assigned
    assert len(placed) == len(set(placed))       # and none of them twice
    for route in routes:
        assert sum(demands[c] for c in route) <= capacity
    assert is_feasible(routes, demands, capacity)


def test_nn_rejects_demand_exceeding_capacity():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    demands = np.array([0, 11, 5])
    dist = compute_distance_matrix(coords)

    try:
        initial_solution_nearest_neighbor(coords, demands, 10, 0, dist)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError: demand 11 exceeds capacity 10")


def test_sweep_rejects_demand_exceeding_capacity():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    demands = np.array([0, 11, 5])

    try:
        initial_solution_sweep(coords, demands, 10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError: demand 11 exceeds capacity 10")


def test_constructors_produce_no_empty_routes():
    # local search operators are not expected to cope with empty routes
    inst = parse_vrp(INSTANCE)
    dist = compute_distance_matrix(inst["coords"])
    args = (inst["coords"], inst["demands"], inst["capacity"], inst["depot_id"])

    for routes in (initial_solution_sweep(*args),
                   initial_solution_nearest_neighbor(*args, dist)):
        assert all(len(route) > 0 for route in routes)
        assert is_feasible(routes, inst["demands"], inst["capacity"])


def _nn_solution(inst):
    routes = initial_solution_nearest_neighbor(
        inst["coords"], inst["demands"], inst["capacity"], inst["depot_id"], inst["dist"]
    )
    return Solution(routes, inst)


def test_delta_relocate():
    # the incremental delta has to agree with a from-scratch cost on arbitrary
    # moves, not only on the improving ones the search would actually take
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    rng = random.Random(20260802)

    applied, attempts = 0, 0
    while applied < 1000:
        attempts += 1
        assert attempts < 100_000, "could not sample enough valid moves"

        occupied = [r for r, route in enumerate(sol.routes) if route]
        from_route = rng.choice(occupied)
        from_pos = rng.randrange(len(sol.routes[from_route]))
        to_route = rng.randrange(len(sol.routes))

        same_route = to_route == from_route
        slots = len(sol.routes[to_route]) - 1 if same_route else len(sol.routes[to_route])
        to_pos = rng.randint(0, slots)
        if same_route and to_pos == from_pos:
            continue

        customer = sol.routes[from_route][from_pos]
        if not same_route and sol.load[to_route] + inst["demands"][customer] > inst["capacity"]:
            continue

        delta = relocate_delta(inst, sol, from_route, from_pos, to_route, to_pos)
        before = sol.cost
        sol.apply_relocate(inst, from_route, from_pos, to_route, to_pos, delta)

        true_cost, true_load = sol.recompute(inst)
        assert abs(before + delta - true_cost) < 1e-6
        assert abs(sol.cost - true_cost) < 1e-6
        assert sol.load == true_load

        applied += 1

    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def _toy_inst(coords, demands, capacity):
    coords = np.array(coords, dtype=float)
    return {
        "coords": coords,
        "demands": np.array(demands),
        "dist": compute_distance_matrix(coords),
        "capacity": capacity,
        "depot_id": 0,
        "dimension": len(coords),
    }


def test_relocate_handles_emptied_and_empty_routes():
    # random moves on a real instance essentially never empty a route, so this
    # case needs a deterministic check of its own
    inst = _toy_inst([[0, 0], [0, 4], [3, 0], [6, 0]], [0, 3, 3, 3], capacity=10)
    sol = Solution([[1], [2, 3]], inst)

    # draining route 0 leaves it empty rather than dropping it
    delta = relocate_delta(inst, sol, 0, 0, 1, 0)
    before = sol.cost
    sol.apply_relocate(inst, 0, 0, 1, 0, delta)
    assert sol.routes[0] == []
    assert len(sol.routes) == 2
    true_cost, true_load = sol.recompute(inst)
    assert abs(before + delta - true_cost) < 1e-6
    assert sol.load[0] == 0 and sol.load == true_load

    # and the empty route is still a valid insertion target afterwards
    delta = relocate_delta(inst, sol, 1, 2, 0, 0)
    before = sol.cost
    sol.apply_relocate(inst, 1, 2, 0, 0, delta)
    assert len(sol.routes[0]) == 1
    true_cost, true_load = sol.recompute(inst)
    assert abs(before + delta - true_cost) < 1e-6
    assert sol.load == true_load
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_relocate_into_empty_route_costs_round_trip():
    # inserting into an empty route creates depot -> c -> depot, i.e. 2 * d[depot][c]
    inst = _toy_inst([[0, 0], [0, 4], [3, 0]], [0, 3, 3], capacity=10)
    sol = Solution([[1, 2], []], inst)
    assert relocate_delta(inst, sol, 0, 1, 1, 0) == (
        inst["dist"][0, 1] - inst["dist"][1, 2] - inst["dist"][2, 0] + 2 * inst["dist"][0, 2]
    )


def test_relocate_identity_move_has_zero_delta():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    # reinserting a customer at the slot it was taken from must cancel exactly
    for from_route, route in enumerate(sol.routes):
        for from_pos in range(len(route)):
            assert relocate_delta(inst, sol, from_route, from_pos, from_route, from_pos) == 0.0


def test_relocate_improves_until_local_optimum():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    start = sol.cost

    with time_limit(10):
        steps = 0
        while relocate(inst, sol):
            steps += 1
            assert steps < 10_000, "relocate did not reach a local optimum"

    assert steps > 0
    assert sol.cost < start
    true_cost, true_load = sol.recompute(inst)
    assert abs(sol.cost - true_cost) < 1e-6
    assert sol.load == true_load
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_delta_swap():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    rng = random.Random(20260803)

    adjacent_seen = 0
    applied, attempts = 0, 0
    while applied < 1000:
        attempts += 1
        assert attempts < 100_000, "could not sample enough valid moves"

        occupied = [r for r, route in enumerate(sol.routes) if route]
        route_a = rng.choice(occupied)
        route_b = rng.choice(occupied)
        pos_a = rng.randrange(len(sol.routes[route_a]))
        pos_b = rng.randrange(len(sol.routes[route_b]))
        if route_a == route_b and pos_a == pos_b:
            continue

        first = sol.routes[route_a][pos_a]
        second = sol.routes[route_b][pos_b]
        if route_a != route_b:
            shift = inst["demands"][second] - inst["demands"][first]
            if (sol.load[route_a] + shift > inst["capacity"]
                    or sol.load[route_b] - shift > inst["capacity"]):
                continue
        elif abs(pos_a - pos_b) == 1:
            adjacent_seen += 1

        delta = swap_delta(inst, sol, route_a, pos_a, route_b, pos_b)
        before = sol.cost
        sol.apply_swap(inst, route_a, pos_a, route_b, pos_b, delta)

        true_cost, true_load = sol.recompute(inst)
        assert abs(before + delta - true_cost) < 1e-6
        assert abs(sol.cost - true_cost) < 1e-6
        assert sol.load == true_load

        applied += 1

    # the adjacent intra-route pair is the one case the general formula gets wrong,
    # so the sample is only meaningful if it contains some
    assert adjacent_seen > 0, "random sample never hit the adjacent-pair case"
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_swap_adjacent_customers_hand_computed():
    # depot -> 1 -> 2 -> 3 -> depot  =  3 + 4 + 3 + 4  = 14
    # depot -> 2 -> 1 -> 3 -> depot  =  5 + 4 + 5 + 4  = 18
    inst = _toy_inst([[0, 0], [0, 3], [4, 3], [4, 0]], [0, 1, 1, 1], capacity=10)
    sol = Solution([[1, 2, 3]], inst)
    assert sol.cost == 14.0

    delta = swap_delta(inst, sol, 0, 0, 0, 1)
    assert delta == 4.0

    sol.apply_swap(inst, 0, 0, 0, 1, delta)
    assert sol.routes[0] == [2, 1, 3]
    assert sol.cost == 18.0
    assert sol.recompute(inst)[0] == 18.0


def test_swap_reversing_two_customer_route_is_free():
    # both customers sit between the depot and each other, so the tour is symmetric
    inst = _toy_inst([[0, 0], [0, 3], [4, 3]], [0, 1, 1], capacity=10)
    sol = Solution([[1, 2]], inst)
    assert swap_delta(inst, sol, 0, 0, 0, 1) == 0.0


def test_swap_argument_order_does_not_matter():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    rng = random.Random(7)
    for _ in range(200):
        route_a = rng.randrange(len(sol.routes))
        route_b = rng.randrange(len(sol.routes))
        pos_a = rng.randrange(len(sol.routes[route_a]))
        pos_b = rng.randrange(len(sol.routes[route_b]))
        if route_a == route_b and pos_a == pos_b:
            continue
        forward = swap_delta(inst, sol, route_a, pos_a, route_b, pos_b)
        backward = swap_delta(inst, sol, route_b, pos_b, route_a, pos_a)
        assert abs(forward - backward) < 1e-9


def test_swap_improves_until_local_optimum():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    start = sol.cost

    with time_limit(10):
        steps = 0
        while swap(inst, sol):
            steps += 1
            assert steps < 10_000, "swap did not reach a local optimum"

    assert steps > 0
    assert sol.cost < start
    true_cost, true_load = sol.recompute(inst)
    assert abs(sol.cost - true_cost) < 1e-6
    assert sol.load == true_load
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_delta_two_opt():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    rng = random.Random(20260804)

    touches_depot_side, touches_tail, whole_route = 0, 0, 0
    for _ in range(1000):
        reversible = [r for r, route in enumerate(sol.routes) if len(route) >= 2]
        route = rng.choice(reversible)
        length = len(sol.routes[route])
        start = rng.randrange(length - 1)
        end = rng.randrange(start + 1, length)

        touches_depot_side += start == 0
        touches_tail += end == length - 1
        whole_route += start == 0 and end == length - 1

        delta = two_opt_delta(inst, sol, route, start, end)
        before = sol.cost
        sol.apply_two_opt(route, start, end, delta)

        true_cost, true_load = sol.recompute(inst)
        assert abs(before + delta - true_cost) < 1e-6
        assert abs(sol.cost - true_cost) < 1e-6
        assert sol.load == true_load

    # segments bounded by the depot have no real predecessor or successor, so the
    # sample is only meaningful if it reaches them
    assert touches_depot_side > 0 and touches_tail > 0 and whole_route > 0
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_two_opt_hand_computed():
    # depot -> 2 -> 1 -> 3 -> depot  =  5 + 4 + 5 + 4  = 18  (the first two cross)
    # reversing positions 0..1 undoes the crossing:
    # depot -> 1 -> 2 -> 3 -> depot  =  3 + 4 + 3 + 4  = 14
    inst = _toy_inst([[0, 0], [0, 3], [4, 3], [4, 0]], [0, 1, 1, 1], capacity=10)
    sol = Solution([[2, 1, 3]], inst)
    assert sol.cost == 18.0

    delta = two_opt_delta(inst, sol, 0, 0, 1)
    assert delta == -4.0

    sol.apply_two_opt(0, 0, 1, delta)
    assert sol.routes[0] == [1, 2, 3]
    assert sol.cost == 14.0
    assert sol.recompute(inst)[0] == 14.0


def test_two_opt_whole_route_reversal_is_free():
    # a reversed route visits the same edges in the opposite direction, and the
    # matrix is symmetric, so the cost cannot change
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    for route, customers in enumerate(sol.routes):
        if len(customers) >= 2:
            assert two_opt_delta(inst, sol, route, 0, len(customers) - 1) == 0.0


def test_two_opt_preserves_route_membership():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    before_sets = [set(route) for route in sol.routes]
    before_load = list(sol.load)

    with time_limit(10):
        while two_opt(inst, sol):
            pass

    assert [set(route) for route in sol.routes] == before_sets
    assert sol.load == before_load


def test_two_opt_improves_until_local_optimum():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    start = sol.cost

    with time_limit(10):
        steps = 0
        while two_opt(inst, sol):
            steps += 1
            assert steps < 10_000, "two_opt did not reach a local optimum"

    assert steps > 0
    assert sol.cost < start
    true_cost, true_load = sol.recompute(inst)
    assert abs(sol.cost - true_cost) < 1e-6
    assert sol.load == true_load
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_delta_or_opt():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    rng = random.Random(20260805)

    seen = {"len2": 0, "len3": 0, "forward": 0, "reversed": 0, "intra": 0,
            "inter": 0, "at_head": 0, "at_tail": 0, "whole_route": 0}

    applied, attempts = 0, 0
    while applied < 1000:
        attempts += 1
        assert attempts < 200_000, "could not sample enough valid moves"

        movable = [r for r, route in enumerate(sol.routes) if len(route) >= min(SEGMENT_LENGTHS)]
        from_route = rng.choice(movable)
        length = rng.choice(SEGMENT_LENGTHS)
        source = sol.routes[from_route]
        if len(source) < length:
            continue

        start = rng.randrange(len(source) - length + 1)
        to_route = rng.randrange(len(sol.routes))
        same_route = to_route == from_route
        slots = len(sol.routes[to_route]) - length if same_route else len(sol.routes[to_route])
        if slots < 0:
            continue

        to_pos = rng.randint(0, slots)
        reverse = rng.choice((False, True))
        if same_route and to_pos == start and not reverse:
            continue

        segment = source[start:start + length]
        demand = sum(inst["demands"][c] for c in segment)
        if not same_route and sol.load[to_route] + demand > inst["capacity"]:
            continue

        seen["len2" if length == 2 else "len3"] += 1
        seen["reversed" if reverse else "forward"] += 1
        seen["intra" if same_route else "inter"] += 1
        seen["at_head"] += start == 0
        seen["at_tail"] += start + length == len(source)
        seen["whole_route"] += length == len(source)

        delta = or_opt_delta(inst, sol, from_route, start, length, to_route, to_pos, reverse)
        before = sol.cost
        sol.apply_or_opt(inst, from_route, start, length, to_route, to_pos, reverse, delta)

        true_cost, true_load = sol.recompute(inst)
        assert abs(before + delta - true_cost) < 1e-6
        assert abs(sol.cost - true_cost) < 1e-6
        assert sol.load == true_load

        applied += 1

    # this move has the most parameters of the five, so the sample is only worth
    # anything if it reached every branch of them
    assert all(count > 0 for count in seen.values()), seen
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_or_opt_hand_computed():
    # five points on a line; the route visits the far pair before the near pair
    # [3,4,1,2] = 10 + 1 + 10 + 1 + 2 = 24
    # moving segment [1,2] to the front gives [1,2,3,4] = 1 + 1 + 8 + 1 + 11 = 22
    inst = _toy_inst([[0, 0], [1, 0], [2, 0], [10, 0], [11, 0]], [0, 1, 1, 1, 1], capacity=10)
    sol = Solution([[3, 4, 1, 2]], inst)
    assert sol.cost == 24.0

    delta = or_opt_delta(inst, sol, 0, 2, 2, 0, 0, False)
    assert delta == -2.0

    sol.apply_or_opt(inst, 0, 2, 2, 0, 0, False, delta)
    assert sol.routes[0] == [1, 2, 3, 4]
    assert sol.cost == 22.0
    assert sol.recompute(inst)[0] == 22.0


def test_or_opt_identity_move_has_zero_delta():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    for route, customers in enumerate(sol.routes):
        for length in SEGMENT_LENGTHS:
            for start in range(len(customers) - length + 1):
                delta = or_opt_delta(inst, sol, route, start, length, route, start, False)
                assert delta == 0.0


def test_or_opt_reversed_in_place_matches_two_opt():
    # reversing a segment without moving it is exactly a 2-opt of that segment,
    # so the two independently derived formulas have to agree
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    compared = 0
    for route, customers in enumerate(sol.routes):
        for length in SEGMENT_LENGTHS:
            for start in range(len(customers) - length + 1):
                as_or_opt = or_opt_delta(inst, sol, route, start, length, route, start, True)
                as_two_opt = two_opt_delta(inst, sol, route, start, start + length - 1)
                assert abs(as_or_opt - as_two_opt) < 1e-9
                compared += 1
    assert compared > 0


def test_or_opt_into_empty_route():
    # random sampling never empties a route, so the empty target is checked here
    inst = _toy_inst([[0, 0], [1, 0], [2, 0], [10, 0]], [0, 2, 2, 2], capacity=10)
    sol = Solution([[1, 2, 3], []], inst)

    delta = or_opt_delta(inst, sol, 0, 0, 2, 1, 0, False)
    before = sol.cost
    sol.apply_or_opt(inst, 0, 0, 2, 1, 0, False, delta)

    assert sol.routes == [[3], [1, 2]]
    true_cost, true_load = sol.recompute(inst)
    assert abs(before + delta - true_cost) < 1e-6
    assert sol.load == true_load == [2, 4]
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_or_opt_improves_until_local_optimum():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    start = sol.cost

    with time_limit(30):
        steps = 0
        while or_opt(inst, sol):
            steps += 1
            assert steps < 10_000, "or_opt did not reach a local optimum"

    assert steps > 0
    assert sol.cost < start
    true_cost, true_load = sol.recompute(inst)
    assert abs(sol.cost - true_cost) < 1e-6
    assert sol.load == true_load
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_delta_cross_exchange():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    rng = random.Random(20260806)

    combinations = set()
    boundaries = {"a_head": 0, "a_tail": 0, "b_head": 0, "b_tail": 0}

    applied, attempts = 0, 0
    while applied < 1000:
        attempts += 1
        assert attempts < 300_000, "could not sample enough valid moves"

        route_a = rng.randrange(len(sol.routes))
        route_b = rng.randrange(len(sol.routes))
        if route_a == route_b:
            continue
        route_a, route_b = min(route_a, route_b), max(route_a, route_b)

        first = sol.routes[route_a]
        second = sol.routes[route_b]
        length_a = rng.choice(CROSS_SEGMENT_LENGTHS)
        length_b = rng.choice(CROSS_SEGMENT_LENGTHS)
        if len(first) < length_a or len(second) < length_b:
            continue

        start_a = rng.randrange(len(first) - length_a + 1)
        start_b = rng.randrange(len(second) - length_b + 1)

        demand_a = sum(inst["demands"][c] for c in first[start_a:start_a + length_a])
        demand_b = sum(inst["demands"][c] for c in second[start_b:start_b + length_b])
        if (sol.load[route_a] - demand_a + demand_b > inst["capacity"]
                or sol.load[route_b] - demand_b + demand_a > inst["capacity"]):
            continue

        combinations.add((length_a, length_b))
        boundaries["a_head"] += start_a == 0
        boundaries["a_tail"] += start_a + length_a == len(first)
        boundaries["b_head"] += start_b == 0
        boundaries["b_tail"] += start_b + length_b == len(second)

        delta = cross_exchange_delta(inst, sol, route_a, start_a, length_a,
                                     route_b, start_b, length_b)
        before = sol.cost
        sol.apply_cross_exchange(inst, route_a, start_a, length_a,
                                 route_b, start_b, length_b, delta)

        true_cost, true_load = sol.recompute(inst)
        assert abs(before + delta - true_cost) < 1e-6
        assert abs(sol.cost - true_cost) < 1e-6
        assert sol.load == true_load

        applied += 1

    expected = {(a, b) for a in CROSS_SEGMENT_LENGTHS for b in CROSS_SEGMENT_LENGTHS}
    assert combinations == expected, sorted(expected - combinations)
    assert all(count > 0 for count in boundaries.values()), boundaries
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_cross_exchange_hand_computed():
    # four points on a line, each route holding one near and one far customer
    # [1,4] = 1 + 10 + 11 = 22 and [3,2] = 10 + 8 + 2 = 20, total 42
    # trading the tails gives [1,2] = 4 and [3,4] = 22, total 26
    inst = _toy_inst([[0, 0], [1, 0], [2, 0], [10, 0], [11, 0]], [0, 1, 1, 1, 1], capacity=10)
    sol = Solution([[1, 4], [3, 2]], inst)
    assert sol.cost == 42.0

    delta = cross_exchange_delta(inst, sol, 0, 1, 1, 1, 1, 1)
    assert delta == -16.0

    sol.apply_cross_exchange(inst, 0, 1, 1, 1, 1, 1, delta)
    assert sol.routes == [[1, 2], [3, 4]]
    assert sol.cost == 26.0
    assert sol.recompute(inst)[0] == 26.0


def test_cross_exchange_unit_segments_match_swap():
    # trading one customer for one between two routes is exactly an inter-route
    # swap, so the two independently derived formulas have to agree
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    compared = 0
    for route_a in range(len(sol.routes)):
        for route_b in range(route_a + 1, len(sol.routes)):
            for pos_a in range(len(sol.routes[route_a])):
                for pos_b in range(len(sol.routes[route_b])):
                    as_cross = cross_exchange_delta(inst, sol, route_a, pos_a, 1,
                                                    route_b, pos_b, 1)
                    as_swap = swap_delta(inst, sol, route_a, pos_a, route_b, pos_b)
                    assert abs(as_cross - as_swap) < 1e-9
                    compared += 1
    assert compared > 0


def test_cross_exchange_rejects_same_route():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    try:
        sol.apply_cross_exchange(inst, 0, 0, 1, 0, 1, 1, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a same-route cross exchange")


def test_cross_exchange_improves_until_local_optimum():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    start = sol.cost

    with time_limit(30):
        steps = 0
        while cross_exchange(inst, sol):
            steps += 1
            assert steps < 10_000, "cross_exchange did not reach a local optimum"

    assert steps > 0
    assert sol.cost < start
    true_cost, true_load = sol.recompute(inst)
    assert abs(sol.cost - true_cost) < 1e-6
    assert sol.load == true_load
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


class _RecordingSelector:
    # wraps the baseline so a test can inspect what vnd asked for
    def __init__(self):
        self.inner = FixedSelector()
        self.picks = []
        self.updates = []

    def pick(self, state, available):
        action = self.inner.pick(state, available)
        self.picks.append((state, tuple(available), action))
        return action

    def update(self, state, action, reward, next_state):
        self.updates.append((state, action, reward, next_state))


def test_reward_registry_holds_both_functions():
    assert set(REWARDS) == {"improvement", "improvement_per_second"}


def test_improvement_reward_ignores_time():
    # the per-attempt reward must not change when the same move takes longer
    fast = improvement_reward(10.0, 1000.0, True, 0.001, 0.0)
    slow = improvement_reward(10.0, 1000.0, True, 5.0, 0.0)
    assert fast == slow == 0.01
    assert improvement_reward(0.0, 1000.0, False, 5.0, -0.5) == -0.5


def test_improvement_per_second_matches_the_old_reward_at_median_cost():
    # a median-cost call scores exactly what the per-attempt reward would, which is
    # what keeps Q magnitudes comparable across the two reward functions
    plain = improvement_reward(10.0, 1000.0, True, REFERENCE_CALL_MS, 0.0)
    timed = improvement_per_second(10.0, 1000.0, True, REFERENCE_CALL_MS, 0.0)
    assert abs(plain - timed) < 1e-12


def test_improvement_per_second_prefers_the_cheaper_call():
    cheap = improvement_per_second(10.0, 1000.0, True, 0.005, 0.0)
    dear = improvement_per_second(10.0, 1000.0, True, 0.160, 0.0)
    assert cheap > dear
    assert abs(cheap / dear - 32.0) < 1e-6  # ratio is exactly the ratio of times


def test_improvement_per_second_punishes_the_costlier_dead_end():
    cheap_miss = improvement_per_second(0.0, 1000.0, False, 0.005, -1.0)
    dear_miss = improvement_per_second(0.0, 1000.0, False, 0.160, -1.0)
    assert dear_miss < cheap_miss < 0
    # with the default penalty the term is inert, exactly as in the per-attempt reward
    assert improvement_per_second(0.0, 1000.0, False, 5.0, 0.0) == 0.0


def test_improvement_per_second_floors_instant_calls():
    # a call timed at zero must not divide by zero or blow the reward up unboundedly
    floored = improvement_per_second(10.0, 1000.0, True, 0.0, 0.0)
    assert floored == improvement_per_second(10.0, 1000.0, True, 1e-9, 0.0)
    assert floored == 0.01 * (REFERENCE_CALL_MS / 0.001)


def test_phase_of_thirds():
    assert phase_of(0.0) == EARLY
    assert phase_of(0.32) == EARLY
    assert phase_of(1 / 3) == MID
    assert phase_of(0.65) == MID
    assert phase_of(2 / 3) == LATE
    assert phase_of(1.0) == LATE


def test_vnd_reaches_a_local_optimum_for_every_neighborhood():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)

    with time_limit(30):
        vnd(inst, sol, DEFAULT_NEIGHBORHOODS, FixedSelector())

    # the postcondition of VND: nothing improves any more
    for neighborhood in DEFAULT_NEIGHBORHOODS:
        assert not neighborhood(inst, sol)


def test_vnd_keeps_cost_and_load_consistent():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    start = sol.cost

    with time_limit(30):
        vnd(inst, sol, DEFAULT_NEIGHBORHOODS, FixedSelector())

    true_cost, true_load = sol.recompute(inst)
    assert sol.cost < start
    assert abs(sol.cost - true_cost) < 1e-6
    assert sol.load == true_load
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_vnd_fixed_selector_follows_textbook_order():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    selector = _RecordingSelector()

    with time_limit(30):
        vnd(inst, sol, DEFAULT_NEIGHBORHOODS, selector)

    # the first pick is neighborhood 0; after an improvement the search returns
    # to 0, after a failure it moves one index up
    assert selector.picks[0][2] == 0
    for (_, _, action), (_, _, _, next_state) in zip(selector.picks[1:], selector.updates):
        improved = next_state[1]
        assert action == 0 if improved else action > 0


def test_vnd_state_and_reward_are_well_formed():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    selector = _RecordingSelector()

    with time_limit(30):
        vnd(inst, sol, DEFAULT_NEIGHBORHOODS, selector, progress=0.5)

    assert len(selector.picks) == len(selector.updates)
    for (state, available, action), (u_state, u_action, reward, next_state) in zip(
            selector.picks, selector.updates):
        assert state == u_state and action == u_action
        assert state[0] == MID and state[1] in (0, 1)      # progress=0.5 -> mid phase
        assert next_state[0] == MID and next_state[1] in (0, 1)
        assert action in available
        # reward is a normalized improvement: zero on failure, positive on success
        assert reward >= 0.0
        assert (reward > 0.0) == bool(next_state[1])


def test_vnd_stats_add_up():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    start = sol.cost

    with time_limit(30):
        stats = vnd(inst, sol, DEFAULT_NEIGHBORHOODS, FixedSelector())

    assert sum(stats.calls) == stats.steps
    assert all(good <= total for good, total in zip(stats.improvements, stats.calls))
    assert abs(stats.gain - (start - sol.cost)) < 1e-6
    assert abs(sum(stats.gains) - stats.gain) < 1e-6   # per-neighborhood gains partition it
    # a neighborhood only banks gain when it improved, and never without being called
    for calls, improvements, gain in zip(stats.calls, stats.improvements, stats.gains):
        assert gain >= 0.0 and (gain == 0.0 or improvements > 0)
        assert improvements == 0 or calls > 0
    # every neighborhood has to be tried at least once before VND may stop
    assert all(count > 0 for count in stats.calls)


def test_vnd_terminates_under_any_selector():
    # the loop must not depend on the selector being sensible: a greedy Q-agent
    # would otherwise spin on one neighborhood forever
    inst = parse_vrp(INSTANCE)
    for seed in range(5):
        sol = _nn_solution(inst)
        with time_limit(30):
            stats = vnd(inst, sol, DEFAULT_NEIGHBORHOODS, RandomSelector(seed))
        for neighborhood in DEFAULT_NEIGHBORHOODS:
            assert not neighborhood(inst, sol)
        assert stats.steps > 0
        assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_vnd_progress_number_freezes_the_phase():
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    selector = _RecordingSelector()

    with time_limit(30):
        vnd(inst, sol, DEFAULT_NEIGHBORHOODS, selector, progress=0.5)

    assert {state[0] for state, _, _ in selector.picks} == {MID}


def test_vnd_progress_callable_is_read_every_iteration():
    # a callable is re-read on each pick and again after each move, so the phase
    # tracks the run instead of being fixed when vnd was entered
    reads = []

    def creeping_progress():
        reads.append(len(reads))
        return len(reads) / 40.0  # crosses both thirds within one VND call

    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    selector = _RecordingSelector()

    with time_limit(30):
        vnd(inst, sol, DEFAULT_NEIGHBORHOODS, selector, progress=creeping_progress)

    assert len(reads) == 2 * len(selector.picks)  # once per pick, once per update
    assert {state[0] for state, _, _ in selector.picks} == {EARLY, MID, LATE}
    # phases only ever move forward
    phases = [state[0] for state, _, _ in selector.picks]
    assert phases == sorted(phases)


def test_elapsed_progress_tracks_the_clock():
    read = elapsed_progress(start=time.time() - 15.0, budget=30.0)
    assert 0.45 < read() < 0.55
    assert phase_of(read()) == MID

    finished = elapsed_progress(start=time.time() - 60.0, budget=30.0)
    assert finished() == 1.0  # clamped, never runs past the end of the budget


def test_random_selector_only_picks_available_and_is_seeded():
    first = RandomSelector(seed=11)
    second = RandomSelector(seed=11)
    for _ in range(50):
        available = [0, 2, 4]
        choice = first.pick((EARLY, 0), available)
        assert choice in available
        assert choice == second.pick((EARLY, 0), available)


def test_vnd_rejects_a_selector_that_ignores_availability():
    class Stubborn:
        def pick(self, state, available):
            return 0  # keeps choosing a neighborhood that has already failed

        def update(self, state, action, reward, next_state):
            pass

    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    with time_limit(30):
        vnd(inst, sol, DEFAULT_NEIGHBORHOODS, FixedSelector())  # drive to a local optimum
        try:
            vnd(inst, sol, DEFAULT_NEIGHBORHOODS, Stubborn())
        except ValueError:
            return
    raise AssertionError("expected ValueError when the selector ignores `available`")


def _agent(n_actions=len(DEFAULT_NEIGHBORHOODS), **kwargs):
    settings = dict(alpha=0.1, gamma=0.9, eps_start=0.9, eps_end=0.05, seed=3)
    settings.update(kwargs)
    return QAgent(n_actions, **settings)


def test_qagent_table_shape_and_state_indexing():
    agent = _agent()
    assert agent.q.shape == (N_STATES, len(DEFAULT_NEIGHBORHOODS)) == (6, 5)
    assert not agent.q.any()

    seen = {agent.get_state(phase, improved)
            for phase in (EARLY, MID, LATE) for improved in (0, 1)}
    assert seen == set(range(N_STATES))  # the six states map onto distinct rows


def test_qagent_epsilon_decays_on_the_budget_not_on_steps():
    agent = _agent(eps_start=1.0, eps_end=0.0)
    assert agent.epsilon == 1.0  # no clock attached yet

    consumed = [0.0]
    agent.set_progress(lambda: consumed[0])
    for fraction, expected in [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (1.0, 0.0)]:
        consumed[0] = fraction
        assert abs(agent.epsilon - expected) < 1e-12

    consumed[0] = 5.0
    assert agent.epsilon == 0.0  # clamped, never runs past eps_end

    # step count must not enter into it: the arms differ in steps by up to 2x
    consumed[0] = 0.5
    agent.steps = 10_000
    assert abs(agent.epsilon - 0.5) < 1e-12


def test_qagent_update_matches_the_formula():
    agent = _agent(alpha=0.5, gamma=0.9)

    # first update off a zero table: target = 1.0 + 0.9 * 0
    agent.update((EARLY, 0), 2, 1.0, (MID, 1))
    assert abs(agent.q[0, 2] - 0.5) < 1e-12
    assert agent.steps == 1

    # second update bootstraps off the best value in the next state's row
    agent.q[agent.get_state(MID, 1)] = [0.0, 0.0, 2.0, 0.0, 0.0]
    agent.update((EARLY, 0), 2, 1.0, (MID, 1))
    assert abs(agent.q[0, 2] - 1.65) < 1e-12  # 0.5 + 0.5 * (1.0 + 0.9*2.0 - 0.5)


def test_qagent_never_picks_outside_available():
    agent = _agent(eps_start=0.0, eps_end=0.0)  # pure greedy
    # action 0 is by far the best, but it is not on offer
    agent.q[agent.get_state(EARLY, 0)] = [9.0, 1.0, 0.5, 0.2, 0.1]
    for _ in range(100):
        assert agent.pick((EARLY, 0), [1, 2, 3]) in (1, 2, 3)

    exploring = _agent(eps_start=1.0, eps_end=1.0)  # pure exploration
    for _ in range(100):
        assert exploring.pick((EARLY, 0), [2, 4]) in (2, 4)


def test_qagent_greedy_picks_the_best_available():
    agent = _agent(eps_start=0.0, eps_end=0.0)
    agent.q[agent.get_state(LATE, 1)] = [0.1, 0.7, 0.2, 0.9, 0.3]
    assert agent.pick((LATE, 1), [0, 1, 2]) == 1
    assert agent.pick((LATE, 1), [0, 2, 3]) == 3
    assert agent.pick((LATE, 1), [0, 2]) == 2


def test_qagent_zero_table_does_not_collapse_to_the_fixed_order():
    # with every value tied at zero a first-index rule would make the greedy branch
    # a copy of FixedSelector, and the two arms would be indistinguishable at the start
    agent = _agent(eps_start=0.0, eps_end=0.0)
    picks = {agent.pick((EARLY, 0), [0, 1, 2, 3, 4]) for _ in range(200)}
    assert picks == {0, 1, 2, 3, 4}


def test_qagent_table_evolves_across_shared_vnd_calls():
    # one agent for the whole run: the table is the learned state and must survive
    # from one vnd() call to the next
    inst = parse_vrp(INSTANCE)
    agent = _agent()
    consumed = [0.0]
    agent.set_progress(lambda: consumed[0])
    before = agent.q.copy()

    calls = 25
    total_steps = 0
    with time_limit(60):
        for call in range(calls):
            consumed[0] = call / calls
            sol = _nn_solution(inst)
            stats = vnd(inst, sol, DEFAULT_NEIGHBORHOODS, agent, progress=call / calls)
            total_steps += stats.steps
            assert is_feasible(sol.routes, inst["demands"], inst["capacity"])

    assert agent.q.any(), "Q-table stayed all zeros"
    assert not np.array_equal(agent.q, before)
    assert agent.steps == total_steps          # one learning step per vnd iteration
    assert agent.epsilon < 0.9                 # epsilon actually decayed
    # sweeping progress across the run has to reach every phase
    assert (np.abs(agent.q).sum(axis=1) > 0).sum() == N_STATES


def test_qvnd_behaves_like_the_other_selectors():
    inst = parse_vrp(INSTANCE)
    arms = {
        "fixed": FixedSelector(),
        "random": RandomSelector(seed=5),
        "q": _agent(),
    }
    with time_limit(60):
        for selector in arms.values():
            for call in range(5):
                sol = _nn_solution(inst)
                vnd(inst, sol, DEFAULT_NEIGHBORHOODS, selector, progress=call / 5)
                # same postcondition regardless of who chose the neighborhoods
                for neighborhood in DEFAULT_NEIGHBORHOODS:
                    assert not neighborhood(inst, sol)
                true_cost, true_load = sol.recompute(inst)
                assert abs(sol.cost - true_cost) < 1e-6
                assert sol.load == true_load
                assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_shake_keeps_cost_and_load_in_step_with_routes():
    # shake ignores cost when choosing, but the values Solution carries still have
    # to match the routes afterwards, or every later delta is built on a wrong base
    inst = parse_vrp(INSTANCE)
    rng = random.Random(4)
    for k in (1, 3, 7, 12):
        sol = _nn_solution(inst)
        shake(inst, sol, k, SHAKE_E, rng)
        true_cost, true_load = sol.recompute(inst)
        assert abs(sol.cost - true_cost) < 1e-6
        assert sol.load == true_load


def test_shake_preserves_every_customer():
    inst = parse_vrp(INSTANCE)
    expected = set(range(1, len(inst["demands"])))
    rng = random.Random(5)
    for k in range(1, 15):
        sol = _nn_solution(inst)
        shake(inst, sol, k, SHAKE_E, rng)
        placed = [c for route in sol.routes for c in route]
        assert sorted(placed) == sorted(expected)  # nothing lost or duplicated


def test_shake_actually_changes_the_solution():
    # a single move can land back where it started (same route, same slot, no
    # reversal), so a small k is not guaranteed to change anything; with k this
    # large every shake has to leave a mark
    inst = parse_vrp(INSTANCE)
    rng = random.Random(6)
    for _ in range(20):
        sol = _nn_solution(inst)
        before = [list(route) for route in sol.routes]
        shake(inst, sol, 10, SHAKE_E, rng)
        assert sol.routes != before


def test_shake_keeps_capacity():
    # a target route is only taken when the segment fits, so a feasible solution
    # stays feasible however hard it is shaken
    rng = random.Random(7)
    for name in ("A-n32-k5", "A-n65-k9", "A-n80-k10"):
        inst = parse_vrp(INSTANCE.parent / f"{name}.vrp")
        for k in (1, 5, 12):
            sol = Solution(
                initial_solution_nearest_neighbor(
                    inst["coords"], inst["demands"], inst["capacity"],
                    inst["depot_id"], inst["dist"]
                ),
                inst,
            )
            shake(inst, sol, k, SHAKE_E, rng)
            assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_shake_source_route_stays_eligible_when_full():
    # every route at capacity: a segment still has somewhere to go, because moving it
    # inside its own route shifts no load at all. Testing the source route with the
    # same load + demand <= capacity rule as the others would double-count the
    # segment and wrongly rule it out.
    coords = [[0, 0]] + [[i, i % 3] for i in range(1, 13)]
    inst = _toy_inst(coords, [0] + [20] * 12, capacity=120)
    sol = Solution([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], inst)
    assert sol.load == [120, 120]

    reversals = shake(inst, sol, 20, SHAKE_E, random.Random(8))
    assert reversals == 0  # nothing was forced, the source route absorbed every move
    assert sol.load == [120, 120]
    assert sorted(c for route in sol.routes for c in route) == list(range(1, 13))
    assert is_feasible(sol.routes, inst["demands"], inst["capacity"])


def test_shake_reverse_probability_controls_in_place_moves():
    inst = parse_vrp(INSTANCE)

    sol = _nn_solution(inst)
    assert shake(inst, sol, 50, SHAKE_E, random.Random(1), reverse_p=0.0) == 0

    sol = _nn_solution(inst)
    assert shake(inst, sol, 50, SHAKE_E, random.Random(1), reverse_p=1.0) == 50

    sol = _nn_solution(inst)
    half = shake(inst, sol, 400, SHAKE_E, random.Random(1), reverse_p=0.5)
    assert 160 < half < 240  # roughly half, well away from either extreme


def test_shake_in_place_reversal_leaves_load_untouched():
    # the whole point of the reversal branch: it perturbs order without shifting load
    inst = parse_vrp(INSTANCE)
    sol = _nn_solution(inst)
    before_load = list(sol.load)
    before_sets = [set(route) for route in sol.routes]

    shake(inst, sol, 30, SHAKE_E, random.Random(2), reverse_p=1.0)

    assert sol.load == before_load
    assert [set(route) for route in sol.routes] == before_sets
    true_cost, true_load = sol.recompute(inst)
    assert abs(sol.cost - true_cost) < 1e-6
    assert sol.load == true_load


def test_gvns_returns_a_feasible_best_within_budget():
    inst = parse_vrp(INSTANCE)
    started = time.time()
    best, stats = gvns(inst, RandomSelector(seed=1), seed=1, budget_seconds=2.0)
    elapsed = time.time() - started

    assert 2.0 <= elapsed < 4.0  # stops on the budget, one VND call may overrun it
    assert stats.iterations > 0
    assert is_feasible(best.routes, inst["demands"], inst["capacity"])
    true_cost, true_load = best.recompute(inst)
    assert abs(best.cost - true_cost) < 1e-6
    assert best.load == true_load


def test_gvns_requires_exactly_one_stopping_mode():
    inst = parse_vrp(INSTANCE)
    for kwargs in ({}, {"budget_seconds": 1.0, "max_iterations": 10}):
        try:
            gvns(inst, FixedSelector(), seed=0, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_gvns_iteration_mode_runs_exactly_that_many():
    inst = parse_vrp(INSTANCE)
    for wanted in (5, 25):
        _, stats = gvns(inst, RandomSelector(seed=0), seed=0, max_iterations=wanted)
        assert stats.iterations == wanted


def test_gvns_iteration_mode_is_reproducible():
    # nothing in this mode reads the clock, so the same seed has to give the same
    # answer run to run; this is what makes it usable for regression checks
    inst = parse_vrp(INSTANCE)
    for arm in ("fixed", "random"):
        runs = []
        for _ in range(2):
            best, stats = gvns(inst, make_selector(arm, 0), seed=0, max_iterations=40)
            runs.append((best.cost, [list(r) for r in best.routes],
                         stats.iterations, stats.accepted, stats.vnd_steps))
        assert runs[0] == runs[1], arm


def test_gvns_timed_and_iteration_modes_agree_on_the_phase_feature():
    # the phase has to advance in both modes, otherwise a run in iteration mode
    # would leave the agent's state feature stuck
    inst = parse_vrp(INSTANCE)
    seen = []

    class Watcher:
        def pick(self, state, available):
            seen.append(state[0])
            return min(available)

        def update(self, state, action, reward, next_state):
            pass

    gvns(inst, Watcher(), seed=0, max_iterations=60)
    assert {EARLY, MID, LATE} <= set(seen)


def test_run_single_records_the_stopping_mode():
    row = run_single(INSTANCE, "fixed", seed=0, max_iterations=15)
    assert row["mode"] == "iterations"
    assert row["max_iterations"] == 15 and row["time_budget"] == ""
    assert row["iterations"] == 15

    try:
        run_single(INSTANCE, "fixed", seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no stopping mode is given")


def test_gvns_never_returns_worse_than_the_first_local_optimum():
    inst = parse_vrp(INSTANCE)
    plain = _nn_solution(inst)
    vnd(inst, plain, DEFAULT_NEIGHBORHOODS, FixedSelector())

    best, _ = gvns(inst, FixedSelector(), seed=2, budget_seconds=2.0)
    assert best.cost <= plain.cost  # acceptance is strictly-better, so it cannot regress


def test_gvns_shares_one_agent_across_the_whole_run():
    inst = parse_vrp(INSTANCE)
    agent = _agent()
    best, stats = gvns(inst, agent, seed=3, budget_seconds=2.0)

    assert agent.steps == stats.vnd_steps      # every VND step went through this agent
    assert agent.q.any()                       # and the table it built survived the run
    assert agent.epsilon < 0.9                 # gvns handed it the run clock
    assert is_feasible(best.routes, inst["demands"], inst["capacity"])


def test_gvns_k_wraps_around_instead_of_growing_without_bound():
    # k climbs on every rejection; with a small k_max it has to wrap back to k_min
    # rather than run away, which a long budget with few accepts would expose
    inst = parse_vrp(INSTANCE)
    sizes = []

    original_shake = gvns.__globals__["shake"]

    def spy(inst_, sol, k, e, rng, reverse_p=0.0):
        sizes.append(k)
        return original_shake(inst_, sol, k, e, rng, reverse_p)

    gvns.__globals__["shake"] = spy
    try:
        gvns(inst, FixedSelector(), seed=4, budget_seconds=2.0, k_min=1, k_step=1, k_max=4)
    finally:
        gvns.__globals__["shake"] = original_shake

    assert sizes
    assert min(sizes) == 1 and max(sizes) <= 4
    assert sizes.count(1) > 1  # wrapped back round at least once


def test_resolve_neighborhoods_maps_names_and_rejects_unknown():
    from qvnd.experiments import resolve_neighborhoods
    assert resolve_neighborhoods(["relocate", "swap"]) == (relocate, swap)
    assert resolve_neighborhoods([f.__name__ for f in DEFAULT_NEIGHBORHOODS]) \
        == tuple(DEFAULT_NEIGHBORHOODS)
    try:
        resolve_neighborhoods(["relocate", "teleport"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown neighborhood")


def test_run_single_honours_a_neighborhood_subset():
    subset = (relocate, swap)
    row = run_single(INSTANCE, "random", seed=0, budget_seconds=2.0, neighborhoods=subset)
    assert row["cfg_neighborhoods"] == "relocate+swap"
    assert sum(row[f"calls_{f.__name__}"] for f in subset) == row["vnd_steps"]
    assert "calls_cross_exchange" not in row  # the excluded ones are not reported at all


def test_make_selector_covers_every_arm():
    for arm in ARMS:
        selector = make_selector(arm, seed=0)
        assert callable(selector.pick) and callable(selector.update)
        assert selector.pick((EARLY, 0), [1, 3]) in (1, 3)

    try:
        make_selector("greedy", seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown arm")


def test_run_single_records_result_and_configuration():
    row = run_single(INSTANCE, "q", seed=0, budget_seconds=2.0,
                     search={"reverse_p": 0.25, "k_max": 7}, agent={"alpha": 0.3})

    assert row["instance"] == "A-n32-k5" and row["optimum"] == 784
    assert row["arm"] == "q" and row["seed"] == 0
    assert row["cost"] >= row["optimum"]
    assert abs(row["gap"] - 100 * (row["cost"] - 784) / 784) < 1e-9
    assert row["iterations"] > 0 and row["vnd_steps"] > 0
    assert row["infeasible"] == 0

    # the configuration travels with the result, including the defaults not passed in
    assert row["cfg_reverse_p"] == 0.25 and row["cfg_k_max"] == 7
    assert row["cfg_alpha"] == 0.3 and row["cfg_gamma"] == 0.9
    assert row["cfg_reward"] == "improvement"
    assert row["mode"] == "time" and row["time_budget"] == 2.0
    assert sum(row[f"calls_{f.__name__}"] for f in DEFAULT_NEIGHBORHOODS) == row["vnd_steps"]

    # per-neighborhood gain and time have to add up to the run-level figures
    for f in DEFAULT_NEIGHBORHOODS:
        assert row[f"gain_{f.__name__}"] >= 0.0
        assert row[f"ms_{f.__name__}"] > 0.0   # every neighborhood is tried at least once
    assert sum(row[f"ms_{f.__name__}"] for f in DEFAULT_NEIGHBORHOODS) < row["wall"] * 1000


def test_run_batch_covers_the_grid_and_writes_a_documented_csv(tmp_path=None):
    import csv as _csv
    import json as _json
    import tempfile

    seen = []
    rows = run_batch([INSTANCE], ["fixed", "random"], range(2), budget_seconds=1.0,
                     log=lambda line, flush=False: seen.append(line))
    assert len(seen) == 4  # the progress logger is called once per finished run
    assert len(rows) == 4
    assert {(r["arm"], r["seed"]) for r in rows} == {("fixed", 0), ("fixed", 1),
                                                     ("random", 0), ("random", 1)}

    with tempfile.TemporaryDirectory() as tmp:
        target = save_results_csv(rows, Path(tmp) / "out.csv")
        back = list(_csv.DictReader(target.open()))
        assert len(back) == 4
        assert back[0]["instance"] == "A-n32-k5"

        meta = _json.loads(target.with_suffix(".json").read_text())
        assert meta["runs"] == 4 and meta["arms"] == ["fixed", "random"]
        assert meta["seeds"] == [0, 1] and meta["time_budget"] == 1.0
        assert meta["mode"] == "time" and meta["comparable_across_arms"] is True
        assert meta["config"]["cfg_reward"] == "improvement"
        assert len(meta["neighborhoods"]) == len(DEFAULT_NEIGHBORHOODS)


def test_summarize_and_paired_comparison_on_known_numbers():
    def row(instance, arm, seed, cost):
        return {"instance": instance, "arm": arm, "seed": seed,
                "cost": cost, "optimum": 100}

    rows = [row("X", "fixed", 0, 110), row("X", "fixed", 1, 130),
            row("X", "q", 0, 105), row("X", "q", 1, 115)]

    fixed, q = summarize(rows)
    assert (fixed["arm"], fixed["best"], fixed["mean"]) == ("fixed", 110, 120.0)
    assert abs(fixed["gap_mean"] - 20.0) < 1e-9
    assert (q["arm"], q["best"], q["mean"]) == ("q", 105, 110.0)
    assert abs(q["gap_best"] - 5.0) < 1e-9

    c = paired_comparison(rows, "q", "fixed")
    assert c["n"] == 2 and c["wins"] == 2 and c["losses"] == 0
    assert abs(c["mean_diff"] - (-10.0)) < 1e-9   # (105-110) and (115-130) -> -5, -15
    assert c["t"] < 0

    assert paired_comparison(rows, "q", "random") is None  # arm absent
    assert "fixed" in format_summary(rows) and "q" in format_comparisons(rows)


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
