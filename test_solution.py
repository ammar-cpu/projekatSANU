import signal
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from qvnd.data import parse_vrp, compute_distance_matrix
from qvnd.solution import (
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
    # pins the TSPLIB EUC_2D convention the CVRPLIB optima are stated under:
    # every distance must be a whole number, not raw floating-point Euclidean
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
    # capacity is deliberately generous so only the duplication can cause failure
    assert not is_feasible([[1, 2], [2, 3]], demands, capacity=100)


def test_duplicate_within_same_route_rejected():
    demands = np.array([0, 10, 10, 10])
    assert not is_feasible([[1, 2, 2], [3]], demands, capacity=100)


def test_missing_customer_rejected():
    demands = np.array([0, 10, 10, 10])
    assert not is_feasible([[1, 2]], demands, capacity=100)


def test_nn_terminates_on_infeasible_demand():
    # capacity 10 with demands 9, 9, 9, 2: no two customers share a route except
    # none at all, so the heuristic is forced to open several routes
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
    # empty routes are what the missing guard used to leak; local search operators
    # are not expected to cope with them
    inst = parse_vrp(INSTANCE)
    dist = compute_distance_matrix(inst["coords"])
    args = (inst["coords"], inst["demands"], inst["capacity"], inst["depot_id"])

    for routes in (initial_solution_sweep(*args),
                   initial_solution_nearest_neighbor(*args, dist)):
        assert all(len(route) > 0 for route in routes)
        assert is_feasible(routes, inst["demands"], inst["capacity"])


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
