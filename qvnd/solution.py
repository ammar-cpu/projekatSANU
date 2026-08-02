import numpy as np


def cost(routes, dist_matrix):
    depot = 0  # data.parse_vrp always remaps the depot to index 0
    total = 0.0
    for route in routes:
        if not route:
            continue
        total += dist_matrix[depot, route[0]]
        total += sum(dist_matrix[a, b] for a, b in zip(route, route[1:]))
        total += dist_matrix[route[-1], depot]
    return total


def is_feasible(routes, demands, capacity):
    visited = set()
    for route in routes:
        route_demand = 0
        for c in route:
            if c in visited:
                return False
            visited.add(c)
            route_demand += demands[c]
        if route_demand > capacity:
            return False
    return visited == set(range(1, len(demands)))


def _check_demands_fit(demands, capacity, depot_id):
    # a customer that cannot fit an empty route makes the instance unsolvable and
    # leaves the constructions below with no valid move
    oversized = [c for c in range(len(demands)) if c != depot_id and demands[c] > capacity]
    if oversized:
        raise ValueError(
            f"invalid instance: customers {oversized} have demand exceeding capacity {capacity}"
        )


def initial_solution_sweep(coords, demands, capacity, depot_id):
    _check_demands_fit(demands, capacity, depot_id)

    depot_x, depot_y = coords[depot_id]
    customers = [i for i in range(len(coords)) if i != depot_id]
    customers.sort(key=lambda c: np.arctan2(coords[c][1] - depot_y, coords[c][0] - depot_x))

    routes = []
    route, load = [], 0
    for c in customers:
        if load + demands[c] > capacity:
            routes.append(route)
            route, load = [], 0
        route.append(c)
        load += demands[c]
    if route:
        routes.append(route)
    return routes


def initial_solution_nearest_neighbor(coords, demands, capacity, depot_id, dist_matrix):
    _check_demands_fit(demands, capacity, depot_id)

    unvisited = set(range(len(coords))) - {depot_id}
    routes = []
    while unvisited:
        # the guard above means every customer fits an empty route, so each pass
        # places at least one customer and the outer loop is guaranteed to end
        route, load, current = [], 0, depot_id
        while True:
            candidates = [c for c in unvisited if load + demands[c] <= capacity]
            if not candidates:
                break
            nxt = min(candidates, key=lambda c: dist_matrix[current, c])
            route.append(nxt)
            load += demands[nxt]
            unvisited.remove(nxt)
            current = nxt
        routes.append(route)
    return routes
