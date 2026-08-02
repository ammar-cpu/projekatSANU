import numpy as np

DEPOT = 0  # parse_vrp always puts the depot at index 0


def cost(routes, dist_matrix):
    total = 0.0
    for route in routes:
        if not route:  # empty routes are allowed
            continue
        total += dist_matrix[DEPOT, route[0]]
        total += sum(dist_matrix[a, b] for a, b in zip(route, route[1:]))
        total += dist_matrix[route[-1], DEPOT]
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
    # a customer larger than the vehicle makes the instance unsolvable: the
    # constructions below would have no valid move left
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
        # every customer fits an empty route, so each pass places at least one
        # and the outer loop terminates
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


class Solution:
    # routes plus the state kept in step with them: per-route load and total cost
    # are updated by each move, never recomputed

    def __init__(self, routes, inst):
        self.routes = [list(route) for route in routes]
        self.load = [int(sum(inst["demands"][c] for c in route)) for route in self.routes]
        self.cost = cost(self.routes, inst["dist"])

    def copy(self):
        clone = object.__new__(Solution)
        clone.routes = [list(route) for route in self.routes]
        clone.load = list(self.load)
        clone.cost = self.cost
        return clone

    def apply_relocate(self, inst, from_route, from_pos, to_route, to_pos, delta):
        # to_pos indexes the target route after the customer is taken out, so for
        # an intra-route move the pop has to come first
        customer = self.routes[from_route].pop(from_pos)
        self.routes[to_route].insert(to_pos, customer)

        demand = int(inst["demands"][customer])
        self.load[from_route] -= demand
        self.load[to_route] += demand  # nets out to zero when the route is the same

        self.cost += delta

    def apply_swap(self, inst, route_a, pos_a, route_b, pos_b, delta):
        first = self.routes[route_a][pos_a]
        second = self.routes[route_b][pos_b]
        self.routes[route_a][pos_a] = second
        self.routes[route_b][pos_b] = first

        demands = inst["demands"]
        shift = int(demands[second]) - int(demands[first])
        self.load[route_a] += shift
        self.load[route_b] -= shift  # nets out to zero when the route is the same

        self.cost += delta

    def apply_or_opt(self, inst, from_route, start, length, to_route, to_pos, reverse, delta):
        # to_pos indexes the target route after the segment is taken out, so for
        # an intra-route move the deletion has to come first
        source = self.routes[from_route]
        segment = source[start:start + length]
        del source[start:start + length]
        if reverse:
            segment.reverse()
        self.routes[to_route][to_pos:to_pos] = segment

        demand = int(sum(inst["demands"][c] for c in segment))
        self.load[from_route] -= demand
        self.load[to_route] += demand  # nets out to zero when the route is the same

        self.cost += delta

    def apply_cross_exchange(self, inst, route_a, start_a, length_a,
                             route_b, start_b, length_b, delta):
        if route_a == route_b:
            # the slice assignments would clobber each other, and the delta
            # assumes four independent broken edges
            raise ValueError("cross_exchange is defined between two distinct routes")

        first = self.routes[route_a]
        second = self.routes[route_b]
        segment_a = first[start_a:start_a + length_a]
        segment_b = second[start_b:start_b + length_b]
        first[start_a:start_a + length_a] = segment_b
        second[start_b:start_b + length_b] = segment_a

        demands = inst["demands"]
        shift = int(sum(demands[c] for c in segment_b)) - int(sum(demands[c] for c in segment_a))
        self.load[route_a] += shift
        self.load[route_b] -= shift

        self.cost += delta

    def apply_two_opt(self, route, start, end, delta):
        # same customers, same route: load is unchanged, so no instance data here
        segment = self.routes[route]
        segment[start:end + 1] = segment[start:end + 1][::-1]
        self.cost += delta

    def recompute(self, inst):
        # for tests only: the incremental state should never need rebuilding
        return (
            cost(self.routes, inst["dist"]),
            [int(sum(inst["demands"][c] for c in route)) for route in self.routes],
        )
