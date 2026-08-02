from .solution import DEPOT

EPS = 1e-9  # a move counts as improving only if it beats this


def _node(route, pos, cut_start=None, cut_length=1):
    # node at `pos` once `cut_length` entries from `cut_start` are taken out of
    # `route`. The depot stands in for both ends, so boundary edges need no
    # special case anywhere below.
    length = len(route) - (0 if cut_start is None else cut_length)
    if pos < 0 or pos >= length:
        return DEPOT
    if cut_start is not None and pos >= cut_start:
        pos += cut_length
    return route[pos]


def relocate_delta(inst, sol, from_route, from_pos, to_route, to_pos):
    # six touched edges, three broken and three created:
    #   remove c from between a and b   ->  -d[a][c] - d[c][b] + d[a][b]
    #   insert c between x and y        ->  -d[x][y] + d[x][c] + d[c][y]
    # for an intra-route move x and y come from the route with c already gone, so
    # putting c straight back between a and b cancels to exactly zero
    dist = inst["dist"]
    route = sol.routes[from_route]
    c = route[from_pos]

    a = route[from_pos - 1] if from_pos > 0 else DEPOT
    b = route[from_pos + 1] if from_pos + 1 < len(route) else DEPOT

    removed = from_pos if to_route == from_route else None
    target = sol.routes[to_route]
    x = _node(target, to_pos - 1, removed)
    y = _node(target, to_pos, removed)

    return (dist[a, b] - dist[a, c] - dist[c, b]
            + dist[x, c] + dist[c, y] - dist[x, y])


def relocate(inst, sol):
    # first-improvement: move a customer to another position (same or different route)
    demands = inst["demands"]
    capacity = inst["capacity"]

    for from_route in range(len(sol.routes)):
        route = sol.routes[from_route]
        for from_pos in range(len(route)):
            demand = demands[route[from_pos]]

            for to_route in range(len(sol.routes)):
                same_route = to_route == from_route
                if not same_route and sol.load[to_route] + demand > capacity:
                    continue

                target = sol.routes[to_route]
                # one slot per gap in the target route as it looks after removal
                slots = len(target) - 1 if same_route else len(target)
                for to_pos in range(slots + 1):
                    if same_route and to_pos == from_pos:
                        continue  # puts the customer back where it started

                    delta = relocate_delta(inst, sol, from_route, from_pos, to_route, to_pos)
                    if delta < -EPS:
                        sol.apply_relocate(inst, from_route, from_pos, to_route, to_pos, delta)
                        return True

    return False


def swap_delta(inst, sol, route_a, pos_a, route_b, pos_b):
    # eight touched edges in general: each customer breaks its in- and out-edge and
    # gains the other's. Two customers adjacent in the same route are the exception:
    # they share an edge, and the eight-edge sum both double-counts it and invents
    # edges from a node to itself.
    dist = inst["dist"]
    first_route = sol.routes[route_a]
    second_route = sol.routes[route_b]
    first = first_route[pos_a]
    second = second_route[pos_b]

    if route_a == route_b:
        if pos_a > pos_b:
            pos_a, pos_b = pos_b, pos_a
            first, second = second, first
        if pos_b == pos_a + 1:


            before = _node(first_route, pos_a - 1)
            after = _node(first_route, pos_b + 1)
            # the shared edge is only reversed, and the matrix is symmetric
            return (dist[before, second] + dist[first, after]
                    - dist[before, first] - dist[second, after])

    a1 = _node(first_route, pos_a - 1)
    b1 = _node(first_route, pos_a + 1)
    a2 = _node(second_route, pos_b - 1)
    b2 = _node(second_route, pos_b + 1)

    return (dist[a1, second] + dist[second, b1] - dist[a1, first] - dist[first, b1]
            + dist[a2, first] + dist[first, b2] - dist[a2, second] - dist[second, b2])


def swap(inst, sol):
    # first-improvement: exchange two customers (within or between routes)
    demands = inst["demands"]
    capacity = inst["capacity"]

    for route_a in range(len(sol.routes)):
        for pos_a in range(len(sol.routes[route_a])):
            first = sol.routes[route_a][pos_a]

            # each unordered pair is visited once
            for route_b in range(route_a, len(sol.routes)):
                same_route = route_b == route_a
                start = pos_a + 1 if same_route else 0
                for pos_b in range(start, len(sol.routes[route_b])):
                    second = sol.routes[route_b][pos_b]

                    if not same_route:
                        shift = demands[second] - demands[first]
                        if (sol.load[route_a] + shift > capacity
                                or sol.load[route_b] - shift > capacity):
                            continue

                    delta = swap_delta(inst, sol, route_a, pos_a, route_b, pos_b)
                    if delta < -EPS:
                        sol.apply_swap(inst, route_a, pos_a, route_b, pos_b, delta)
                        return True

    return False


def two_opt_delta(inst, sol, route, start, end):
    # only four touched edges: the two bounding the segment are broken and rebuilt
    # crossed over. The edges inside the segment flip direction, which is free on a
    # symmetric matrix. On an asymmetric instance this is wrong and the whole
    # reversed run has to be re-summed.
    dist = inst["dist"]
    customers = sol.routes[route]
    before = _node(customers, start - 1)
    after = _node(customers, end + 1)

    return (dist[before, customers[end]] + dist[customers[start], after]
            - dist[before, customers[start]] - dist[customers[end], after])


def two_opt(inst, sol):
    # first-improvement: reverse a segment within a route. Intra-route only, so
    # capacity cannot be violated and no feasibility check is needed.
    for route in range(len(sol.routes)):
        length = len(sol.routes[route])
        for start in range(length):
            for end in range(start + 1, length):
                delta = two_opt_delta(inst, sol, route, start, end)
                if delta < -EPS:
                    sol.apply_two_opt(route, start, end, delta)
                    return True

    return False


SEGMENT_LENGTHS = (2, 3)  # length 1 is already covered by relocate


def or_opt_delta(inst, sol, from_route, start, length, to_route, to_pos, reverse):
    # relocate generalised to a run of consecutive customers: the segment's internal
    # edges travel with it at no cost, so this is still six touched edges. Reversing
    # the segment flips those internal edges, free on a symmetric matrix, and swaps
    # which end connects to which neighbour.
    dist = inst["dist"]
    source = sol.routes[from_route]
    head = source[start]
    tail = source[start + length - 1]

    a = _node(source, start - 1)
    b = _node(source, start + length)

    cut = start if to_route == from_route else None
    target = sol.routes[to_route]
    x = _node(target, to_pos - 1, cut, length)
    y = _node(target, to_pos, cut, length)

    if reverse:
        head, tail = tail, head

    return (dist[a, b] - dist[a, source[start]] - dist[source[start + length - 1], b]
            + dist[x, head] + dist[tail, y] - dist[x, y])


def or_opt(inst, sol):
    # first-improvement: move a segment of 2-3 customers, in either orientation
    demands = inst["demands"]
    capacity = inst["capacity"]

    for from_route in range(len(sol.routes)):
        source = sol.routes[from_route]
        for length in SEGMENT_LENGTHS:
            for start in range(len(source) - length + 1):
                demand = int(sum(demands[c] for c in source[start:start + length]))

                for to_route in range(len(sol.routes)):
                    same_route = to_route == from_route
                    if not same_route and sol.load[to_route] + demand > capacity:
                        continue

                    target = sol.routes[to_route]
                    slots = len(target) - length if same_route else len(target)
                    for to_pos in range(slots + 1):
                        for reverse in (False, True):
                            if same_route and to_pos == start and not reverse:
                                continue  # puts the segment back exactly as it was

                            delta = or_opt_delta(inst, sol, from_route, start, length,
                                                 to_route, to_pos, reverse)
                            if delta < -EPS:
                                sol.apply_or_opt(inst, from_route, start, length,
                                                 to_route, to_pos, reverse, delta)
                                return True

    return False


CROSS_SEGMENT_LENGTHS = (1, 2, 3)


def cross_exchange_delta(inst, sol, route_a, start_a, length_a, route_b, start_b, length_b):
    # eight touched edges, four broken and four created. Both segments keep their
    # internal edges, which travel along at no cost.
    # Precondition: route_a != route_b. Two segments from one route could overlap,
    # and the four broken edges would no longer be independent.
    dist = inst["dist"]
    first = sol.routes[route_a]
    second = sol.routes[route_b]

    head_a, tail_a = first[start_a], first[start_a + length_a - 1]
    head_b, tail_b = second[start_b], second[start_b + length_b - 1]

    before_a = _node(first, start_a - 1)
    after_a = _node(first, start_a + length_a)
    before_b = _node(second, start_b - 1)
    after_b = _node(second, start_b + length_b)

    return (dist[before_a, head_b] + dist[tail_b, after_a]
            + dist[before_b, head_a] + dist[tail_a, after_b]
            - dist[before_a, head_a] - dist[tail_a, after_a]
            - dist[before_b, head_b] - dist[tail_b, after_b])


def cross_exchange(inst, sol):
    # first-improvement: exchange segments between two routes. Inter-route only, so
    # an empty route never takes part; relocate and or_opt cover that case.
    demands = inst["demands"]
    capacity = inst["capacity"]

    for route_a in range(len(sol.routes)):
        first = sol.routes[route_a]
        # route_b > route_a, so each unordered pair of segments is visited once
        for route_b in range(route_a + 1, len(sol.routes)):
            second = sol.routes[route_b]

            for length_a in CROSS_SEGMENT_LENGTHS:
                for start_a in range(len(first) - length_a + 1):
                    demand_a = int(sum(demands[c] for c in first[start_a:start_a + length_a]))

                    for length_b in CROSS_SEGMENT_LENGTHS:
                        for start_b in range(len(second) - length_b + 1):
                            segment_b = second[start_b:start_b + length_b]
                            shift = int(sum(demands[c] for c in segment_b)) - demand_a
                            if (sol.load[route_a] + shift > capacity
                                    or sol.load[route_b] - shift > capacity):
                                continue

                            delta = cross_exchange_delta(inst, sol, route_a, start_a, length_a,
                                                         route_b, start_b, length_b)
                            if delta < -EPS:
                                sol.apply_cross_exchange(inst, route_a, start_a, length_a,
                                                         route_b, start_b, length_b, delta)
                                return True

    return False
