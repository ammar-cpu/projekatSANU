def relocate(routes, dist_matrix, demands, capacity):
    # first-improvement: move one customer to another position (same or different route), delta-cost
    pass


def swap(routes, dist_matrix, demands, capacity):
    # first-improvement: exchange two customers (within or between routes), delta-cost
    pass


def two_opt(routes, dist_matrix):
    # first-improvement: reverse a segment within a route, delta-cost
    pass


def or_opt(routes, dist_matrix, demands, capacity):
    # first-improvement: move a segment of 2-3 customers, delta-cost (later extension)
    pass


def cross_exchange(routes, dist_matrix, demands, capacity):
    # first-improvement: exchange segments between two routes, delta-cost (later extension)
    pass
