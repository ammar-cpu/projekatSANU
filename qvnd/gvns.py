def shaking(routes, k, demands, capacity, seed):
    # k random relocate/swap moves (insert/relocate 50/50)
    pass


def gvns(coords, demands, capacity, depot_id, dist_matrix, selector_factory,
         k_min, k_step, k_max, time_budget, seed):
    # outer GVNS: shaking + VND + strict-better acceptance, reset k to k_min
    pass
