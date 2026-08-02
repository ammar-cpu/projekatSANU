class FixedSelector:
    # baseline: fixed order of neighborhoods, same interface as the Q-selector

    def __init__(self, neighborhoods):
        pass

    def pick(self, state):
        pass

    def update(self, state, action, reward, next_state):
        pass


def vnd(routes, dist_matrix, demands, capacity, neighborhoods, selector):
    # VND loop: selector.pick chooses the neighborhood, return to the first after every improvement,
    # stop when none improves; calls selector.update after every attempt
    pass
