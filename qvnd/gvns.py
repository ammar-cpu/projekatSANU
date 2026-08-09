import random
import time

from .local_search import EPS, or_opt_delta
from .solution import Solution, initial_solution_nearest_neighbor, is_feasible
from .vnd import DEFAULT_NEIGHBORHOODS, elapsed_progress, improvement_reward, vnd

SHAKE_E = 5  # segment lengths run from 2 to SHAKE_E + 1


def shake(inst, sol, k, e, rng, reverse_p=0.0):
    # k random segment moves applied straight to sol with no regard for cost. Cost is
    # never consulted, but capacity is: the target is drawn from the routes the segment
    # fits into.
    #
    # `reverse_p` is the chance a move is taken as a reversal of the segment where it
    # already lies instead of a move between routes. It perturbs the order within a
    # route without shifting load, which is a weaker kick than a relocation.
    # Returns how many of the k moves were taken that way.
    demands = inst["demands"]
    capacity = inst["capacity"]
    reversals = 0

    for _ in range(k):
        length = rng.randint(2, e + 1)
        sources = [r for r, route in enumerate(sol.routes) if len(route) >= length]
        if not sources:
            continue  # no route long enough for a segment of this length

        from_route = rng.choice(sources)
        start = rng.randrange(len(sol.routes[from_route]) - length + 1)
        segment = sol.routes[from_route][start:start + length]
        demand = int(sum(demands[c] for c in segment))

        if rng.random() < reverse_p:
            to_route, to_pos, reverse = from_route, start, True
            reversals += 1
        else:
            # the source route is listed unconditionally: the segment is already
            # counted in its load, so the test that applies to the others would
            # double-count it. The real condition there is load <= capacity, which
            # holds by invariant. This also keeps the list from ever being empty.
            eligible = [from_route] + [
                r for r in range(len(sol.routes))
                if r != from_route and sol.load[r] + demand <= capacity
            ]
            to_route = rng.choice(eligible)
            same_route = to_route == from_route
            slots = len(sol.routes[to_route]) - length if same_route else len(sol.routes[to_route])
            to_pos = rng.randint(0, slots)
            if rng.random() < 0.5:
                reverse = False              # insert: the segment keeps its orientation
            else:
                reverse = rng.random() < 0.5  # relocate: it may be turned around

        # the delta is still computed, not to judge the move but to keep the cost and
        # load that Solution carries in step with the routes
        delta = or_opt_delta(inst, sol, from_route, start, length, to_route, to_pos, reverse)
        sol.apply_or_opt(inst, from_route, start, length, to_route, to_pos, reverse, delta)

    return reversals


class GvnsStats:
    def __init__(self, n_actions):
        self.iterations = 0
        self.accepted = 0
        self.infeasible = 0
        self.reversals = 0
        self.vnd_steps = 0
        self.calls = [0] * n_actions
        self.improvements = [0] * n_actions

    def absorb(self, vnd_stats):
        self.vnd_steps += vnd_stats.steps
        for a, (c, i) in enumerate(zip(vnd_stats.calls, vnd_stats.improvements)):
            self.calls[a] += c
            self.improvements[a] += i


def gvns(inst, selector, seed, budget_seconds=None, max_iterations=None,
         k_min=1, k_step=1, k_max=12, neighborhoods=DEFAULT_NEIGHBORHOODS,
         e=SHAKE_E, reverse_p=0.0, reward=improvement_reward):
    # Two stopping modes, exactly one of them at a time.
    #
    # budget_seconds is the mode to compare arms under. Arms differ in what an
    # iteration costs -- a fixed order that keeps hitting the cheap neighborhoods
    # fits far more iterations into the same wall clock -- so equal time is the
    # only setting where they are charged for what they actually spend.
    #
    # max_iterations is deterministic for a given seed, since nothing then depends
    # on how loaded the machine is. It is for regression checks and for results a
    # reader has to be able to reproduce. Tables produced this way must NOT be used
    # to argue that one arm beats another: handing every arm the same iteration
    # count silently subsidises whichever one has the cheaper iteration.
    if (budget_seconds is None) == (max_iterations is None):
        raise ValueError("pass exactly one of budget_seconds or max_iterations")

    rng = random.Random(seed)
    start = time.time()
    stats = GvnsStats(len(neighborhoods))

    if max_iterations is None:
        progress = elapsed_progress(start, budget_seconds)
        keep_going = lambda: time.time() - start < budget_seconds
    else:
        progress = lambda: min(1.0, stats.iterations / max_iterations)
        keep_going = lambda: stats.iterations < max_iterations

    # a selector that schedules anything over the run reads the same clock the phase
    # feature does; the plain selectors carry no schedule and do not define this
    set_progress = getattr(selector, "set_progress", None)
    if set_progress is not None:
        set_progress(progress)

    best = Solution(
        initial_solution_nearest_neighbor(
            inst["coords"], inst["demands"], inst["capacity"], inst["depot_id"], inst["dist"]
        ),
        inst,
    )
    stats.absorb(vnd(inst, best, neighborhoods, selector, progress=progress, reward=reward))

    k = k_min
    while keep_going():
        candidate = best.copy()
        stats.reversals += shake(inst, candidate, k, e, rng, reverse_p)
        stats.absorb(vnd(inst, candidate, neighborhoods, selector, progress=progress,
                         reward=reward))
        stats.iterations += 1

        # shake and the neighborhoods both respect capacity, so this should always
        # hold; it stays as a guard because accepting an infeasible best would
        # silently poison every iteration after it
        feasible = is_feasible(candidate.routes, inst["demands"], inst["capacity"])
        if not feasible:
            stats.infeasible += 1

        if feasible and candidate.cost < best.cost - EPS:
            best = candidate
            stats.accepted += 1
            k = k_min
        else:
            k += k_step
            if k > k_max:
                k = k_min

    return best, stats
