import random
import time

from .local_search import EPS, or_opt_delta
from .solution import Solution, initial_solution_nearest_neighbor, is_feasible
from .vnd import DEFAULT_NEIGHBORHOODS, elapsed_progress, improvement_reward, vnd

SHAKE_E = 5  # segment lengths run from 2 to SHAKE_E + 1


def shake(inst, sol, k, e, rng, reverse_p=0.0):
    # k random segment moves, ignoring cost but respecting capacity.
    # reverse_p: chance a move reverses the segment in place instead of moving it,
    # a weaker kick that shifts no load. Returns how many went that way.
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
            # source route is unconditional: its load already counts the segment, so
            # the usual test would double-count it. Also keeps the list non-empty.
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

        # delta is computed only to keep Solution's cached cost and load in step
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
    # Exactly one stopping mode.
    # budget_seconds: for comparing arms, since they differ in iteration cost.
    # max_iterations: deterministic per seed, for regression checks and reproducible
    # tables. Not for arm-vs-arm claims -- it subsidises the cheaper iteration.
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

    # selectors that schedule over the run read the same clock the phase does
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

        # always true now that shake respects capacity; kept because an infeasible
        # best would poison every iteration after it
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
