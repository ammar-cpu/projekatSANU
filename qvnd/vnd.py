import random
import time

from .local_search import cross_exchange, or_opt, relocate, swap, two_opt

DEFAULT_NEIGHBORHOODS = (relocate, swap, two_opt, or_opt, cross_exchange)

EARLY, MID, LATE = 0, 1, 2


def phase_of(progress):
    # early/mid/late thirds of the run. `progress` is the fraction of the overall
    # budget consumed and comes from outside: a single VND call is far too short
    # to have phases of its own.
    if progress < 1 / 3:
        return EARLY
    if progress < 2 / 3:
        return MID
    return LATE


def _progress_reader(progress):
    # a plain number freezes the phase for the whole call, which is what standalone
    # runs and tests want. A zero-argument callable is read again on every iteration,
    # so a caller holding the wall clock can let the phase move mid-call.
    if callable(progress):
        return progress
    return lambda: progress


def elapsed_progress(start, budget):
    # the callable GVNS passes in: fraction of the time budget consumed so far
    return lambda: min(1.0, (time.time() - start) / budget) if budget > 0 else 1.0


def improvement_reward(gain, cost_before, improved, failure_penalty):
    # relative drop in cost, so the scale does not depend on the instance
    if not improved:
        return failure_penalty
    return gain / cost_before if cost_before > 0 else 0.0


REWARDS = {"improvement": improvement_reward}


class VndStats:
    # per-neighborhood bookkeeping. Different orders tend to land on the same local
    # optimum, so the work spent getting there is what separates them, not the cost.
    def __init__(self, count):
        self.calls = [0] * count
        self.improvements = [0] * count
        self.steps = 0
        self.gain = 0.0

    def record(self, action, improved, gain):
        self.steps += 1
        self.calls[action] += 1
        if improved:
            self.improvements[action] += 1
            self.gain += gain


class FixedSelector:
    # baseline: the lowest-numbered neighborhood still worth trying. Together with
    # the reset in vnd() this is textbook VND -- start at the first, return to it
    # after every improvement, move on only once one fails.
    def pick(self, state, available):
        return min(available)

    def update(self, state, action, reward, next_state):
        pass


class RandomSelector:
    # uniform choice among the neighborhoods still eligible. This is the control
    # arm: a Q-agent that cannot beat an unbiased choice has learned nothing.
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def pick(self, state, available):
        return self.rng.choice(available)

    def update(self, state, action, reward, next_state):
        pass


def vnd(inst, sol, neighborhoods, selector, progress=0.0, failure_penalty=0.0,
        reward=improvement_reward):
    # `selector` decides which neighborhood to try next. FixedSelector reproduces the
    # baseline and a Q-agent plugs into the same interface, keeping the comparison fair.
    stats = VndStats(len(neighborhoods))
    read_progress = _progress_reader(progress)
    improved_previously = 0

    # a neighborhood that failed on the current solution fails again until the
    # solution changes, so it is parked until the next improvement. This is what
    # makes the loop terminate whatever the selector does.
    exhausted = set()

    while len(exhausted) < len(neighborhoods):
        available = [a for a in range(len(neighborhoods)) if a not in exhausted]
        state = (phase_of(read_progress()), improved_previously)

        action = selector.pick(state, available)
        if action not in available:
            raise ValueError(
                f"selector picked neighborhood {action}, which is not among {available}"
            )

        before = sol.cost
        improved = neighborhoods[action](inst, sol)
        gain = before - sol.cost

        if improved:
            exhausted.clear()
        else:
            exhausted.add(action)

        value = reward(gain, before, improved, failure_penalty)
        improved_previously = 1 if improved else 0
        # read again after the move: at a phase boundary the agent transitions into
        # the new phase, and that is the state the Q-update has to bootstrap from
        next_state = (phase_of(read_progress()), improved_previously)
        selector.update(state, action, value, next_state)
        stats.record(action, improved, gain)

    return stats
