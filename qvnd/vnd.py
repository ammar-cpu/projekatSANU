import random
import time

from .local_search import cross_exchange, or_opt, relocate, swap, two_opt

DEFAULT_NEIGHBORHOODS = (relocate, swap, two_opt, or_opt, cross_exchange)

EARLY, MID, LATE = 0, 1, 2


def phase_of(progress):
    # early/mid/late thirds of the run. progress comes from outside: one VND call
    # is far too short to have phases of its own.
    if progress < 1 / 3:
        return EARLY
    if progress < 2 / 3:
        return MID
    return LATE


def _progress_reader(progress):
    # a number freezes the phase for the call; a callable is re-read each iteration
    if callable(progress):
        return progress
    return lambda: progress


def elapsed_progress(start, budget):
    # fraction of the time budget consumed so far
    return lambda: min(1.0, (time.time() - start) / budget) if budget > 0 else 1.0


def improvement_reward(gain, cost_before, improved, failure_penalty):
    # relative drop in cost, so the scale does not depend on the instance
    if not improved:
        return failure_penalty
    return gain / cost_before if cost_before > 0 else 0.0


REWARDS = {"improvement": improvement_reward}


class VndStats:
    # per-neighborhood bookkeeping: orders mostly reach the same local optimum,
    # so what separates them is the work spent getting there
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
    # lowest-numbered neighborhood still worth trying; with the reset in vnd()
    # this is textbook VND
    def pick(self, state, available):
        return min(available)

    def update(self, state, action, reward, next_state):
        pass


class RandomSelector:
    # control arm: a Q-agent that cannot beat an unbiased choice learned nothing
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def pick(self, state, available):
        return self.rng.choice(available)

    def update(self, state, action, reward, next_state):
        pass


def vnd(inst, sol, neighborhoods, selector, progress=0.0, failure_penalty=0.0,
        reward=improvement_reward):
    # every selector sees the same interface, which is what keeps the comparison fair
    stats = VndStats(len(neighborhoods))
    read_progress = _progress_reader(progress)
    improved_previously = 0

    # a failed neighborhood fails again until the solution changes, so it is parked
    # until the next improvement. This is what makes the loop terminate.
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
        # re-read: at a phase boundary this is the state the Q-update bootstraps from
        next_state = (phase_of(read_progress()), improved_previously)
        selector.update(state, action, value, next_state)
        stats.record(action, improved, gain)

    return stats
