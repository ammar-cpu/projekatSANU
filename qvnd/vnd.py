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


def improvement_reward(gain, cost_before, improved, elapsed_ms, failure_penalty):
    # relative drop in cost, so the scale does not depend on the instance
    if not improved:
        return failure_penalty
    return gain / cost_before if cost_before > 0 else 0.0


REFERENCE_CALL_MS = 0.02  # measured median neighborhood call across instances
FLOOR_CALL_MS = 0.001     # clock-resolution guard, below the fastest observed call


def improvement_per_second(gain, cost_before, improved, elapsed_ms, failure_penalty):
    # Under a fixed time budget the currency is improvement per unit time, so paying
    # per attempt misprices the actions: a scan costs anywhere from 0.001 to 3.4 ms,
    # ~29x between the cheapest and dearest neighborhood at the median.
    # Scaled by the reference call so a median-cost attempt scores exactly what
    # improvement_reward would, keeping Q values in the range earlier runs used.
    scale = REFERENCE_CALL_MS / max(elapsed_ms, FLOOR_CALL_MS)
    if not improved:
        # a costly dead end wastes more of the budget than a cheap one
        return failure_penalty / scale
    return (gain / cost_before if cost_before > 0 else 0.0) * scale


REWARDS = {
    "improvement": improvement_reward,
    "improvement_per_second": improvement_per_second,
}


class VndStats:
    # per-neighborhood bookkeeping: orders mostly reach the same local optimum,
    # so what separates them is the work spent getting there
    def __init__(self, count):
        self.calls = [0] * count
        self.improvements = [0] * count
        # gain and time per neighborhood, to price a success against what it costs
        self.gains = [0.0] * count
        self.millis = [0.0] * count
        self.steps = 0
        self.gain = 0.0

    def record(self, action, improved, gain, elapsed_ms):
        self.steps += 1
        self.calls[action] += 1
        self.millis[action] += elapsed_ms
        if improved:
            self.improvements[action] += 1
            self.gains[action] += gain
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
        started = time.perf_counter()
        improved = neighborhoods[action](inst, sol)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        gain = before - sol.cost

        if improved:
            exhausted.clear()
        else:
            exhausted.add(action)

        value = reward(gain, before, improved, elapsed_ms, failure_penalty)
        improved_previously = 1 if improved else 0
        # re-read: at a phase boundary this is the state the Q-update bootstraps from
        next_state = (phase_of(read_progress()), improved_previously)
        selector.update(state, action, value, next_state)
        stats.record(action, improved, gain, elapsed_ms)

    return stats
