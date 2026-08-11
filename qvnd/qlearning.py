import random

import numpy as np

N_PHASES = 3      # early / mid / late
N_IMPROVED = 2    
N_STATES = N_PHASES * N_IMPROVED


class QAgent:
    # tabular Q-learning. One instance lives for a whole run and is shared across
    # every vnd() call in it -- rebuilding it per call throws the learning away.

    def __init__(self, n_actions, alpha, gamma, eps_start, eps_end, seed=None):
        self.q = np.zeros((N_STATES, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.steps = 0
        self.rng = random.Random(seed)
        self._progress = lambda: 0.0

    def set_progress(self, progress):
        # epsilon decays on the budget, not on a step count: steps differ between
        # arms by up to 2x, which would decay the schedules at different rates
        self._progress = progress

    @property
    def epsilon(self):
        fraction = min(1.0, max(0.0, self._progress()))
        return self.eps_start + (self.eps_end - self.eps_start) * fraction

    def get_state(self, phase, improved_prev):
        return phase * N_IMPROVED + improved_prev

    def pick(self, state, available):
        # both branches draw from `available` only. Taking the argmax over the whole
        # row would eventually return a neighborhood the caller has already parked.
        if self.rng.random() < self.epsilon:
            return self.rng.choice(available)

        row = self.q[self.get_state(*state)]
        best = max(row[a] for a in available)
        # ties are broken at random: the table starts at zero, so a first-index rule
        # would make the greedy branch a copy of the fixed order until it fills in
        return self.rng.choice([a for a in available if row[a] == best])

    def update(self, state, action, reward, next_state):
        # max over the full next row, not over what happens to be available next:
        # the available set is a property of the search, not of the state
        current = self.get_state(*state)
        target = reward + self.gamma * self.q[self.get_state(*next_state)].max()
        self.q[current, action] += self.alpha * (target - self.q[current, action])
        self.steps += 1
