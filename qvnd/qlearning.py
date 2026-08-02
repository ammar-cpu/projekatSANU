class QAgent:
    # tabular Q-learning: a NumPy Q-table of shape |S| x |A|

    def __init__(self, n_states, n_actions, alpha, gamma, eps_start, eps_end):
        pass

    def get_state(self, phase, improved_prev):
        # (phase, improved_prev) -> row index; phase is early/mid/late
        pass

    def pick(self, state, available):
        # epsilon-greedy over `available` only; the greedy branch has to mask the
        # Q-row, otherwise it can return a neighborhood vnd() has parked
        pass

    def update(self, state, action, reward, next_state):
        # Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        pass

    def decay_epsilon(self, step, total_steps):
        # linear decay from eps_start to eps_end
        pass
