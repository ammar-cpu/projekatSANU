class QAgent:
    # isolated Q-learning module: NumPy Q-table |S| x |A|, no neural networks

    def __init__(self, n_states, n_actions, alpha, gamma, eps_start, eps_end):
        pass

    def get_state(self, phase, improved_prev):
        # (phase, improved_prev) -> state index; phase = early/mid/late
        pass

    def pick(self, state):
        # epsilon-greedy action selection (neighborhood index)
        pass

    def update(self, state, action, reward, next_state):
        # Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        pass

    def decay_epsilon(self, step, total_steps):
        # linear decay eps_start -> eps_end
        pass
