# Q-learning for neighborhood selection in VND

A GVNS solver for the Capacitated Vehicle Routing Problem where the VND step
picks its next neighborhood through a Q-learning agent instead of a fixed
order. Written from scratch in Python; NumPy is the only dependency.

## Setup

Unless noted: Augerat A-n62-k8 / A-n65-k9 / A-n80-k10, 10 seeds, 20 s
wall-clock per run, arms `fixed` / `random` / `q`, paired by seed. Raw data in
`results/`, one CSV row per run plus a JSON sidecar with the shared
configuration.

The agent's state is `(search phase, previous attempt improved)` — 6 states
over 5 actions. Hyperparameters α = 0.1, γ = 0.9, ε decaying 0.9 → 0.05 over
the time budget. Three reward formulations were run on identical seeds,
instances and budget, so the reward is the only variable between them:
`results_v5` per-attempt improvement, `results_v6` improvement per unit time,
`results_v7` marginal improvement (credit scaled by how many neighborhoods had
already failed on the current solution).

## Results

**Learning does not beat random selection.** q vs random across the three
rewards: t = -0.58, +0.55, +0.90 — never significant, and the sign flips.
Both stochastic arms do beat fixed order decisively (q vs fixed t = -5.12 to
-5.58, random vs fixed t = -4.41 to -4.96 across v5-v7), so the gain comes
from diversification, not from learning. The random arm was kept as a
permanent baseline for exactly this reason: measured against fixed order
alone, the diversification win would read as an ML win.

**The agent learns what it is paid for; quality does not follow.** Each reward
produced a stable, seed-consistent policy that tracked it closely —
per-attempt → two_opt 29.9%; per-second → two_opt 38.2% and 43% more
iterations; marginal → cross_exchange 35.1% and iterations down to 203.
Quality moved with none of them. two_opt and cross_exchange yield nearly the
same gain per success (8.93 vs 9.07, random arm in `results_v7.csv`) at 72x
different cost per call (0.086 vs 6.15 ms), but two_opt's gains are
redundant: relocate and swap would have found them anyway. The first two
rewards credit realized rather than marginal contribution. The third credits
marginal contribution, but through `n_exhausted`, which the state does not
contain — see Open.

**Only cross_exchange adds anything on top of relocate+swap.** Ablation over
cumulative subsets, random selector (`results/ablation_*.csv`): +two_opt
t = +0.02, +or_opt t = +0.83, +cross_exchange t = -3.90 (21/0/9).
cross_exchange wins despite cutting GVNS iterations roughly threefold. What
helps is a non-redundant neighborhood, not a larger set.

**Capacity-aware shaking was the largest effect in the project.** Shake
applied its moves without checking capacity, so 90.8 / 90.9 / 94.1% of GVNS
iterations produced an infeasible solution and were discarded at acceptance.
Restricting the insertion target to routes with room cut that to zero and took
the A-n80-k10 mean gap from 6.02% to 1.72% on the q arm (6.66% → 2.06%
averaged over arms) — 2-4x larger than any difference between selectors.

A-n32-k5 and A-n33-k5 reach the known optimum in all 30 runs at a 10 s budget
and give no comparative signal (`results_gvns.csv`).

## Running

    pip install -r requirements.txt pytest

    python3 -m qvnd.main A-n62-k8 A-n65-k9 --seeds 10 --budget 20 --out run.csv

`--reward` selects the reward function: `improvement` (default, v5),
`improvement_per_second` (v6), `marginal_improvement` (v7). The ablation runs
restrict both the arm and the neighborhood set:

    python3 -m qvnd.main A-n62-k8 --arms random --neighborhoods relocate swap --out run.csv

Runs are bounded by wall-clock time (`--budget`, used for every comparison
here) or by iteration count (`--max-iterations`), which is deterministic per
seed but not fair across arms, since arms differ in cost per iteration.

## Layout

    qvnd/
      data.py            .vrp parsing (CVRPLIB), distance matrix
      solution.py        routes, cost/load bookkeeping, feasibility
      local_search.py    relocate, swap, two_opt, or_opt, cross_exchange
      vnd.py             VND loop, fixed/random selectors, reward functions
      qlearning.py       Q-table agent, epsilon-greedy
      gvns.py            capacity-aware shaking, acceptance, outer loop
      experiments.py     batch runs, CSV/JSON output, paired statistics
      main.py            CLI
      instances/         Augerat A set (27 instances)
    results/             raw data behind every number above
    test_solution.py     99 tests

Neighborhoods are first-improvement with incremental (delta-cost) evaluation;
the delta is checked against a full recomputation over a thousand random moves
per neighborhood. The VND loop tracks failed neighborhoods in an `exhausted`
set, cleared on any improvement, so it stops only at a local optimum with
respect to all five at once.

## Tests

    python3 -m pytest

## Open

Expanding the agent's state with `n_exhausted` — the marginal reward is
defined by it, but the state is not, so the agent is paid on something it
cannot see and can only learn "more cross_exchange, always". A failure penalty
was never charged in any run; it stays at 0.0 and is not wired through
`gvns`/`experiments`. The q arm has the lower coefficient of variation in 8 of
9 instance-runs, which would need 30-50 seeds to claim.