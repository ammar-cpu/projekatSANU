import csv
import json
import statistics
import time
from pathlib import Path

from .data import parse_vrp
from .gvns import SHAKE_E, gvns
from .qlearning import QAgent
from .solution import is_feasible
from .vnd import DEFAULT_NEIGHBORHOODS, REWARDS, FixedSelector, RandomSelector

ARMS = ("fixed", "random", "q")

# kept apart from the search knobs so a sweep can vary one without the other
DEFAULT_AGENT = {"alpha": 0.1, "gamma": 0.9, "eps_start": 0.9, "eps_end": 0.05}

DEFAULT_SEARCH = {"k_min": 1, "k_step": 1, "k_max": 12, "e": SHAKE_E,
                  "reverse_p": 0.0, "reward": "improvement"}


def make_selector(arm, seed, n_actions=len(DEFAULT_NEIGHBORHOODS), agent=None):
    if arm == "fixed":
        return FixedSelector()
    if arm == "random":
        return RandomSelector(seed)
    if arm == "q":
        return QAgent(n_actions, seed=seed, **(agent or DEFAULT_AGENT))
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def run_single(instance_path, arm, seed, budget_seconds=None, max_iterations=None,
               search=None, agent=None, neighborhoods=DEFAULT_NEIGHBORHOODS):
    # one (instance, arm, seed) run. The row carries its own configuration so a
    # stored CSV explains itself. Only the timed mode supports arm-vs-arm claims.
    search = {**DEFAULT_SEARCH, **(search or {})}
    agent = {**DEFAULT_AGENT, **(agent or {})}

    inst = parse_vrp(instance_path)
    selector = make_selector(arm, seed, len(neighborhoods), agent)

    started = time.perf_counter()
    best, stats = gvns(
        inst, selector, seed,
        budget_seconds=budget_seconds, max_iterations=max_iterations,
        k_min=search["k_min"], k_step=search["k_step"], k_max=search["k_max"],
        neighborhoods=neighborhoods, e=search["e"], reverse_p=search["reverse_p"],
        reward=REWARDS[search["reward"]],
    )
    wall = time.perf_counter() - started

    if not is_feasible(best.routes, inst["demands"], inst["capacity"]):
        raise AssertionError(f"{inst['name']} {arm} seed {seed}: infeasible result")
    true_cost, true_load = best.recompute(inst)
    if abs(best.cost - true_cost) > 1e-6 or best.load != true_load:
        raise AssertionError(f"{inst['name']} {arm} seed {seed}: cached state drifted")

    optimum = inst["optimum"]
    return {
        "instance": inst["name"], "n": inst["dimension"], "optimum": optimum,
        "arm": arm, "seed": seed,
        "cost": best.cost, "gap": 100 * (best.cost - optimum) / optimum,
        "routes": sum(1 for r in best.routes if r), "wall": round(wall, 3),
        "iterations": stats.iterations, "accepted": stats.accepted,
        "infeasible": stats.infeasible, "reversals": stats.reversals,
        "vnd_steps": stats.vnd_steps,
        "epsilon_end": getattr(selector, "epsilon", ""),
        **{f"calls_{f.__name__}": c for f, c in zip(neighborhoods, stats.calls)},
        **{f"impr_{f.__name__}": c for f, c in zip(neighborhoods, stats.improvements)},
        "mode": "iterations" if max_iterations is not None else "time",
        "time_budget": budget_seconds if budget_seconds is not None else "",
        "max_iterations": max_iterations if max_iterations is not None else "",
        **{f"cfg_{k}": v for k, v in search.items()},
        **{f"cfg_{k}": v for k, v in agent.items()},
    }


def run_batch(instance_paths, arms, seeds, budget_seconds=None, max_iterations=None,
              search=None, agent=None, neighborhoods=DEFAULT_NEIGHBORHOODS, log=print):
    rows = []
    for path in instance_paths:
        for arm in arms:
            for seed in seeds:
                row = run_single(path, arm, seed, budget_seconds, max_iterations,
                                 search, agent, neighborhoods)
                rows.append(row)
                if log:
                    # flushed: a redirected stdout shows nothing for tens of minutes
                    log(f"{row['instance']:11} {arm:6} seed={seed:<3} "
                        f"cost={row['cost']:.0f} gap={row['gap']:5.2f}% "
                        f"iter={row['iterations']} acc={row['accepted']}", flush=True)
    return rows


def save_results_csv(rows, path):
    # one row per run in the CSV, what they all share in a sidecar JSON
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shared = {k: v for k, v in rows[0].items() if k.startswith("cfg_")}
    meta = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": len(rows),
        "instances": sorted({r["instance"] for r in rows}),
        "arms": sorted({r["arm"] for r in rows}),
        "seeds": sorted({r["seed"] for r in rows}),
        "mode": rows[0]["mode"],
        "time_budget": rows[0]["time_budget"],
        "max_iterations": rows[0]["max_iterations"],
        "comparable_across_arms": rows[0]["mode"] == "time",
        "neighborhoods": [k[len("calls_"):] for k in rows[0] if k.startswith("calls_")],
        "config": shared,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def summarize(rows):
    # one line per (instance, arm)
    out = []
    for instance in sorted({r["instance"] for r in rows}):
        for arm in [a for a in ARMS if any(r["arm"] == a for r in rows)]:
            costs = [r["cost"] for r in rows
                     if r["instance"] == instance and r["arm"] == arm]
            if not costs:
                continue
            optimum = next(r["optimum"] for r in rows if r["instance"] == instance)
            mean = statistics.mean(costs)
            std = statistics.stdev(costs) if len(costs) > 1 else 0.0
            out.append({
                "instance": instance, "arm": arm, "runs": len(costs),
                "best": min(costs), "mean": mean, "std": std,
                "gap_best": 100 * (min(costs) - optimum) / optimum,
                "gap_mean": 100 * (mean - optimum) / optimum,
                "cv": 100 * std / mean if mean else 0.0,
            })
    return out


def paired_comparison(rows, first, second, instance=None):
    # pairing by seed removes the run-to-run spread that swamps an unpaired diff
    costs = {}
    for r in rows:
        if instance is None or r["instance"] == instance:
            costs.setdefault((r["instance"], r["seed"]), {})[r["arm"]] = r["cost"]

    diffs = [c[first] - c[second] for c in costs.values() if first in c and second in c]
    if not diffs:
        return None

    mean = statistics.mean(diffs)
    std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return {
        "first": first, "second": second, "instance": instance or "all",
        "n": len(diffs), "mean_diff": mean, "std_diff": std,
        "t": mean / (std / len(diffs) ** 0.5) if std else float("nan"),
        "wins": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "losses": sum(d > 0 for d in diffs),
    }


def format_summary(rows):
    lines = [f"{'instance':11} {'arm':7} {'best':>7} {'mean':>9} {'std':>6} "
             f"{'gap_best':>9} {'gap_mean':>9} {'CV':>7}", "-" * 70]
    previous = None
    for s in summarize(rows):
        if previous and s["instance"] != previous:
            lines.append("-" * 70)
        lines.append(f"{s['instance']:11} {s['arm']:7} {s['best']:7.0f} {s['mean']:9.1f} "
                     f"{s['std']:6.2f} {s['gap_best']:8.2f}% {s['gap_mean']:8.2f}% "
                     f"{s['cv']:6.2f}%")
        previous = s["instance"]
    return "\n".join(lines)


def format_comparisons(rows, pairs=(("q", "fixed"), ("random", "fixed"), ("q", "random"))):
    lines = ["paired by seed (negative = first arm better)"]
    for instance in sorted({r["instance"] for r in rows}) + [None]:
        lines.append(f"\n{instance or 'ALL INSTANCES'}")
        for first, second in pairs:
            c = paired_comparison(rows, first, second, instance)
            if c:
                lines.append(f"   {first:7} vs {second:7} diff={c['mean_diff']:+8.2f} "
                             f"t={c['t']:+6.2f}  {c['wins']}/{c['ties']}/{c['losses']} "
                             f"(n={c['n']})")
    return "\n".join(lines)
