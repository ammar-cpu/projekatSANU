import argparse
from pathlib import Path

from .experiments import (
    ARMS,
    DEFAULT_AGENT,
    DEFAULT_SEARCH,
    format_comparisons,
    format_summary,
    run_batch,
    save_results_csv,
)

INSTANCE_DIR = Path(__file__).parent / "instances"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="run GVNS experiments over the Augerat A set")
    p.add_argument("instances", nargs="+",
                   help="instance names without the .vrp suffix, e.g. A-n65-k9")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seeds", type=int, default=10, help="number of seeds, 0..n-1")
    stop = p.add_mutually_exclusive_group()
    stop.add_argument("--budget", type=float,
                      help="seconds per run; the mode to compare arms under")
    stop.add_argument("--max-iterations", type=int,
                      help="GVNS iterations per run; deterministic for a given seed, "
                           "for regression checks and reproducible tables. Not for "
                           "arm-vs-arm claims: arms differ in iteration cost.")
    p.add_argument("--out", default="results.csv")

    p.add_argument("--k-max", type=int, default=DEFAULT_SEARCH["k_max"])
    p.add_argument("--e", type=int, default=DEFAULT_SEARCH["e"])
    p.add_argument("--reverse-p", type=float, default=DEFAULT_SEARCH["reverse_p"])
    p.add_argument("--reward", default=DEFAULT_SEARCH["reward"])

    p.add_argument("--alpha", type=float, default=DEFAULT_AGENT["alpha"])
    p.add_argument("--gamma", type=float, default=DEFAULT_AGENT["gamma"])
    p.add_argument("--eps-start", type=float, default=DEFAULT_AGENT["eps_start"])
    p.add_argument("--eps-end", type=float, default=DEFAULT_AGENT["eps_end"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = [INSTANCE_DIR / f"{name}.vrp" for name in args.instances]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"no such instance file: {', '.join(missing)}")

    search = {"k_max": args.k_max, "e": args.e,
              "reverse_p": args.reverse_p, "reward": args.reward}
    agent = {"alpha": args.alpha, "gamma": args.gamma,
             "eps_start": args.eps_start, "eps_end": args.eps_end}

    budget = args.budget
    if budget is None and args.max_iterations is None:
        budget = 20.0  # timed mode is the default because it is the comparable one

    rows = run_batch(paths, args.arms, range(args.seeds), budget, args.max_iterations,
                     search, agent)
    written = save_results_csv(rows, args.out)

    print()
    print(format_summary(rows))
    print()
    if rows[0]["mode"] == "iterations":
        print("mode=iterations: reproducible, but arms are not comparable here "
              "(equal iterations favour whichever arm has the cheaper iteration)")
    print(format_comparisons(rows))
    print(f"\nwritten to {written} and {written.with_suffix('.json')}")


if __name__ == "__main__":
    main()
