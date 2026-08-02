def run_single(instance_path, algorithm, seed, time_budget):
    # one run (algorithm x instance x seed), returns best/mean/std/gap/CV + neighborhood usage stats
    pass


def run_batch(instance_paths, algorithms, seeds, time_budget):
    # repeats run_single for all instance/algorithm/seed combinations
    pass


def save_results_csv(results, out_path):
    # writes results to CSV (one row per run)
    pass
