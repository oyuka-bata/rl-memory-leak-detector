#!/usr/bin/env python3

import argparse
import csv
import os
import sys

from memory_leak_env import MemoryLeakEnv


def load_manifest(dataset_dir):
    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    with open(manifest_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def run_episode(trace_path, policy_fn, sample_interval=1.0, rate_window=2.0):
    env = MemoryLeakEnv(trace_paths=[trace_path], sample_interval=sample_interval,
                         rate_window=rate_window, shuffle_traces=False)
    obs, info = env.reset()
    total_reward = 0.0
    terminated = truncated = False
    n_steps = 0
    while not (terminated or truncated):
        action = policy_fn()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1
        if n_steps > 20000:
            print(f"WARNING: {trace_path} did not terminate after 20000 steps")
            break
    return total_reward


def run_all(dataset_dir, sample_interval, rate_window):
    manifest = load_manifest(dataset_dir)
    by_pattern = {}
    for row in manifest:
        by_pattern.setdefault(row["pattern"], []).append(row["filename"])

    results = {} 
    for pattern, filenames in by_pattern.items():
        always_totals = []
        never_totals = []
        for fname in filenames:
            path = os.path.join(dataset_dir, fname)
            always_totals.append(run_episode(path, lambda: 1, sample_interval, rate_window))
            never_totals.append(run_episode(path, lambda: 0, sample_interval, rate_window))
        results[pattern] = {"always_flag": always_totals, "never_flag": never_totals}

    return results


def summarize_and_check(results):
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"{'pattern':<12} {'always-flag avg':>18} {'never-flag avg':>18} {'n_traces':>10}")
    print("-" * 62)
    for pattern, r in results.items():
        n = len(r["always_flag"])
        print(f"{pattern:<12} {avg(r['always_flag']):>18.2f} "
              f"{avg(r['never_flag']):>18.2f} {n:>10}")
    print()

    all_ok = True

    def check(condition, message):
        nonlocal all_ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_ok = False
        print(f"[{status}] {message}")

    for pattern in ("clean", "spiky"):
        if pattern in results:
            r = results[pattern]
            af_avg = avg(r["always_flag"])
            nf_avg = avg(r["never_flag"])
            check(af_avg < 0,
                  f"{pattern}: always-flag is strongly negative "
                  f"(all false positives, got {af_avg:.2f})")
            check(abs(nf_avg) < 1e-6,
                  f"{pattern}: never-flag totals exactly 0.0 "
                  f"(no leaks exist, got {nf_avg:.2f})")

    if "leaky" in results:
        r = results["leaky"]
        af_avg = avg(r["always_flag"])
        nf_avg = avg(r["never_flag"])
        check(af_avg > 0,
              f"leaky: always-flag is strongly positive "
              f"(catches every leak + bonus, got {af_avg:.2f})")
        check(nf_avg < 0,
              f"leaky: never-flag is strongly negative "
              f"(misses every leak, got {nf_avg:.2f})")
        check(af_avg > nf_avg,
              f"leaky: always-flag clearly outperforms never-flag "
              f"({af_avg:.2f} vs {nf_avg:.2f})")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- review reward function")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Sanity-check reward totals with dummy agents")
    parser.add_argument("--dataset-dir", type=str, default="traces/dataset")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--rate-window", type=float, default=2.0)
    args = parser.parse_args()

    results = run_all(args.dataset_dir, args.sample_interval, args.rate_window)
    ok = summarize_and_check(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()