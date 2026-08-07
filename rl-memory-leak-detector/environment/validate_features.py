#!/usr/bin/env python3
"""
validate_features.py - Day 9: validate feature extraction against known trace patterns

Runs feature_extraction.py against every trace in a dataset (per manifest.csv)
and checks that the extracted features behave the way each pattern SHOULD:

    clean  - unfreed_ratio decays toward ~0, no leaked addresses,
             relative_lifespan for all addresses stays near ~1
    leaky  - unfreed_ratio stabilizes at a nonzero level (~leak rate),
             leaked addresses have relative_lifespan >> 1 (outliers)
    spiky  - unfreed_ratio spikes early then decays (false-positive trap),
             but NO addresses actually leak, and relative_lifespan for
             addresses stays near ~1 (nothing is truly anomalous)

This is a sanity check on the feature pipeline itself, run BEFORE any RL
code touches these features -- if this fails, the environment/reward
function built on top of it can't be trusted either.

Usage:
    python3 validate_features.py --dataset-dir traces/dataset
"""

import argparse
import csv
import os
import sys

from feature_extraction import parse_trace, extract_features


def load_manifest(dataset_dir):
    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    with open(manifest_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def analyze_trace(csv_path, sample_interval=1.0, rate_window=2.0):
    events = parse_trace(csv_path)
    snapshots, ground_truth = extract_features(
        events, sample_interval=sample_interval, rate_window=rate_window
    )
    if not snapshots:
        return None

    leaked_addrs = {a for a, g in ground_truth.items() if g["freed_at"] is None}
    n_addrs = len(ground_truth)
    leak_rate = len(leaked_addrs) / n_addrs if n_addrs else 0.0

    max_t = snapshots[-1]["snapshot_time"]
    early = [s["unfreed_ratio"] for s in snapshots if s["snapshot_time"] <= max_t * 0.2]
    late = [s["unfreed_ratio"] for s in snapshots if s["snapshot_time"] >= max_t * 0.8]
    early_ratio = sum(early) / len(early) if early else 0.0
    late_ratio = sum(late) / len(late) if late else 0.0

    # last known relative_lifespan per address, split by leaked vs healthy
    last_snap = {}
    for s in snapshots:
        last_snap[s["address"]] = s
    leaked_rel = [last_snap[a]["relative_lifespan"] for a in leaked_addrs
                  if a in last_snap and last_snap[a]["relative_lifespan"] is not None]
    healthy_rel = [last_snap[a]["relative_lifespan"] for a, s in last_snap.items()
                   if a not in leaked_addrs and s["relative_lifespan"] is not None]
    avg_leaked_rel = sum(leaked_rel) / len(leaked_rel) if leaked_rel else None
    avg_healthy_rel = sum(healthy_rel) / len(healthy_rel) if healthy_rel else None

    return {
        "n_addrs": n_addrs,
        "leak_rate": leak_rate,
        "early_unfreed_ratio": early_ratio,
        "late_unfreed_ratio": late_ratio,
        "avg_leaked_relative_lifespan": avg_leaked_rel,
        "avg_healthy_relative_lifespan": avg_healthy_rel,
    }


def run_validation(dataset_dir, sample_interval, rate_window):
    manifest = load_manifest(dataset_dir)
    by_pattern = {}
    for row in manifest:
        pattern = row["pattern"]
        by_pattern.setdefault(pattern, []).append(row["filename"])

    results = {}
    for pattern, filenames in by_pattern.items():
        agg = []
        for fname in filenames:
            path = os.path.join(dataset_dir, fname)
            r = analyze_trace(path, sample_interval, rate_window)
            if r:
                agg.append(r)
        results[pattern] = agg

    print(f"{'pattern':<12} {'leak_rate':>10} {'unfreed(early->late)':>22} "
          f"{'leaked_rel_lifespan':>20} {'healthy_rel_lifespan':>20}")
    print("-" * 90)
    for pattern, agg in results.items():
        if not agg:
            continue
        n = len(agg)
        avg_leak_rate = sum(r["leak_rate"] for r in agg) / n
        avg_early = sum(r["early_unfreed_ratio"] for r in agg) / n
        avg_late = sum(r["late_unfreed_ratio"] for r in agg) / n
        leaked_vals = [r["avg_leaked_relative_lifespan"] for r in agg
                       if r["avg_leaked_relative_lifespan"] is not None]
        healthy_vals = [r["avg_healthy_relative_lifespan"] for r in agg
                        if r["avg_healthy_relative_lifespan"] is not None]
        avg_leaked_rel = sum(leaked_vals) / len(leaked_vals) if leaked_vals else None
        avg_healthy_rel = sum(healthy_vals) / len(healthy_vals) if healthy_vals else None

        leaked_str = f"{avg_leaked_rel:.2f}" if avg_leaked_rel is not None else "n/a"
        healthy_str = f"{avg_healthy_rel:.2f}" if avg_healthy_rel is not None else "n/a"
        print(f"{pattern:<12} {avg_leak_rate:>9.1%} "
              f"{avg_early:>10.3f} -> {avg_late:<9.3f} "
              f"{leaked_str:>20} {healthy_str:>20}")

    print()
    return check_expectations(results)


def check_expectations(results):
    """Assert-style checks. Returns True if all patterns behave as expected."""
    all_ok = True

    def check(condition, message):
        nonlocal all_ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_ok = False
        print(f"[{status}] {message}")

    if "clean" in results and results["clean"]:
        agg = results["clean"]
        avg_leak_rate = sum(r["leak_rate"] for r in agg) / len(agg)
        avg_late = sum(r["late_unfreed_ratio"] for r in agg) / len(agg)
        check(avg_leak_rate < 0.05, f"clean: leak_rate near 0 (got {avg_leak_rate:.1%})")
        check(avg_late < 0.15, f"clean: late unfreed_ratio decays low (got {avg_late:.3f})")

    if "leaky" in results and results["leaky"]:
        agg = results["leaky"]
        avg_leak_rate = sum(r["leak_rate"] for r in agg) / len(agg)
        avg_late = sum(r["late_unfreed_ratio"] for r in agg) / len(agg)
        leaked_vals = [r["avg_leaked_relative_lifespan"] for r in agg
                       if r["avg_leaked_relative_lifespan"] is not None]
        healthy_vals = [r["avg_healthy_relative_lifespan"] for r in agg
                        if r["avg_healthy_relative_lifespan"] is not None]
        avg_leaked_rel = sum(leaked_vals) / len(leaked_vals) if leaked_vals else 0
        avg_healthy_rel = sum(healthy_vals) / len(healthy_vals) if healthy_vals else 0

        check(0.15 < avg_leak_rate < 0.55,
              f"leaky: leak_rate in plausible range (got {avg_leak_rate:.1%})")
        check(avg_late > 0.15,
              f"leaky: unfreed_ratio stabilizes ABOVE zero (got {avg_late:.3f})")
        check(avg_leaked_rel > avg_healthy_rel * 3,
              f"leaky: leaked addresses have much higher relative_lifespan "
              f"({avg_leaked_rel:.2f} vs healthy {avg_healthy_rel:.2f})")

    if "spiky" in results and results["spiky"]:
        agg = results["spiky"]
        avg_leak_rate = sum(r["leak_rate"] for r in agg) / len(agg)
        avg_early = sum(r["early_unfreed_ratio"] for r in agg) / len(agg)
        avg_late = sum(r["late_unfreed_ratio"] for r in agg) / len(agg)
        healthy_vals = [r["avg_healthy_relative_lifespan"] for r in agg
                        if r["avg_healthy_relative_lifespan"] is not None]
        avg_healthy_rel = sum(healthy_vals) / len(healthy_vals) if healthy_vals else 0

        check(avg_leak_rate < 0.05, f"spiky: leak_rate near 0 (got {avg_leak_rate:.1%})")
        check(avg_early > avg_late,
              f"spiky: unfreed_ratio is highest early, decays later "
              f"({avg_early:.3f} -> {avg_late:.3f}) -- the false-positive trap")
        check(avg_healthy_rel < 3.0,
              f"spiky: relative_lifespan stays near-normal despite high ratio "
              f"(got {avg_healthy_rel:.2f})")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- review output above")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate feature extraction against known patterns")
    parser.add_argument("--dataset-dir", type=str, default="traces/dataset")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--rate-window", type=float, default=2.0)
    args = parser.parse_args()

    ok = run_validation(args.dataset_dir, args.sample_interval, args.rate_window)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()