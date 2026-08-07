#!/usr/bin/env python3
"""
generate_dataset.py - batch-generate a full labeled synthetic trace dataset

Runs simulate_alloc.py's AllocSimulator across many seeds for each pattern,
writing all traces into an output directory plus a manifest.csv that records
which file is which pattern/seed -- so Week 3's train/held-out split has
ground truth to work from without re-deriving it from filenames.

Usage:
    python3 generate_dataset.py --out-dir traces/dataset --n-seeds 25
    python3 generate_dataset.py --out-dir traces/dataset --n-seeds 25 \
        --patterns clean leaky spiky
"""

import argparse
import csv
import os

from simulate_alloc import AllocSimulator, PATTERN_CONFIGS, write_csv


def generate_dataset(out_dir, patterns, n_seeds, duration, pid_base, base_seed):
    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []

    for pattern in patterns:
        for i in range(n_seeds):
            seed = base_seed + i
            pid = pid_base + i
            filename = f"{pattern}_{seed}.csv"
            out_path = os.path.join(out_dir, filename)

            sim = AllocSimulator(pattern, duration, pid, seed=seed)
            events = sim.run()
            rows = sim.to_rows(events)
            write_csv(rows, out_path)

            n_allocs = sum(1 for e in events if e["event_type"] == "alloc")
            n_frees = sum(1 for e in events if e["event_type"] == "free")
            manifest_rows.append({
                "filename": filename,
                "pattern": pattern,
                "seed": seed,
                "n_allocs": n_allocs,
                "n_frees": n_frees,
                "n_unfreed": n_allocs - n_frees,
            })

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "pattern", "seed", "n_allocs", "n_frees", "n_unfreed"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nGenerated {len(manifest_rows)} traces across {len(patterns)} patterns "
          f"({n_seeds} seeds each) in {out_dir}/")
    print(f"Manifest: {manifest_path}")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Batch-generate labeled synthetic traces")
    parser.add_argument("--out-dir", type=str, default="traces/dataset",
                         help="Output directory for traces + manifest (default: traces/dataset)")
    parser.add_argument("--patterns", nargs="+",
                         default=["clean", "leaky", "spiky"],
                         choices=list(PATTERN_CONFIGS),
                         help="Which patterns to generate (default: clean leaky spiky)")
    parser.add_argument("--n-seeds", type=int, default=25,
                         help="Number of seeded runs per pattern (default: 25)")
    parser.add_argument("--duration", type=float, default=30.0,
                         help="Simulated duration per trace in seconds (default: 30)")
    parser.add_argument("--pid-base", type=int, default=4021,
                         help="Base PID; each trace gets pid_base + seed_index (default: 4021)")
    parser.add_argument("--base-seed", type=int, default=0,
                         help="Starting seed value (default: 0)")
    args = parser.parse_args()

    generate_dataset(
        out_dir=args.out_dir,
        patterns=args.patterns,
        n_seeds=args.n_seeds,
        duration=args.duration,
        pid_base=args.pid_base,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()