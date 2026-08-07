#!/usr/bin/env python3
"""
feature_extraction.py - convert a raw trace CSV into RL-ready feature snapshots

Standalone module (Day 8 of the project plan) -- deliberately independent of
gymnasium/stable-baselines3 so it can be unit tested and sanity-checked
against known trace patterns before any RL code touches it.

Core idea: at fixed time intervals, take a "snapshot" of every allocation
that is still live (not yet freed) and compute four features using ONLY
information available up to that point in time:

    lifespan      - seconds since this allocation happened
    size          - allocation size in bytes
    alloc_rate    - allocations/sec in a trailing window (process-wide)
    unfreed_ratio - (live allocations so far) / (total allocations so far)

This matches the state vector in the project README:
    State: [lifespan=104, size=2048, alloc_rate=3.2/s, unfreed_ratio=0.81]

Ground truth (whether an allocation ultimately leaked) is tracked SEPARATELY
and never fed into the feature vector -- the agent must not see the future.

Usage as a script:
    python3 feature_extraction.py traces/dataset/leaky_0.csv

Usage as a module:
    from feature_extraction import parse_trace, extract_features
    events = parse_trace("traces/dataset/leaky_0.csv")
    snapshots, ground_truth = extract_features(events, sample_interval=0.5)
"""

import argparse
import csv
import sys
from collections import deque


def parse_trace(csv_path):
    """Read a trace CSV and return events sorted by time, in seconds.

    Expected columns: timestamp_ns, wall_time, pid, tid, address, size, event_type
    """
    events = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "t": int(row["timestamp_ns"]) / 1e9,
                "pid": int(row["pid"]),
                "tid": int(row["tid"]),
                "address": row["address"],
                "size": int(row["size"]),
                "event_type": row["event_type"],
            })
    events.sort(key=lambda e: e["t"])
    return events


def extract_features(events, sample_interval=0.5, rate_window=2.0):
    """Walk the trace and emit periodic feature snapshots for every live allocation.

    Args:
        events: list of event dicts from parse_trace(), sorted by time.
        sample_interval: seconds between snapshots.
        rate_window: trailing window (seconds) used to compute alloc_rate.

    Returns:
        (snapshots, ground_truth)
        snapshots: list of dicts with keys
            address, snapshot_time, lifespan, size, alloc_rate, unfreed_ratio
        ground_truth: dict address -> {"alloc_time": t, "freed_at": t or None}
            None means the allocation was never freed in this trace (a leak).
    """
    if not events:
        return [], {}

    live = {}              # address -> {"alloc_time": t, "size": n}
    ground_truth = {}      # address -> {"alloc_time": t, "freed_at": t or None}
    alloc_history = deque()  # timestamps of recent allocations, for rate calc
    total_allocs_seen = 0
    freed_lifespan_sum = 0.0   # running sum of lifespans for FREED allocations only
    freed_lifespan_count = 0
    snapshots = []

    end_time = events[-1]["t"]
    next_snapshot_t = 0.0
    event_idx = 0
    n_events = len(events)

    def prune_alloc_history(now):
        while alloc_history and now - alloc_history[0] > rate_window:
            alloc_history.popleft()

    def take_snapshot(now):
        prune_alloc_history(now)
        alloc_rate = len(alloc_history) / rate_window
        n_live = len(live)
        unfreed_ratio = (n_live / total_allocs_seen) if total_allocs_seen else 0.0

        # Baseline "typical" lifespan, from allocations that have ALREADY
        # been freed successfully. None until at least one free has happened
        # -- there's no baseline to compare against yet.
        avg_freed_lifespan = (
            freed_lifespan_sum / freed_lifespan_count
            if freed_lifespan_count > 0 else None
        )

        for addr, info in live.items():
            lifespan = now - info["alloc_time"]
            # How many multiples of the "normal" freed lifespan this
            # allocation has already lived. >>1 means "this has outlived
            # what usually gets freed around here" -- a per-address signal
            # that doesn't depend on the global unfreed_ratio at all.
            if avg_freed_lifespan and avg_freed_lifespan > 0:
                relative_lifespan = round(lifespan / avg_freed_lifespan, 4)
            else:
                relative_lifespan = None  # no baseline yet, e.g. very early in trace

            snapshots.append({
                "address": addr,
                "snapshot_time": round(now, 4),
                "lifespan": round(lifespan, 4),
                "size": info["size"],
                "alloc_rate": round(alloc_rate, 4),
                "unfreed_ratio": round(unfreed_ratio, 4),
                "relative_lifespan": relative_lifespan,
            })

    # Advance through events, taking snapshots at each interval boundary
    # BEFORE processing events that occur after it -- this ensures a
    # snapshot only reflects information known up to that instant.
    while event_idx < n_events:
        event = events[event_idx]

        while next_snapshot_t <= event["t"] and next_snapshot_t <= end_time:
            take_snapshot(next_snapshot_t)
            next_snapshot_t += sample_interval

        addr = event["address"]
        if event["event_type"] == "alloc":
            live[addr] = {"alloc_time": event["t"], "size": event["size"]}
            ground_truth[addr] = {"alloc_time": event["t"], "freed_at": None}
            alloc_history.append(event["t"])
            total_allocs_seen += 1
        elif event["event_type"] == "free":
            if addr in live:
                freed_lifespan_sum += event["t"] - live[addr]["alloc_time"]
                freed_lifespan_count += 1
                del live[addr]
            if addr in ground_truth:
                ground_truth[addr]["freed_at"] = event["t"]

        event_idx += 1

    # Final snapshot at end_time, and any remaining scheduled snapshot points
    while next_snapshot_t <= end_time:
        take_snapshot(next_snapshot_t)
        next_snapshot_t += sample_interval

    return snapshots, ground_truth


def summarize(snapshots, ground_truth, label=""):
    """Quick sanity-check summary: does unfreed_ratio trend make sense?"""
    n_addrs = len(ground_truth)
    n_leaked = sum(1 for g in ground_truth.values() if g["freed_at"] is None)

    if snapshots:
        early = [s["unfreed_ratio"] for s in snapshots
                 if s["snapshot_time"] <= snapshots[-1]["snapshot_time"] * 0.2]
        late = [s["unfreed_ratio"] for s in snapshots
                if s["snapshot_time"] >= snapshots[-1]["snapshot_time"] * 0.8]
        early_avg = sum(early) / len(early) if early else 0.0
        late_avg = sum(late) / len(late) if late else 0.0
    else:
        early_avg = late_avg = 0.0

    print(f"[{label}] {n_addrs} allocations, {n_leaked} never freed "
          f"({n_leaked/n_addrs*100:.1f}% leak rate)" if n_addrs else f"[{label}] no allocations")
    print(f"[{label}] {len(snapshots)} feature snapshots, "
          f"avg unfreed_ratio early={early_avg:.3f} -> late={late_avg:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Extract RL features from a trace CSV")
    parser.add_argument("trace_csv", help="Path to trace CSV")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--rate-window", type=float, default=2.0)
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to write feature snapshots as CSV")
    args = parser.parse_args()

    events = parse_trace(args.trace_csv)
    snapshots, ground_truth = extract_features(
        events, sample_interval=args.sample_interval, rate_window=args.rate_window
    )
    summarize(snapshots, ground_truth, label=args.trace_csv)

    if args.out:
        fieldnames = ["address", "snapshot_time", "lifespan", "size",
                      "alloc_rate", "unfreed_ratio", "relative_lifespan"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(snapshots)
        print(f"Wrote {len(snapshots)} snapshots to {args.out}")


if __name__ == "__main__":
    main()