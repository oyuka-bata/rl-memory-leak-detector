#!/usr/bin/env python3

import argparse
import csv
import sys
from collections import deque


def parse_trace(csv_path, strict=False):
    events = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader, start=2):  # header is line 1
            try:
                event_type = row["event_type"]
                event = {
                    "t": int(row["timestamp_ns"]) / 1e9,
                    "pid": int(row["pid"]),
                    "tid": int(row["tid"]) if row.get("tid") not in (None, "") else -1,
                    "address": row.get("address", ""),
                    "size": int(row["size"]) if row.get("size") not in (None, "") else 0,
                    "event_type": event_type,
                }
                if event_type not in ("alloc", "free", "exit"):
                    raise ValueError(f"unknown event_type '{event_type}'")
                if event_type in ("alloc", "free") and not event["address"]:
                    raise ValueError("alloc/free event missing address")
                if event["t"] < 0 or event["size"] < 0:
                    raise ValueError("negative timestamp or size")
            except (KeyError, ValueError, TypeError) as e:
                if strict:
                    raise
                print(f"WARNING: skipping malformed row at {csv_path}:{line_num}: {e}",
                      file=sys.stderr)
                continue
            events.append(event)
    events.sort(key=lambda e: e["t"])
    return events


def extract_features(events, sample_interval=0.5, rate_window=2.0):
    if not events:
        return [], {}

    live = {}              
    ground_truth = {}     
    active_instance = {}   
    next_instance = {}     
    alloc_history = deque()  
    total_allocs_seen = 0
    freed_lifespan_sum = 0.0   
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
        avg_freed_lifespan = (
            freed_lifespan_sum / freed_lifespan_count
            if freed_lifespan_count > 0 else None
        )

        for (pid, addr, instance), info in live.items():
            lifespan = now - info["alloc_time"]
            if avg_freed_lifespan and avg_freed_lifespan > 0:
                relative_lifespan = round(lifespan / avg_freed_lifespan, 4)
            else:
                relative_lifespan = None 

            snapshots.append({
                "pid": pid,
                "address": addr,
                "alloc_instance": instance,
                "snapshot_time": round(now, 4),
                "lifespan": round(lifespan, 4),
                "size": info["size"],
                "alloc_rate": round(alloc_rate, 4),
                "unfreed_ratio": round(unfreed_ratio, 4),
                "relative_lifespan": relative_lifespan,
            })

    while event_idx < n_events:
        event = events[event_idx]

        while next_snapshot_t <= event["t"] and next_snapshot_t <= end_time:
            take_snapshot(next_snapshot_t)
            next_snapshot_t += sample_interval

        pid = event["pid"]
        addr = event["address"]
        pid_addr = (pid, addr)

        if event["event_type"] == "alloc":
            if pid_addr in active_instance:
                old_key = (pid, addr, active_instance[pid_addr])
                if old_key in ground_truth:
                    ground_truth[old_key]["superseded"] = True
                if old_key in live:
                    del live[old_key]

            instance = next_instance.get(pid_addr, 0)
            key = (pid, addr, instance)
            active_instance[pid_addr] = instance
            next_instance[pid_addr] = instance + 1

            live[key] = {"alloc_time": event["t"], "size": event["size"]}
            ground_truth[key] = {
                "alloc_time": event["t"], "freed_at": None,
                "reclaimed": False, "superseded": False,
            }
            alloc_history.append(event["t"])
            total_allocs_seen += 1

        elif event["event_type"] == "free":
            if pid_addr in active_instance:
                key = (pid, addr, active_instance[pid_addr])
                if key in live:
                    freed_lifespan_sum += event["t"] - live[key]["alloc_time"]
                    freed_lifespan_count += 1
                    del live[key]
                if key in ground_truth:
                    ground_truth[key]["freed_at"] = event["t"]
                del active_instance[pid_addr]
        elif event["event_type"] == "exit":
            pid_addrs_to_close = [pa for pa in active_instance if pa[0] == pid]
            for pa in pid_addrs_to_close:
                key = (pa[0], pa[1], active_instance[pa])
                if key in ground_truth:
                    ground_truth[key]["reclaimed"] = True
                    ground_truth[key]["freed_at"] = event["t"]
                if key in live:
                    del live[key]
                del active_instance[pa]

        event_idx += 1

    while next_snapshot_t <= end_time:
        take_snapshot(next_snapshot_t)
        next_snapshot_t += sample_interval

    return snapshots, ground_truth


def is_leak(gt_entry):
    return (gt_entry["freed_at"] is None
            and not gt_entry["reclaimed"]
            and not gt_entry["superseded"])


def summarize(snapshots, ground_truth, label=""):
    n_addrs = len(ground_truth)
    n_leaked = sum(1 for g in ground_truth.values() if is_leak(g))
    n_reclaimed = sum(1 for g in ground_truth.values() if g["reclaimed"])
    n_superseded = sum(1 for g in ground_truth.values() if g["superseded"])

    if snapshots:
        early = [s["unfreed_ratio"] for s in snapshots
                 if s["snapshot_time"] <= snapshots[-1]["snapshot_time"] * 0.2]
        late = [s["unfreed_ratio"] for s in snapshots
                if s["snapshot_time"] >= snapshots[-1]["snapshot_time"] * 0.8]
        early_avg = sum(early) / len(early) if early else 0.0
        late_avg = sum(late) / len(late) if late else 0.0
    else:
        early_avg = late_avg = 0.0

    if n_addrs:
        print(f"[{label}] {n_addrs} allocations, {n_leaked} genuine leaks "
              f"({n_leaked/n_addrs*100:.1f}%), {n_reclaimed} reclaimed at exit, "
              f"{n_superseded} superseded (missed free)")
    else:
        print(f"[{label}] no allocations")
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
        fieldnames = ["pid", "address", "alloc_instance", "snapshot_time", "lifespan",
                      "size", "alloc_rate", "unfreed_ratio", "relative_lifespan"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(snapshots)
        print(f"Wrote {len(snapshots)} snapshots to {args.out}")


if __name__ == "__main__":
    main()