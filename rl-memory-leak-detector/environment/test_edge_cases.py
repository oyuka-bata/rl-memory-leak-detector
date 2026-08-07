#!/usr/bin/env python3
import csv
import os
import tempfile

from feature_extraction import parse_trace, extract_features, is_leak


def write_trace(rows, path):
    fieldnames = ["timestamp_ns", "wall_time", "pid", "tid", "address", "size", "event_type"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            full = {"wall_time": "", "tid": r.get("tid", r["pid"] + 1), **r}
            writer.writerow(full)


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return condition


def test_interleaved_pids(tmpdir):
    print("\n--- Test 1: interleaved PIDs reusing the same address ---")
    path = os.path.join(tmpdir, "interleaved.csv")
    write_trace([
        {"timestamp_ns": 0,          "pid": 100, "address": "0xAAA", "size": 64, "event_type": "alloc"},
        {"timestamp_ns": 500_000_000, "pid": 200, "address": "0xAAA", "size": 128, "event_type": "alloc"},
        {"timestamp_ns": 1_000_000_000, "pid": 100, "address": "0xAAA", "size": 64, "event_type": "free"},
    ], path)

    events = parse_trace(path)
    snapshots, gt = extract_features(events, sample_interval=0.25, rate_window=2.0)

    key_100 = (100, "0xAAA", 0)
    key_200 = (200, "0xAAA", 0)

    all_ok = True
    all_ok &= check(len(gt) == 2, f"two independent (pid, address, instance) entries tracked (got {len(gt)})")
    all_ok &= check(gt[key_100]["freed_at"] is not None, "PID 100's allocation was freed")
    all_ok &= check(gt[key_200]["freed_at"] is None, "PID 200's allocation is still live (not merged with PID 100's free)")
    all_ok &= check(is_leak(gt[key_200]), "PID 200's allocation correctly counts as a leak")
    all_ok &= check(not is_leak(gt[key_100]), "PID 100's allocation correctly does NOT count as a leak")
    return all_ok


def test_process_exit_reclamation(tmpdir):
    print("\n--- Test 2: process exit reclaims live memory (not a leak) ---")
    path = os.path.join(tmpdir, "exit.csv")
    write_trace([
        {"timestamp_ns": 0,           "pid": 300, "address": "0xBBB", "size": 256, "event_type": "alloc"},
        {"timestamp_ns": 200_000_000, "pid": 300, "address": "0xCCC", "size": 512, "event_type": "alloc"},
        {"timestamp_ns": 1_000_000_000, "pid": 300, "address": "", "size": 0, "event_type": "exit"},
    ], path)

    events = parse_trace(path)
    snapshots, gt = extract_features(events, sample_interval=0.25, rate_window=2.0)

    all_ok = True
    all_ok &= check(len(gt) == 2, f"both allocations tracked (got {len(gt)})")
    for key in [(300, "0xBBB", 0), (300, "0xCCC", 0)]:
        all_ok &= check(gt[key]["reclaimed"] is True, f"{key}: marked reclaimed at exit")
        all_ok &= check(not is_leak(gt[key]), f"{key}: correctly NOT counted as a leak")
    return all_ok


def test_malformed_and_missed_free(tmpdir):
    print("\n--- Test 3: malformed row + missed-free/re-alloc collision ---")
    path = os.path.join(tmpdir, "malformed.csv")
    fieldnames = ["timestamp_ns", "wall_time", "pid", "tid", "address", "size", "event_type"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"timestamp_ns": 0, "wall_time": "", "pid": 400, "tid": 401,
                          "address": "0xDDD", "size": 64, "event_type": "alloc"})
        writer.writerow({"timestamp_ns": "not_a_number", "wall_time": "", "pid": 400, "tid": 401,
                          "address": "0xEEE", "size": 64, "event_type": "alloc"})
        writer.writerow({"timestamp_ns": 500_000_000, "wall_time": "", "pid": 400, "tid": 401,
                          "address": "0xDDD", "size": 128, "event_type": "alloc"})
        writer.writerow({"timestamp_ns": 800_000_000, "wall_time": "", "pid": 400, "tid": 401,
                          "address": "0xDDD", "size": 128, "event_type": "free"})
        writer.writerow({"timestamp_ns": 900_000_000, "wall_time": "", "pid": 400, "tid": 401,
                          "address": "0xFFF", "size": 0, "event_type": "free"})

    events = parse_trace(path)  
    all_ok = check(len(events) == 4, f"malformed row skipped, 4 valid events parsed (got {len(events)})")

    snapshots, gt = extract_features(events, sample_interval=0.25, rate_window=2.0)
    key_instance_0 = (400, "0xDDD", 0)  
    key_instance_1 = (400, "0xDDD", 1) 

    all_ok &= check(len(gt) == 2,
                     f"BOTH allocation instances at 0xDDD preserved as separate entries (got {len(gt)})")
    all_ok &= check(gt[key_instance_0]["superseded"] is True,
                     "the FIRST (missed-free) instance is correctly marked superseded")
    all_ok &= check(not is_leak(gt[key_instance_0]),
                     "superseded instance is excluded from leak count (unknown true outcome)")
    all_ok &= check(gt[key_instance_1]["freed_at"] is not None,
                     "the SECOND instance was freed correctly and is untouched by the collision")
    all_ok &= check(not is_leak(gt[key_instance_1]), "second instance correctly not a leak")

    try:
        parse_trace(path, strict=True)
        all_ok &= check(False, "strict=True raises on the malformed row")
    except (KeyError, ValueError, TypeError):
        all_ok &= check(True, "strict=True raises on the malformed row")
    return all_ok


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        results = [
            test_interleaved_pids(tmpdir),
            test_process_exit_reclamation(tmpdir),
            test_malformed_and_missed_free(tmpdir),
        ]
    print()
    print("ALL EDGE CASE TESTS PASSED" if all(results) else "SOME EDGE CASE TESTS FAILED")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)