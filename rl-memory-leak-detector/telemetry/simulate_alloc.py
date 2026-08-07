#!/usr/bin/env python3
import argparse
import csv
import random
from datetime import datetime, timedelta

PATTERN_CONFIGS = {
    "clean": {
        "arrival": "poisson",       
        "rate": 5.0,               
        "size_range": (16, 4096),  
        "leak_fraction": 0.0,       
        "lifetime_range": (0.05, 2.0),  
        "burst": False,
    },
    "leaky": {
        "arrival": "poisson",
        "rate": 5.0,
        "size_range": (16, 4096),
        "leak_fraction": 0.35,      # ~1/3 of allocations leak
        "lifetime_range": (0.05, 2.0),
        "burst": False,
    },
    "spiky": {
        "arrival": "bursty",        
        "rate": 2.0,                
        "burst_rate": 30.0,         
        "burst_prob": 0.05,         
        "burst_duration": (0.5, 2.0),
        "size_range": (65536, 1048576),  
        "leak_fraction": 0.0,       
        "lifetime_range": (3.0, 8.0),    
        "burst": True,
    },
    "clean_app": {
        "arrival": "fixed",
        "rate": 10.0,                
        "size_range": (1024, 1024),  
        "leak_fraction": 0.0,
        "lifetime_range": (0.09, 0.11),  
        "burst": False,
        "n_events": 20,               
    },
    "leaky_app": {
        "arrival": "fixed",
        "rate": 6.67,                 
        "size_range": (2048, 2048),   
        "leak_fraction": 1.0,         
        "lifetime_range": (0.09, 0.11),  
        "burst": False,
        "n_events": 30,               
    },
    "spiky_app": {
        "arrival": "batch",
        "batch_size": 5,               
        "n_batches": 3,                
        "inter_alloc_gap": 0.05,       
        "hold_time": 1.0,              
        "inter_batch_gap": 0.2,        
        "size_range": (102400, 102400),  
        "leak_fraction": 0.0,
        "lifetime_range": (1.0, 1.0),  
        "burst": True,
    },
}


class AllocSimulator:
    def __init__(self, pattern, duration, pid, seed=None):
        if pattern not in PATTERN_CONFIGS:
            raise ValueError(f"Unknown pattern '{pattern}'. "
                              f"Choose from {list(PATTERN_CONFIGS)}")
        self.pattern = pattern
        self.cfg = PATTERN_CONFIGS[pattern]
        self.duration = duration
        self.pid = pid
        self.rng = random.Random(seed)
        self._next_addr = 0x7f0000000000  # fake heap base
        self._wall_start = datetime.now()

    def _fake_address(self):
        # advance a fake heap pointer so addresses look plausible/unique
        self._next_addr += self.rng.randint(32, 512)
        return self._next_addr

    def _sample_size(self):
        lo, hi = self.cfg["size_range"]
        # log-uniform: realistic allocators skew toward small sizes
        return int(round(2 ** self.rng.uniform(
            __import__("math").log2(lo), __import__("math").log2(hi)
        )))

    def _sample_lifetime(self):
        lo, hi = self.cfg["lifetime_range"]
        return self.rng.uniform(lo, hi)

    def _will_leak(self):
        return self.rng.random() < self.cfg["leak_fraction"]

    def _gen_arrival_times(self):
        """Return a sorted list of allocation timestamps (seconds from t=0)."""
        times = []
        t = 0.0

        if self.cfg["arrival"] == "poisson":
            rate = self.cfg["rate"]
            while t < self.duration:
                t += self.rng.expovariate(rate)
                if t < self.duration:
                    times.append(t)

        elif self.cfg["arrival"] == "fixed":
            # Evenly spaced allocations matching a real app's usleep() loop.
            # n_events overrides duration-based generation when present, so
            # counts match the real benchmark exactly (e.g. ITERATIONS=20).
            gap = 1.0 / self.cfg["rate"]
            n = self.cfg.get("n_events") or int(self.duration / gap)
            for i in range(n):
                times.append(i * gap)

        elif self.cfg["arrival"] == "batch":
            # Matches spiky_app.c: batches of N allocs, held, then freed,
            # repeated for n_batches cycles.
            gap = self.cfg["inter_alloc_gap"]
            batch_gap = self.cfg["inter_batch_gap"]
            for cycle in range(self.cfg["n_batches"]):
                for i in range(self.cfg["batch_size"]):
                    times.append(t)
                    t += gap
                t += self.cfg["hold_time"] + batch_gap

        elif self.cfg["arrival"] == "bursty":
            rate = self.cfg["rate"]
            burst_rate = self.cfg["burst_rate"]
            burst_prob = self.cfg["burst_prob"]
            bd_lo, bd_hi = self.cfg["burst_duration"]
            while t < self.duration:
                if self.rng.random() < burst_prob:
                    burst_len = self.rng.uniform(bd_lo, bd_hi)
                    burst_end = min(t + burst_len, self.duration)
                    while t < burst_end:
                        t += self.rng.expovariate(burst_rate)
                        if t < burst_end:
                            times.append(t)
                else:
                    t += self.rng.expovariate(rate)
                    if t < self.duration:
                        times.append(t)
        return times

    def run(self):
        events = []
        alloc_times = self._gen_arrival_times()
        tid_pool = [self.pid + i for i in range(1, 4)]  # a few fake threads
        is_batch_mode = self.cfg["arrival"] == "batch"

        for idx, t_alloc in enumerate(alloc_times):
            addr = self._fake_address()
            size = self._sample_size()
            tid = self.rng.choice(tid_pool)

            events.append({
                "t": t_alloc,
                "pid": self.pid,
                "tid": tid,
                "address": addr,
                "size": size,
                "event_type": "alloc",
            })

            if not self._will_leak():
                if is_batch_mode:
                    batch_size = self.cfg["batch_size"]
                    batch_start = (idx // batch_size) * batch_size
                    batch_alloc_start_t = alloc_times[batch_start]
                    lifetime = (batch_alloc_start_t + self.cfg["hold_time"]) - t_alloc
                    lifetime = max(lifetime, 0.01)
                else:
                    lifetime = self._sample_lifetime()
                t_free = t_alloc + lifetime
                if t_free < self.duration + 30:  # allow frees a bit past window
                    events.append({
                        "t": t_free,
                        "pid": self.pid,
                        "tid": tid,
                        "address": addr,
                        "size": size,
                        "event_type": "free",
                    })
            
        events.sort(key=lambda e: e["t"])
        return events

    def to_rows(self, events):
        rows = []
        for e in events:
            ts_ns = int(e["t"] * 1e9)
            wall = self._wall_start + timedelta(seconds=e["t"])
            rows.append({
                "timestamp_ns": ts_ns,
                "wall_time": wall.isoformat(),
                "pid": e["pid"],
                "tid": e["tid"],
                "address": hex(e["address"]),
                "size": e["size"],
                "event_type": e["event_type"],
            })
        return rows


def write_csv(rows, out_path):
    fieldnames = ["timestamp_ns", "wall_time", "pid", "tid",
                  "address", "size", "event_type"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} events ({out_path})")


def main():
    parser = argparse.ArgumentParser(description="Synthetic malloc/free trace generator")
    parser.add_argument("--pattern", choices=list(PATTERN_CONFIGS), required=True)
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    parser.add_argument("--duration", type=float, default=30.0,
                         help="Simulated duration in seconds (default: 30)")
    parser.add_argument("--pid", type=int, default=4021,
                         help="Fake PID to stamp on events (default: 4021)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")
    args = parser.parse_args()

    sim = AllocSimulator(args.pattern, args.duration, args.pid, seed=args.seed)
    events = sim.run()
    rows = sim.to_rows(events)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
