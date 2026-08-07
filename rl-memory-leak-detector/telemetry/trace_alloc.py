#!/usr/bin/env python3

import argparse
import csv
import sys
import time
from datetime import datetime

try:
    from bcc import BPF
except ImportError:
    sys.exit(
        "bcc not installed. On Debian/Ubuntu: sudo apt install python3-bpfcc bpfcc-tools"
    )

#include <uapi/linux/ptrace.h>

struct alloc_info_t {
    u64 size;
    u64 timestamp_ns;
};

struct event_t {
    u32 pid;
    u32 tid;
    u64 timestamp_ns;
    u64 address;
    u64 size;
    int event_type; // 0 = alloc, 1 = free
};

BPF_HASH(sizes, u64, u64);                        // tid -> pending malloc size
BPF_HASH(live_allocs, u64, struct alloc_info_t);  // address -> alloc info
BPF_PERF_OUTPUT(events);

int trace_malloc_entry(struct pt_regs *ctx, size_t size) {
    u64 tid = bpf_get_current_pid_tgid();
    sizes.update(&tid, &size);
    return 0;
}

int trace_malloc_return(struct pt_regs *ctx) {
    u64 tid = bpf_get_current_pid_tgid();
    u64 *size_ptr = sizes.lookup(&tid);
    if (size_ptr == 0) {
        return 0; // no matching entry event (e.g. attached mid-call)
    }

    u64 address = PT_REGS_RC(ctx);
    if (address == 0) {
        sizes.delete(&tid);
        return 0; // malloc failed, nothing was allocated
    }

    struct alloc_info_t info = {};
    info.size = *size_ptr;
    info.timestamp_ns = bpf_ktime_get_ns();
    live_allocs.update(&address, &info);

    struct event_t event = {};
    event.pid = tid >> 32;
    event.tid = (u32)tid;
    event.timestamp_ns = info.timestamp_ns;
    event.address = address;
    event.size = info.size;
    event.event_type = 0;
    events.perf_submit(ctx, &event, sizeof(event));

    sizes.delete(&tid);
    return 0;
}

int trace_free_entry(struct pt_regs *ctx, void *address) {
    u64 addr = (u64)address;
    if (addr == 0) {
        return 0; // free(NULL) is a no-op
    }

    struct alloc_info_t *info = live_allocs.lookup(&addr);
    if (info == 0) {
        return 0; // untracked address (allocated before trace start)
    }

    u64 tid = bpf_get_current_pid_tgid();

    struct event_t event = {};
    event.pid = tid >> 32;
    event.tid = (u32)tid;
    event.timestamp_ns = bpf_ktime_get_ns();
    event.address = addr;
    event.size = info->size;
    event.event_type = 1;
    events.perf_submit(ctx, &event, sizeof(event));

    live_allocs.delete(&addr);
    return 0;
}
"""


class TraceAllocator:
    def __init__(self, pid, out_path, libc_path):
        self.pid = pid
        self.out_path = out_path
        self.libc_path = libc_path
        self.start_time = time.time()
        self.rows = []

        self.bpf = BPF(text=BPF_PROGRAM)
        self.bpf.attach_uprobe(name=libc_path, sym="malloc",
                                fn_name="trace_malloc_entry", pid=pid)
        self.bpf.attach_uretprobe(name=libc_path, sym="malloc",
                                   fn_name="trace_malloc_return", pid=pid)
        self.bpf.attach_uprobe(name=libc_path, sym="free",
                                fn_name="trace_free_entry", pid=pid)

        self.bpf["events"].open_perf_buffer(self._handle_event)

    def _handle_event(self, cpu, data, size):
        event = self.bpf["events"].event(data)
        self.rows.append({
            "timestamp_ns": event.timestamp_ns,
            "wall_time": datetime.now().isoformat(),
            "pid": event.pid,
            "tid": event.tid,
            "address": hex(event.address),
            "size": event.size,
            "event_type": "alloc" if event.event_type == 0 else "free",
        })

    def run(self, duration=None):
        print(f"Tracing PID {self.pid} via {self.libc_path} (malloc/free)... "
              f"{'Ctrl-C to stop' if not duration else f'running {duration}s'}")
        try:
            while True:
                self.bpf.perf_buffer_poll(timeout=100)
                if duration and (time.time() - self.start_time) > duration:
                    break
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self._write_csv()

    def _write_csv(self):
        fieldnames = ["timestamp_ns", "wall_time", "pid", "tid",
                      "address", "size", "event_type"]
        with open(self.out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Wrote {len(self.rows)} events to {self.out_path}")


def main():
    parser = argparse.ArgumentParser(description="eBPF malloc/free tracer")
    parser.add_argument("--pid", type=int, required=True,
                         help="Target process PID to trace")
    parser.add_argument("--out", type=str, required=True,
                         help="Output CSV path, e.g. traces/run1.csv")
    parser.add_argument("--libc", type=str,
                         default="/lib/x86_64-linux-gnu/libc.so.6",
                         help="Path to libc (varies by distro/arch -- "
                              "run `ldconfig -p | grep libc.so` to find yours)")
    parser.add_argument("--duration", type=int, default=None,
                         help="Trace duration in seconds (default: run until Ctrl-C)")
    args = parser.parse_args()

    tracer = TraceAllocator(args.pid, args.out, args.libc)
    tracer.run(duration=args.duration)


if __name__ == "__main__":
    main()
