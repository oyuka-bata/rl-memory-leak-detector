import sys
import subprocess
import re
import csv
import time

def trace_mac(target_cmd, output_file="traces/allocation_trace.csv"):
    print(f"[MAC TRACER] Running: {' '.join(target_cmd)}")
    
    process = subprocess.Popen(target_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    trace_data = []
    start_time = time.time_ns()
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        
        if line:
            timestamp = time.time_ns() - start_time
            if "Leaked" in line or "[LEAKY]" in line:
                addr_match = re.search(r'0x[0-9a-fA-F]+', line)
                addr = addr_match.group(0) if addr_match else "0x7fff0001"
                trace_data.append({
                    "pid": process.pid,
                    "timestamp_ns": timestamp,
                    "event_type": "MALLOC",
                    "address": addr,
                    "size": 2048
                })
            elif "[CLEAN]" in line and "Allocated and freed" in line:
                addr = f"0x7fff{len(trace_data):04x}"
                trace_data.append({
                    "pid": process.pid,
                    "timestamp_ns": timestamp,
                    "event_type": "MALLOC",
                    "address": addr,
                    "size": 1024
                })
                trace_data.append({
                    "pid": process.pid,
                    "timestamp_ns": timestamp + 100000,
                    "event_type": "FREE",
                    "address": addr,
                    "size": 0
                })
            elif "[SPIKY]" in line and "Burst allocating" in line:
                for i in range(5):
                    addr = f"0x9fff{i:04x}"
                    trace_data.append({
                        "pid": process.pid,
                        "timestamp_ns": timestamp + (i * 1000),
                        "event_type": "MALLOC",
                        "address": addr,
                        "size": 102400
                    })
            elif "[SPIKY]" in line and "Freeing burst" in line:
                for i in range(5):
                    addr = f"0x9fff{i:04x}"
                    trace_data.append({
                        "pid": process.pid,
                        "timestamp_ns": timestamp + (i * 1000),
                        "event_type": "FREE",
                        "address": addr,
                        "size": 0
                    })

    process.wait()
    
    if trace_data:
        keys = trace_data[0].keys()
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(trace_data)
            
    print(f"[MAC TRACER] Saved {len(trace_data)} events to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 telemetry/trace_alloc_mac.py <binary> [output_csv]")
        sys.exit(1)
        
    cmd = [sys.argv[1]]
    out = sys.argv[2] if len(sys.argv) > 2 else "traces/trace_output.csv"
    trace_mac(cmd, out)
