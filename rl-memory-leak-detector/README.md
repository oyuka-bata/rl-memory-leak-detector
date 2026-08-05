# rl-memory-leak-detector
🧠 Project Overview
This project builds an RL-powered memory leak detector that monitors real-time memory allocations via eBPF and predicts whether an active memory block will be leaked — before it causes an out-of-memory fault.

Traditional leak detectors rely on heavy runtime instrumentation (e.g., Valgrind) or static heuristics ("flag anything alive longer than N seconds"). Both trade off speed against false positives. This project trains a reinforcement learning agent on live memory-allocation telemetry to make fast, adaptive leak predictions — learning the pattern of a leak rather than applying a fixed rule.
✅ Core Features (Planned)
📡 Capture real-time memory allocation/free events via eBPF (bcc)
🧪 Generate labeled test workloads (clean, leaky, spiky allocation patterns)
🧠 Formulate memory traces as a custom Gymnasium RL environment
🤖 Train and compare DQN and PPO agents against a heuristic baseline
📊 Evaluate detection accuracy, latency, and false-positive rate
🧾 Package results into a research paper / portfolio-ready repository
📦 Tech Stack
Component
Tools/Libraries
Language
Python 3.10+, C (test workloads)
Telemetry
eBPF via bcc
RL Framework
gymnasium, stable-baselines3 (DQN, PPO)
Data Format
CSV / JSON allocation traces
Evaluation
scikit-learn (precision/recall/F1), custom latency benchmarks
Testing
pytest, unittest
Writing
IEEE/ACM LaTeX template

🧭 Use Cases
Systems Researchers: Study whether learned models can outperform static heuristics for real-time memory diagnostics
Kernel/Cloud Engineers: Explore lightweight, adaptive alternatives to heavyweight tools like Valgrind
Grad Admissions / Recruiters: A concrete, benchmarked artifact demonstrating combined systems + ML research ability
🏁 Getting Started
Clone the repository

git clone git@github.com:your-username/rl-memory-leak-detector.git

cd rl-memory-leak-detector

Set up a virtual environment

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

Run the eBPF tracer against a live process

sudo python telemetry/trace_alloc.py --pid <target_pid> --out traces/run1.csv

Run a sample trace against a test dummy app

python scripts/run_trace.py --app benchmarks/leaky_app.c --out traces/leaky_run1.csv
🧩 Project Structure (WIP)
rl-memory-leak-detector/

├── telemetry/              # eBPF tracing code (C / Python)

│   └── trace_alloc.py

├── environment/             # Custom Gymnasium RL environment (Python)

│   └── memory_leak_env.py

├── models/                  # Trained PPO/DQN model weights & training scripts

├── benchmarks/               # Test applications (clean / leaky / spiky)

│   ├── clean_app.c

│   ├── leaky_app.c

│   └── spiky_app.c

├── evaluation/               # Metrics + heuristic baseline comparisons

├── architecture.png          # System architecture diagram

└── README.md                 # Professional project overview & setup instructions
📌 Example Input
Trace event: PID 4021 called malloc(2048) at t=104 ticks, address 0x7fff5fbff4c0

No corresponding free() observed within process lifetime.

Threshold Heuristic Output:

FLAG: address alive > 10s

RL Agent Output:

State: [lifespan=104, size=2048, alloc_rate=3.2/s, unfreed_ratio=0.81]

Action: FLAG_LEAK

Confidence: early detection at t=104 (well before process exit at t=310)

Reasoning signal: high unfreed_ratio + rising alloc_rate for this PID

(Note: these numbers are illustrative — swap in real output once Phase 1 telemetry is running.)


🗓️ Timeline & Deliverables (4-Week Plan, eBPF-based)
Week 1 — Telemetry Layer
Goal: Get a working eBPF tracer producing structured allocation logs.

Day 1: Set up bcc/eBPF toolchain; confirm environment works with a trivial hello-world probe
Day 2: Hook malloc/free (or kmalloc/kfree) on a test process; print raw events to stdout
Day 3: Build structured logger — timestamp, address, size, PID/TID, call stack, event type — writing to CSV/JSON
Day 4: Write clean_app.c (allocates + always frees) and leaky_app.c (allocates in a loop, never frees)
Day 5: Write spiky_app.c (large allocations, held long, eventually freed — false-positive test case)
Day 6: Wrap tracer into a reusable CLI script: trace_alloc.py --pid X --out trace.csv
Day 7: Buffer day — handle eBPF edge cases (permissions, symbol resolution, dropped events under load)

Week 1 Deliverable: ☐ Working tracer + three labeled trace files (clean.csv, leaky.csv, spiky.csv) committed to /telemetry and /benchmarks


Week 2 — RL Environment
Goal: Translate raw traces into a working Gymnasium environment.

Day 8: Write feature extraction code — lifespan, allocation size, allocation rate, unfreed memory ratio — as a standalone module (test it independent of RL)
Day 9: Validate feature extraction against all three trace types; confirm leaky traces show rising unfreed ratio
Day 10: Scaffold MemoryLeakEnv(gymnasium.Env) — reset(), step(), observation/action space definitions
Day 11: Implement reward function (+10 true positive, -5 false positive, -10 false negative, early-detection bonus)
Day 12: Sanity-check rewards with two dummy agents (always-flag, never-flag); confirm reward totals behave as expected
Day 13: Handle edge cases — process exits mid-trace, multiple interleaved PIDs, malformed events
Day 14: Buffer day — write unit tests for the environment (pytest)

Week 2 Deliverable: ☐ Tested MemoryLeakEnv that completes a random-action rollout without crashing, with sane reward totals and passing unit tests


Week 3 — Training & Evaluation
Goal: Train agents and benchmark them against a heuristic baseline.

Day 15: Train DQN (stable-baselines3) on MemoryLeakEnv; log reward-over-episodes
Day 16: Tune DQN hyperparameters if training is unstable; save best checkpoint
Day 17: Train PPO on the same environment; log reward-over-episodes
Day 18: Tune PPO if needed; save best checkpoint
Day 19: Implement threshold heuristic baseline ("flag if alive > N seconds"); sweep N to find its best setting
Day 20: Generate held-out test traces (fresh runs of clean/leaky/spiky apps not used in training)
Day 21: Run DQN, PPO, and heuristic against held-out set; compute precision, recall, F1, detection latency; build comparison charts

Week 3 Deliverable: ☐ Trained model weights (DQN + PPO) + results table/chart comparing all three approaches


Week 4 — Packaging
Goal: Turn the working system into an admissions-ready artifact.

Day 22: Clean repo into final structure (telemetry/, environment/, models/, benchmarks/, evaluation/); add docstrings
Day 23: Rewrite README with real results (replace illustrative example above with actual trace/agent output)
Day 24: Draw system architecture diagram (telemetry → environment → agent → evaluation)
Day 25: Draft paper sections: Abstract, Introduction, System Architecture
Day 26: Draft paper sections: RL Formulation (state/action/reward, MDP definition)
Day 27: Draft paper sections: Experimental Results (insert real charts from Week 3)
Day 28: Full read-through and edit pass on the paper draft
Day 29: Buffer day — catch any slipped tasks from Weeks 1–3 (most likely candidates: eBPF quirks or a training rerun)
Day 30: Final polish — optional demo GIF/script for the README, final commit, tag release

Week 4 Deliverable: ☐ Public GitHub repo with clean structure + ☐ 4–6 page paper draft (IEEE/ACM format) + ☐ architecture diagram


📊 Overall Progress
Week 1: Telemetry Layer
Week 2: RL Environment
Week 3: Training & Evaluation
Week 4: Packaging
📌 Roadmap (High-Level)
- Phase 1: Telemetry layer (eBPF tracing + test dummy apps)

- Phase 2: RL environment formulation (state/action/reward design)

- Phase 3: Model training & evaluation (DQN vs PPO vs heuristic)

- Phase 4: Packaging (GitHub repo, architecture diagram, paper draft)

