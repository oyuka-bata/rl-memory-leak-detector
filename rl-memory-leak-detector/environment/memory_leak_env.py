#!/usr/bin/env python3

import random
import sys
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import parse_trace, extract_features

RELATIVE_LIFESPAN_SENTINEL = -1.0  
OBS_LOW = np.array([0.0, 0.0, 0.0, 0.0, RELATIVE_LIFESPAN_SENTINEL], dtype=np.float32)
OBS_HIGH = np.array([1e6, 1e9, 1e4, 1.0, 1e4], dtype=np.float32)


class MemoryLeakEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, trace_paths, sample_interval=1.0, rate_window=2.0, shuffle_traces=True):
        super().__init__()
        if not trace_paths:
            raise ValueError("trace_paths must be a non-empty list of CSV paths")
        self.trace_paths = list(trace_paths)
        self.sample_interval = sample_interval
        self.rate_window = rate_window
        self.shuffle_traces = shuffle_traces

        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Discrete(2)  

        self._steps = []          
        self._ground_truth = {}   
        self._pointer = 0
        self._resolved_addrs = set()  
        self._current_trace_path = None

    def _load_trace_as_steps(self, trace_path):
        events = parse_trace(trace_path)
        snapshots, ground_truth = extract_features(
            events, sample_interval=self.sample_interval, rate_window=self.rate_window
        )
        snapshots.sort(key=lambda s: (s["snapshot_time"], s["address"]))
        return snapshots, ground_truth

    def _obs_from_step(self, step):
        rel = step["relative_lifespan"]
        rel = rel if rel is not None else RELATIVE_LIFESPAN_SENTINEL
        return np.array([
            step["lifespan"],
            step["size"],
            step["alloc_rate"],
            step["unfreed_ratio"],
            rel,
        ], dtype=np.float32)

    def _advance_to_next_valid_step(self):
        while self._pointer < len(self._steps):
            addr = self._steps[self._pointer]["address"]
            if addr not in self._resolved_addrs:
                return True
            self._pointer += 1
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        trace_path = (random.choice(self.trace_paths) if self.shuffle_traces
                      else self.trace_paths[0])
        self._current_trace_path = trace_path
        self._steps, self._ground_truth = self._load_trace_as_steps(trace_path)
        self._pointer = 0
        self._resolved_addrs = set()

        if not self._steps:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"trace_path": trace_path, "empty_trace": True}
            return obs, info

        self._advance_to_next_valid_step()
        obs = self._obs_from_step(self._steps[self._pointer])
        info = {
            "trace_path": trace_path,
            "address": self._steps[self._pointer]["address"],
            "snapshot_time": self._steps[self._pointer]["snapshot_time"],
        }
        return obs, info

    def step(self, action):
        # --- Placeholder reward logic (Day 10 scaffold only) ---
        # TODO (Day 11): replace with the real reward function:
        #   +10 true positive, -5 false positive, -10 false negative,
        #   plus an early-detection bonus. Requires comparing `action`
        #   against self._ground_truth[addr]["freed_at"] (None = real leak).
        reward = 0.0
        # ---------------------------------------------------------

        if not self._steps:
            return (np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0, True, False, {"empty_trace": True})

        current = self._steps[self._pointer]
        addr = current["address"]

        if action == 1:  
            self._resolved_addrs.add(addr)

        self._pointer += 1
        has_next = self._advance_to_next_valid_step()

        terminated = not has_next
        truncated = False 

        if has_next:
            next_step = self._steps[self._pointer]
            obs = self._obs_from_step(next_step)
            info = {
                "address": next_step["address"],
                "snapshot_time": next_step["snapshot_time"],
                "acted_on_address": addr,
                "action_taken": action,
            }
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"acted_on_address": addr, "action_taken": action, "episode_done": True}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  
    
    def close(self):
        pass