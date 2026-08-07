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
REWARD_TRUE_POSITIVE = 10.0    
REWARD_FALSE_POSITIVE = -5.0   
REWARD_FALSE_NEGATIVE = -10.0  
EARLY_BONUS_MAX = 5.0          
EARLY_BONUS_DECAY = 0.5        
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
        snapshots.sort(key=lambda s: (s["snapshot_time"], s["pid"], s["address"], s["alloc_instance"]))

        last_occurrence = {}
        for idx, s in enumerate(snapshots):
            key = (s["pid"], s["address"], s["alloc_instance"])
            last_occurrence[key] = idx  

        return snapshots, ground_truth, last_occurrence

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
        """Skip steps for allocations already resolved (flagged earlier)."""
        while self._pointer < len(self._steps):
            step = self._steps[self._pointer]
            key = (step["pid"], step["address"], step["alloc_instance"])
            if key not in self._resolved_addrs:
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
        self._steps, self._ground_truth, self._last_occurrence = self._load_trace_as_steps(trace_path)
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

    def _is_leak(self, key):
        gt = self._ground_truth[key]
        return (gt["freed_at"] is None
                and not gt["reclaimed"]
                and not gt["superseded"])

    def _early_detection_bonus(self, relative_lifespan):
        if relative_lifespan is None:
            return EARLY_BONUS_MAX  
        overshoot = max(relative_lifespan - 1.0, 0.0)
        return max(0.0, EARLY_BONUS_MAX - EARLY_BONUS_DECAY * overshoot)

    def step(self, action):
        if not self._steps:
            return (np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0, True, False, {"empty_trace": True})

        current = self._steps[self._pointer]
        addr = current["address"]
        key = (current["pid"], addr, current["alloc_instance"])
        current_idx = self._pointer

        if action == 1:  
            if self._is_leak(key):
                bonus = self._early_detection_bonus(current["relative_lifespan"])
                reward = REWARD_TRUE_POSITIVE + bonus
            else:
                reward = REWARD_FALSE_POSITIVE
            self._resolved_addrs.add(key)
        else:  
            reward = 0.0
            gt = self._ground_truth[key]
            if (current_idx == self._last_occurrence.get(key, -1)
                    and self._is_leak(key)):
                reward = REWARD_FALSE_NEGATIVE

        self._pointer += 1
        has_next = self._advance_to_next_valid_step()

        terminated = not has_next
        truncated = False  # no time-limit truncation in this scaffold yet

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