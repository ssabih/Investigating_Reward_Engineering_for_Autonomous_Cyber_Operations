import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import csv

# -----------------------------
# Environment Definition
# -----------------------------
class TwoSubnetEnv_RM(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.float32)
        self.state = self._reset_state()
        self.internal_states = self.state.copy()
        self.compromise_reward = 10
        self.fail_reward = -10
        self.cost = -0.1
        self.reward = 0
        
        # Define observation space
        # Each machine's state vector will have 4 elements: [discovered, compromised, vulnerable, healthy]
        # There are 4 machines, so the observation space is 4x4

    def _reset_state(self):
        return np.array([
            [0, 0, 0, 0], # machine in subnet 1 (initial state: undiscovered, vulnerable)
            [0, 0, 0, 0], # Another machine in subnet 1 (initial state: undiscovered, healthy)
            [0, 0, 0, 0], # Machine in subnet 2 (initial state: undiscovered, healthy)
            [0, 0, 0, 0]  # Another machine in subnet 2 (initial state: undiscovered, vulnerable)
        ], dtype=np.float32)

    def reset(self, **kwargs):
        self.state = self._reset_state()
        self.internal_states = self.state.copy()
        return self.state, {}
        

    def step(self, action):
        self.internal_states = self.state.copy()
        reward = 0
        if action == 0: self._subnet1_scan()
        elif action == 1: self._subnet2_scan()
        elif action == 2: self._machine1_scan()
        elif action == 3: reward = self._machine1_access()
        elif action == 4: self._machine2_scan()
        elif action == 5: reward = self._machine2_access()
        elif action == 6: self._machine3_scan()
        elif action == 7: reward = self._machine3_access()
        elif action == 8: self._machine4_scan()
        elif action == 9: reward = self._machine4_access()
        self.internal_states = self.state.copy()
        done = self._check_done()
        reward += self.cost
        return self.state, reward, done, False, {}

    def get_events(self, action):
        events = []

        def scan_event(machine_idx, label):
            if self.state[machine_idx][2] == 1:
                events.append(f"scan_{label}_vuln")
            elif self.state[machine_idx][3] == 1:
                events.append(f"scan_{label}_clean")
            else:
                events.append(f"scan_{label}")

        def compromise_event(machine_idx, label):
            if self.state[machine_idx][2] == 1:
                events.append(f"compromise_{label}")
            else:
                events.append(f"fail_{label}")

        if action == 0:
            events.append("subnet1_scan")
        elif action == 1:
            events.append("subnet2_scan")
        elif action == 2 and self.state[0][0] == 1:
            scan_event(0, "m1")
        elif action == 3 and self.state[0][0] == 1:
            compromise_event(0, "m1")
        elif action == 4 and self.state[1][0] == 1:
            scan_event(1, "m2")
        elif action == 5 and self.state[1][0] == 1:
            compromise_event(1, "m2")
        elif action == 6 and self.state[2][0] == 1:
            scan_event(2, "m3")
        elif action == 7 and self.state[2][0] == 1:
            compromise_event(2, "m3")
        elif action == 8 and self.state[3][0] == 1:
            scan_event(3, "m4")
        elif action == 9 and self.state[3][0] == 1:
            compromise_event(3, "m4")

        if not events:
            events.append("no_event")

        return events


    def _subnet1_scan(self):
        self.state[0][0] = 1
        self.state[1][0] = 1

    def _subnet2_scan(self):
        self.state[2][0] = 1
        self.state[3][0] = 1

    def _machine1_scan(self):
        if self.state[0][0] == 1: self.state[0][2] = 1

    def _machine2_scan(self):
        if self.state[1][0] == 1: self.state[1][3] = 1

    def _machine3_scan(self):
        if self.state[2][0] == 1: self.state[2][3] = 1

    def _machine4_scan(self):
        if self.state[3][0] == 1: self.state[3][2] = 1

    def _machine1_access(self):
        # Simulate accessing a machine
        reward=0
        if self.state[0][0] == 1:  # If machine is discovered
            if self.state[0][2] == 1:  # If machine is vulnerable
                if self.state[0][1] == 0:  # If machine is not Compromised
                    self.state[0][1] = 1  # Compromise the machine
                    reward = self.compromise_reward
            elif self.state[0][3] == 1:  # If machine is healthy
                reward = self.fail_reward
        return reward
    def _machine2_access(self):
        reward=0
        # Simulate accessing a machine
        if self.state[1][0] == 1:  # If machine is discovered
            if self.state[1][2] == 1:  # If machine is vulnerable
                self.state[1][1] = 1  # Compromise the machine
                reward = self.compromise_reward
            elif self.state[1][3] == 1:  # If machine is healthy
                reward = self.fail_reward
        return reward
    def _machine3_access(self):
        reward=0
        # Simulate accessing a machine
        if self.state[2][0] == 1:  # If machine is discovered
            if self.state[2][2] == 1:  # If machine is vulnerable
                self.state[2][1] = 1  # Compromise the machine
                reward = self.compromise_reward
            elif self.state[2][3] == 1:  # If machine is healthy
                reward = self.fail_reward
        return reward
    def _machine4_access(self):
        reward=0
        # Simulate accessing a machine
        if self.state[3][0] == 1:  # If machine is discovered
            if self.state[3][2] == 1:  # If machine is vulnerable
                if self.state[3][1] == 0:  # If machine is not Compromised
                    self.state[3][1] = 1  # Compromise the machine
                    reward = self.compromise_reward
            elif self.state[3][3] == 1:  # If machine is healthy
                reward = self.fail_reward
        return reward

    def _check_done(self):
        # Check if all machines are discovered
        if np.all(self.state[:, 0] == 1):
            # Check if all vulnerable machines are compromised
            if self.state[0, 1] == 1 and self.state[3, 1] == 1:
                return True
            else:
                return False
        else:
            return False
