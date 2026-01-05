import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnhancedTwoSubnetEnvRM(gym.Env):
    def __init__(self):
        super(EnhancedTwoSubnetEnvRM, self).__init__()

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.float32)

        self.internal_states = None  # Ground truth
        self.state = None            # Observable state

        self.compromise_reward = 10
        self.fail_reward = -10
        self.cost = -0.1

    def _initialize_internal_states(self):
        self.internal_states = np.zeros((4, 4), dtype=np.float32)
        num_vulnerable = np.random.randint(2, 5)
        vulnerable_indices = np.random.choice(4, num_vulnerable, replace=False)
        for idx in vulnerable_indices:
            self.internal_states[idx, 2] = 1  # Vulnerable
        for idx in range(4):
            if idx not in vulnerable_indices:
                self.internal_states[idx, 3] = 1  # Healthy

    def _reset_state(self):
        return np.zeros((4, 4), dtype=np.float32)

    def reset(self, **kwargs):
        self._initialize_internal_states()
        self.state = self._reset_state()
        return self.state, {}

    def step(self, action):
        reward = 0
        event = None

        if action == 0: self._subnet1_scan()
        elif action == 1: self._subnet2_scan()
        elif action == 2: self._machine_scan(0)
        elif action == 3: reward, event = self._machine_access(0)
        elif action == 4: self._machine_scan(1)
        elif action == 5: reward, event = self._machine_access(1)
        elif action == 6: self._machine_scan(2)
        elif action == 7: reward, event = self._machine_access(2)
        elif action == 8: self._machine_scan(3)
        elif action == 9: reward, event = self._machine_access(3)

        self._time_dependent_state_change()
        done = self._check_done()
        reward += self.cost

        return self.state, reward, done, False, {"event": event}

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

    def _machine_scan(self, idx):
        if self.state[idx][0] == 1:
            self.state[idx][2] = self.internal_states[idx][2]
            self.state[idx][3] = self.internal_states[idx][3]

    def _machine_access(self, idx):
        reward = 0
        event = None
        if self.state[idx][0] == 1:
            if self.internal_states[idx][2] == 1:
                if self.state[idx][1] == 0:
                    self.state[idx][1] = 1
                    reward = self.compromise_reward
                    event = f"compromise_m{idx+1}"
            elif self.internal_states[idx][3] == 1:
                reward = self.fail_reward
                event = f"fail_m{idx+1}"
        return reward, event

    def _check_done(self):
        if np.all(self.state[:, 0] == 1):
            for i in range(4):
                if self.internal_states[i][2] == 1 and self.state[i][1] == 0:
                    return False
            return True
        return False

    def _time_dependent_state_change(self):
        for i in range(4):
            if self.internal_states[i][2] == 1 and np.random.rand() < 0.15:
                self.internal_states[i][3] = 1
                self.internal_states[i][2] = 0
                self.internal_states[i][1] = 0
            elif self.internal_states[i][3] == 1 and np.random.rand() < 0.15:
                self.internal_states[i][2] = 1
                self.internal_states[i][3] = 0

    def render(self, mode='human'):
        if mode == 'human':
            print("\nCurrent State of Machines:")
            print("Machine | Discovered | Compromised | Vulnerable | Healthy")
            print("--------------------------------------------------------")
            for i in range(4):
                print(f"   {i+1}    |     {int(self.state[i][0])}      |      {int(self.state[i][1])}       |     {int(self.state[i][2])}      |    {int(self.state[i][3])}")
            print("\nInternal States (Ground Truth):")
            print("Machine | Vulnerable | Healthy")
            print("-----------------------------")
            for i in range(4):
                print(f"   {i+1}    |     {int(self.internal_states[i][2])}      |    {int(self.internal_states[i][3])}")
            print("\n")
