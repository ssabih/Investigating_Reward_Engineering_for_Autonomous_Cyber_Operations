""" Key Enhancements:
Randomized Machine Vulnerability and Health: Each episode starts with a random number of machines being vulnerable, making it unpredictable.
Time-Dependent State Changes: Machines can change their state over time, simulating real-world conditions where machines can be patched or become vulnerable.

Action Cost: Each action incurs a cost, discouraging unnecessary actions and promoting efficient exploration.
Complex Reward Structure: Rewards and penalties are provided for different actions, making the learning process more nuanced. """

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnhancedTwoSubnetEnv(gym.Env):
    def __init__(self):
        super(EnhancedTwoSubnetEnv, self).__init__()
        
        # Define action space: 0 - subnet scan, 1 - machine scan, 2 - machine access
        self.action_space = spaces.Discrete(10)
        
        # Define observation space
        # Each machine's state vector will have 4 elements: [discovered, compromised, vulnerable, healthy]
        # There are 4 machines, so the observation space is 4x4
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.float32)
        
        # Initialize internal attributes
        self.internal_states = None  # Internal ground truth for healthy and vulnerable machines
        self.state = None  # Agent's observable state
        self.compromise_reward = 10  # Reward for compromising a vulnerable machine
        self.fail_reward = -10  # Penalty for attempting to compromise a healthy machine
        self.cost = -0.1  # Cost for each action

    def _initialize_internal_states(self):
        """
        Initialize the internal states of the environment.
        These represent the ground truth of whether machines are healthy or vulnerable.
        """
        self.internal_states = np.zeros((4, 4), dtype=np.float32)
        num_vulnerable = np.random.randint(2, 5)
        vulnerable_indices = np.random.choice(4, num_vulnerable, replace=False)
        for idx in vulnerable_indices:
            self.internal_states[idx, 2] = 1  # Set as vulnerable
        for idx in range(4):
            if idx not in vulnerable_indices:
                self.internal_states[idx, 3] = 1  # Set as healthy

    def _reset_state(self):
        """
        Reset the agent's observable state to all zeros.
        """
        return np.zeros((4, 4), dtype=np.float32)

    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        self._initialize_internal_states()  # Set internal ground truth
        self.state = self._reset_state()  # Set observable state to all zeros
        return self.state

    def step(self, action):
        """
        Perform the action and update the state of the environment.
        """
        done = False
        reward = 0
        
        if action == 0:  # Subnet1 scan
            self._subnet1_scan()
        elif action == 1:  # Subnet2 scan
            self._subnet2_scan()
        elif action == 2:  # Machine1 scan
            self._machine_scan(0)
        elif action == 3:  # Machine1 access
            reward = self._machine_access(0)
        elif action == 4:  # Machine2 scan
            self._machine_scan(1)
        elif action == 5:  # Machine2 access
            reward = self._machine_access(1)
        elif action == 6:  # Machine3 scan
            self._machine_scan(2)
        elif action == 7:  # Machine3 access
            reward = self._machine_access(2)
        elif action == 8:  # Machine4 scan
            self._machine_scan(3)
        elif action == 9:  # Machine4 access
            reward = self._machine_access(3)

        # Introduce a time-dependent state change
        self._time_dependent_state_change()
        
        # Check if all vulnerable machines are compromised
        done = self._check_done()
        reward += self.cost  # Deduct cost for the action

        return self.state, reward, done, {}

    def _subnet1_scan(self):
        # Simulate a subnet scan
        self.state[0][0] = 1  # Discover machine 1
        self.state[1][0] = 1  # Discover machine 2

    def _subnet2_scan(self):
        # Simulate a subnet scan
        self.state[2][0] = 1  # Discover machine 3
        self.state[3][0] = 1  # Discover machine 4

    def _machine_scan(self, machine_index):
        # Simulate a machine scan
        if self.state[machine_index][0] == 1:  # If machine is discovered
            # Reveal the actual vulnerability and health status
            self.state[machine_index][2] = self.internal_states[machine_index][2]  # Vulnerable status
            self.state[machine_index][3] = self.internal_states[machine_index][3]  # Healthy status

    def _machine_access(self, machine_index):
        # Attempt to access a machine
        reward = 0
        if self.state[machine_index][0] == 1:  # If machine is discovered
            if self.internal_states[machine_index][2] == 1:  # If machine is vulnerable
                if self.state[machine_index][1] == 0:  # If machine is not Compromised
                    self.state[machine_index][1] = 1  # Compromise the machine
                    reward = self.compromise_reward
            elif self.internal_states[machine_index][3] == 1:  # If machine is healthy
                reward = self.fail_reward
        return reward

    def _check_done(self):
        # Check if all machines are discovered
        if np.all(self.state[:, 0] == 1):
            # Check if all vulnerable machines are compromised
            for i in range(4):
                if self.internal_states[i][2] == 1 and self.state[i][1] == 0:
                    return False
            return True
        else:
            return False


    def _time_dependent_state_change(self):
        # Introduce time-dependent state changes
        for i in range(4):
            if self.internal_states[i][2] == 1 and np.random.rand() < 0.15: #If a machine is currently vulnerable (`self.internal_states[i][2] == 1`), there is a 15% chance (`np.random.rand() < 0.15`) that it will become healthy. 
                self.internal_states[i][3] = 1  # the machine’s state is updated: its "healthy" status (`self.internal_states[i][3]`) is set to 1
                self.internal_states[i][2] = 0  # its "vulnerable" status (`self.internal_states[i][2]`) is set to 0. [simulating patching]
                self.internal_states[i][1] = 0  # Reset compromised status [simulating backup restore]

            elif self.internal_states[i][3] == 1 and np.random.rand() < 0.15:
                self.internal_states[i][2] = 1  # Healthy becomes vulnerable
                self.internal_states[i][3] = 0

    def render(self, mode='human'):
        """
        Render the current state of the environment.
        """
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