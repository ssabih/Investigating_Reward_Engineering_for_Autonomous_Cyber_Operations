import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TwoSubnetEnv(gym.Env):
    def __init__(self):
        super(TwoSubnetEnv, self).__init__()
        
        # Define action space: 0 - subnet scan, 1 - machine scan, 2 - machine access
        self.action_space = spaces.Discrete(10)
        
        # Define observation space
        # Each machine's state vector will have 4 elements: [discovered, compromised, vulnerable, healthy]
        # There are 4 machines, so the observation space is 4x4
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.float32)
        
        # Initialize state
        self.state = self._reset_state()
        
        # Define the rewards
        self.compromise_reward = 10  # Reward for compromising a vulnerable machine
        self.fail_reward = -10  # Penalty for attempting to compromise a healthy machine
        self.cost = -0.1 # Cost for each action
        self.reward = 0  # Initialize reward

    def _reset_state(self):
        # Initial states for the 4 machines
        # Example: all machines start as undiscovered and healthy
        initial_state = np.array([
            [0, 0, 0, 0],  # machine in subnet 1 (initial state: undiscovered, vulnerable)
            [0, 0, 0, 0],  # Another machine in subnet 1 (initial state: undiscovered, healthy)
            [0, 0, 0, 0],  # Machine in subnet 2 (initial state: undiscovered, healthy)
            [0, 0, 0, 0]   # Another machine in subnet 2 (initial state: undiscovered, vulnerable)
        ], dtype=np.float32)
        return initial_state

    def reset(self,**kwargs):
        self.state = self._reset_state()
        return self.state

    def step(self, action):
        done = False
        reward = 0
        if action == 0:  # Subnet1 scan
            self._subnet1_scan()
        elif action == 1:  # Subnet2 scan
            self._subnet2_scan()
        elif action == 2:  # Machine1 scan
            self._machine1_scan()
        elif action == 3:  # Machine1 access
            reward = self._machine1_access()
        elif action == 4:  # Machine2 scan
            self._machine2_scan()
        elif action == 5:  # Machine2 access
            reward = self._machine2_access()
        elif action == 6:  # Machine3 scan
            self._machine3_scan()
        elif action == 7:  # Machine3 access
            reward = self._machine3_access()
        elif action == 8:  # Machine4 scan
            self._machine4_scan()
        elif action == 9:  # Machine4 access
            reward = self._machine4_access()

        
        # Check if all machines are compromised to end the episode
        done = self._check_done()
        reward = reward + self.cost

        return self.state, reward, done, {}

    def _subnet1_scan(self):
        # Simulate a subnet scan
        # Discover machines in the respective subnets
            self.state[0][0] = 1  # Discover first machine in subnet 1
            self.state[1][0] = 1  # Discover second machine in subnet 1
    
    def _subnet2_scan(self):
        # Simulate a subnet scan
        # Discover machines in the respective subnets
            self.state[2][0] = 1  # Discover first machine in subnet 2
            self.state[3][0] = 1  # Discover second machine in subnet 2

    def _machine1_scan(self):
        # Simulate a machine scan for machine 1
        # Reveal the healthy and vulnerable status of a discovered machine
            if self.state[0][0] == 1:  # If machine is discovered
                self.state[0][2] = 1  # Reveal vulnerability. 
    def _machine2_scan(self):
        # Simulate a machine scan for machine 1
        # Reveal the healthy and vulnerable status of a discovered machine
            if self.state[1][0] == 1:  # If machine is discovered
                self.state[1][3] = 1  # Reveal healthy status. 
    def _machine3_scan(self):
        # Simulate a machine scan for machine 1
        # Reveal the healthy and vulnerable status of a discovered machine
            if self.state[2][0] == 1:  # If machine is discovered
                self.state[2][3] = 1  # Reveal healthy status. 
    def _machine4_scan(self):
        # Simulate a machine scan for machine 1
        # Reveal the healthy and vulnerable status of a discovered machine
            if self.state[3][0] == 1:  # If machine is discovered
                self.state[3][2] = 1  # Reveal vulnerability. 

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

    #def render(self, mode='human'):
        #print("State of machines:")
        #print(self.state)

    def close(self):
        pass

    def render(self, mode='human'):
        """
        Render the current state of the environment.
        """
        if mode == 'human':
            print("\nCurrent State:")
            print("Machine | Discovered | Compromised | Vulnerable | Healthy")
            print("--------------------------------------------------------")
            for i in range(4):
                print(f"   {i+1}    |     {int(self.state[i][0])}      |      {int(self.state[i][1])}       |     {int(self.state[i][2])}      |    {int(self.state[i][3])}")
            print("-----------------------------")
            print("\n")
            # Display available actions with their index numbers
        print("\nAvailable Actions:")
        print("Index | Action")
        print("--------------")
        print("  0   | Subnet 1 Scan")
        print("  1   | Subnet 2 Scan")
        print("  2   | Machine 1 Scan")
        print("  3   | Machine 1 Access")
        print("  4   | Machine 2 Scan")
        print("  5   | Machine 2 Access")
        print("  6   | Machine 3 Scan")
        print("  7   | Machine 3 Access")
        print("  8   | Machine 4 Scan")
        print("  9   | Machine 4 Access")
        print("\n")