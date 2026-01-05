
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import csv
import gymnasium as gym
import os

# Define the Reward Machine class
class RewardMachine:
    def __init__(self, fsm):
        self.fsm = fsm
        self.reset()

    def reset(self):
        self.current_state = 's0'

    def get_state(self):
        return self.current_state

    def get_one_hot_state(self):
        states = list(self.fsm.keys())
        one_hot = np.zeros(len(states))
        idx = states.index(self.current_state)
        one_hot[idx] = 1
        return one_hot

    def step(self, events):
        reward = 0
        #print(events)
        for event in events:
            if event in self.fsm[self.current_state]:
                next_state, r = self.fsm[self.current_state][event]
                #print(f"RM Transition: {self.current_state} --{event}/{r}--> {next_state}")
                self.current_state = next_state
                reward += r
        return reward

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent with RM
class DQNAgentRM:
    def __init__(self, state_dim, rm_state_dim, action_dim):
        self.state_dim = state_dim
        self.rm_state_dim = rm_state_dim
        self.input_dim = state_dim + rm_state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 0.1 #0.1 Vs 1
        self.epsilon_decay = 0.99999 #99999 Vs 995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.learning_rate = 0.001
        self.completed_episodes = 0

                
        self.model = DQN(self.input_dim, action_dim)
        self.target_model = DQN(self.input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, rm_state, action, reward, next_state, next_rm_state, done):
        state = state.flatten().astype(np.float32)
        next_state = next_state.flatten().astype(np.float32)
        self.memory.append((state, rm_state, action, reward, next_state, next_rm_state, done))

    def act(self, state, rm_state):
        combined_state = np.concatenate([state, rm_state])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(combined_state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, rm_states, actions, rewards, next_states, next_rm_states, dones = zip(*batch)

        states = np.array([np.concatenate([s, r]) for s, r in zip(states, rm_states)], dtype=np.float32)
        next_states = np.array([np.concatenate([s, r]) for s, r in zip(next_states, next_rm_states)], dtype=np.float32)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

full_fsm = {
    's0': {'subnet1_scan': ('s1', 1.0)},
    's1': {
        'scan_m1_vuln': ('s2', 1.5),
        'scan_m2_vuln': ('s3', 1.5),
        'scan_m1_clean': ('s2', 1.0),
        'scan_m2_clean': ('s3', 1.0),
        'subnet2_scan': ('s1', -0.5),
        'no_event': ('s1', -0.5)
    },
    's2': {
        'compromise_m1': ('s4', 6.0),
        'fail_m1': ('s2', -1.0),
        'subnet2_scan': ('s2', -0.5),
        'no_event': ('s2', -0.5)
    },
    's3': {
        'compromise_m2': ('s4', 6.0),
        'fail_m2': ('s3', -1.0),
        'subnet2_scan': ('s3', -0.5),
        'no_event': ('s3', -0.5)
    },
    's4': {'subnet2_scan': ('s5', 1.0)},
    's5': {
        'scan_m4_vuln': ('s6', 1.5),
        'scan_m3_vuln': ('s7', 1.5),
        'scan_m4_clean': ('s6', 1.0),
        'scan_m3_clean': ('s7', 1.0),
        'no_event': ('s5', -0.5)
    },
    's6': {
        'compromise_m4': ('s_terminal', 6.0),
        'fail_m4': ('s6', -1.0),
        'no_event': ('s6', -0.5)
    },
    's7': {
        'compromise_m3': ('s_terminal', 6.0),
        'fail_m3': ('s7', -1.0),
        'no_event': ('s7', -0.5)
    },
    's_terminal': {'no_event': ('s_terminal', -0.1)}
}


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_with_seed(seed):
    set_seed(seed)
    # Create the environment
    from gymnasium.envs.registration import register
    register(id='EnhancedTwoSubnetEnvRM-v0', entry_point='envs:EnhancedTwoSubnetEnvRM',) #EnhancedTwoSubnetEnvRM
    env = gym.make('EnhancedTwoSubnetEnvRM-v0')
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    rm_state_dim = len(full_fsm)
    action_dim = env.action_space.n
    agent = DQNAgentRM(state_dim, rm_state_dim, action_dim)
    rm = RewardMachine(full_fsm)

    n_episodes = 1000
    rewardList = []

    for e in range(n_episodes):
        state = env.reset()[0]
        state = state.flatten().astype(np.float32)
        rm.reset()
        rm_state = rm.get_one_hot_state()
        cumulative_reward = 0
        cumulative_env_reward = 0

        for t in range(50):
            action = agent.act(state, rm_state)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = next_state.flatten().astype(np.float32)
            events = env.get_events(action)
            rm_reward = rm.step(events)
            total_reward = rm_reward + env_reward
            next_rm_state = rm.get_one_hot_state()

            agent.remember(state, rm_state, action, total_reward, next_state, next_rm_state, done)
            state = next_state
            rm_state = next_rm_state
            cumulative_reward += total_reward
            cumulative_env_reward += env_reward

            if done:
                agent.update_target_model()
                agent.completed_episodes += 1
                rewardList.append((e+1, t+1, cumulative_env_reward, done))
                break

            if len(agent.memory) > agent.batch_size:
                agent.replay()

            if t == 49:
                rewardList.append((e+1, t+1, cumulative_env_reward, done))
                agent.update_target_model()

    # Save rewards to CSV (filename includes seed)
    env_name = 'Enhanced_TwoSubnetEnv_RM'
    filename = f'rm_rewards_{env_name}.csv'
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(['env', 'config', 'seed', 'episode', 'steps_in_episode', 'reward_env', 'success'])
        for r in rewardList:
            writer.writerow(['Advanced', 'Reward Machine DQN', seed, r[0], r[1], r[2], r[3]])

    print(f"Successfully completed: {agent.completed_episodes} episodes for seed {seed}")

seeds = [6, 8, 13, 17, 21, 27, 31, 38, 43, 76]
for seed in seeds:
    print(f"\n=== Running training with seed {seed} ===")
    train_with_seed(seed)
