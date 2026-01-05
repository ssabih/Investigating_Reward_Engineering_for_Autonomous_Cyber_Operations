import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import csv
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 0.1
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.learning_rate = 0.001
        self.completed_episodes = 0

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert lists of numpy arrays to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
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

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_with_seed(seed):
    set_seed(seed)
    # Create and register the environment
    from gymnasium.envs.registration import register
    register(id='EnhancedTwoSubnetEnv-v0', entry_point='envs:EnhancedTwoSubnetEnv',)
    env = gym.make('EnhancedTwoSubnetEnv-v0')

    # Define the training loop
    n_episodes = 1000
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    done = False
    batch_size = 64
    rewardList = []

    for e in range(n_episodes):
        state = env.reset()
        state = state.flatten()  # Flatten the state
        cumulative_reward = 0
        for time in range(50):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()  # Flatten the next state
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            cumulative_reward += reward  # Accumulate reward
            if done:
                agent.update_target_model()
                agent.completed_episodes += 1
                rewardList.append((e+1, time+1, cumulative_reward, done))
                #rewardList.append(cumulative_reward)
                #print(f"episode: {e+1}/{n_episodes}, steps: {time+1}, score: {cumulative_reward}")
                break
            if len(agent.memory) > batch_size:
                agent.replay()
            if time == 49:
                rewardList.append((e+1, time+1, cumulative_reward, done))
                #rewardList.append(cumulative_reward)
                agent.update_target_model()
                #print(f"episode: {e+1}/{n_episodes}, steps: {time+1}, score: {cumulative_reward}")

    # Save results to CSV
    write_header = not os.path.exists('dqn_agent_Enhanced.csv') or os.path.getsize('dqn_agent_Enhanced.csv') == 0
    with open('dqn_agent_Enhanced.csv', 'a', newline='') as csvfile: 
        csvwriter = csv.writer(csvfile)
        if write_header:
            csvwriter.writerow(['env', 'config', 'seed', 'episode', 'steps_in_episode', 'reward_env', 'success'])
        for eachreward in rewardList:
            csvwriter.writerow(['Advanced', 'Baseline DQN', seed, eachreward[0], eachreward[1], eachreward[2], eachreward[3]])
            #csvwriter.writerow([eachreward])

    print(f"Successfully completed: {agent.completed_episodes} episodes")

seeds = [6, 8, 13, 17, 21, 27, 31, 38, 43, 76]
for seed in seeds:
    print(f"\n=== Running training with seed {seed} ===")
    train_with_seed(seed)