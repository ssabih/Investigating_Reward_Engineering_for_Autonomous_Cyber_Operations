import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym

# ------------------ Common Utils ------------------

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ensure_header(path, header):
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    return f, w

# ------------------ Base DQN ------------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ------------------ Shaped DQN ------------------

class DQNAgentShaped:
    def __init__(self, state_dim, action_dim, gamma=0.9, eps=0.1, eps_decay=0.99999, eps_min=0.01, lr=1e-3, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.batch_size = batch_size

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.completed_episodes = 0
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def potential_function(self, state):
        # state is 1D flattened; reshape to (n_machines, 4) = [discovered, compromised, vulnerable, healthy]
        t = torch.as_tensor(state, dtype=torch.float32).view(-1, 4)
        discovered  = t[:,0].sum().item()
        compromised = t[:,1].sum().item()
        vulnerable  = t[:,2].sum().item()
        lam = 1.0; beta = 2.0; zeta = 3.0
        return lam*discovered + beta*vulnerable + zeta*compromised

    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.model(st)
        return int(torch.argmax(q[0]).item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s  = torch.as_tensor(np.array(s),  dtype=torch.float32)
        a  = torch.as_tensor(np.array(a),  dtype=torch.long)
        r  = torch.as_tensor(np.array(r),  dtype=torch.float32)
        s2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
        d  = torch.as_tensor(np.array(d),  dtype=torch.float32)

        pot  = torch.as_tensor([self.potential_function(x) for x in s],  dtype=torch.float32)
        pot2 = torch.as_tensor([self.potential_function(x) for x in s2], dtype=torch.float32)
        shaped_r = r + self.gamma*pot2 - pot

        q      = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        nq     = self.target_model(s2).max(1)[0]
        target = shaped_r + self.gamma*nq*(1-d)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ------------------ Reward Machine DQN ------------------

class RewardMachine:
    def __init__(self, fsm):
        self.fsm = fsm
        self.states = list(fsm.keys())
        self.reset()
    def reset(self):
        self.current_state = 's0'
    def get_one_hot(self):
        oh = np.zeros(len(self.states), dtype=np.float32)
        oh[self.states.index(self.current_state)] = 1.0
        return oh
    def step(self, events):
        r = 0.0
        for ev in events:
            if ev in self.fsm[self.current_state]:
                nxt, rew = self.fsm[self.current_state][ev]
                self.current_state = nxt
                r += rew
        return r

class DQNAgentRM:
    def __init__(self, state_dim, rm_dim, action_dim, gamma=0.9, eps=0.1, eps_decay=0.99999, eps_min=0.01, lr=1e-3, batch_size=64):
        self.state_dim = state_dim
        self.rm_dim = rm_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.batch_size = batch_size

        self.model = DQN(state_dim + rm_dim, action_dim)
        self.target_model = DQN(state_dim + rm_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.completed_episodes = 0
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, rm_s, a, r, s2, rm_s2, d):
        self.memory.append((s, rm_s, a, r, s2, rm_s2, d))

    def act(self, s, rm_s):
        combined = np.concatenate([s, rm_s]).astype(np.float32)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        st = torch.as_tensor(combined, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.model(st)
        return int(torch.argmax(q[0]).item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, rm_s, a, r, s2, rm_s2, d = zip(*batch)

        s  = np.array([np.concatenate([x, y]).astype(np.float32) for x, y in zip(s, rm_s)])
        s2 = np.array([np.concatenate([x, y]).astype(np.float32) for x, y in zip(s2, rm_s2)])
        s  = torch.as_tensor(s,  dtype=torch.float32)
        s2 = torch.as_tensor(s2, dtype=torch.float32)
        a  = torch.as_tensor(a,  dtype=torch.long)
        r  = torch.as_tensor(r,  dtype=torch.float32)
        d  = torch.as_tensor(d,  dtype=torch.float32)

        q      = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        nq     = self.target_model(s2).max(1)[0]
        target = r + self.gamma*nq*(1-d)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# FSM copied for self-contained script
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
    's2': {'compromise_m1': ('s4', 6.0), 'fail_m1': ('s2', -1.0), 'subnet2_scan': ('s2', -0.5), 'no_event': ('s2', -0.5)},
    's3': {'compromise_m2': ('s4', 6.0), 'fail_m2': ('s3', -1.0), 'subnet2_scan': ('s3', -0.5), 'no_event': ('s3', -0.5)},
    's4': {'subnet2_scan': ('s5', 1.0)},
    's5': {
        'scan_m4_vuln': ('s6', 1.5), 'scan_m3_vuln': ('s7', 1.5),
        'scan_m4_clean': ('s6', 1.0), 'scan_m3_clean': ('s7', 1.0), 'no_event': ('s5', -0.5)
    },
    's6': {'compromise_m4': ('s_terminal', 6.0), 'fail_m4': ('s6', -1.0), 'no_event': ('s6', -0.5)},
    's7': {'compromise_m3': ('s_terminal', 6.0), 'fail_m3': ('s7', -1.0), 'no_event': ('s7', -0.5)},
    's_terminal': {'no_event': ('s_terminal', -0.1)}
}

# ------------------ Behavior Audit Runner ------------------

def run_agent(agent_type, seeds, n_episodes=1000, max_steps=50):
    from gymnasium.envs.registration import register
    if agent_type == "RM":
        register(id='EnhancedTwoSubnetEnvRM-v0', entry_point='envs:EnhancedTwoSubnetEnvRM')
        env = gym.make('EnhancedTwoSubnetEnvRM-v0')
    else:
        register(id='EnhancedTwoSubnetEnv-v0', entry_point='envs:EnhancedTwoSubnetEnv')
        env = gym.make('EnhancedTwoSubnetEnv-v0')

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    os.makedirs("behavior_audit", exist_ok=True)
    per_episode_path = f"behavior_audit/{agent_type}_per_episode.csv"
    per_step_path = f"behavior_audit/{agent_type}_per_step.csv"

    f_ep, w_ep = ensure_header(per_episode_path,
        ["env","config","seed","episode","steps_in_episode","reward_env","success"])
    f_st, w_st = ensure_header(per_step_path,
        ["env","config","seed","episode","t","action","env_reward_step","events","was_compromise","target_state","info"])

    for seed in seeds:
        set_seed(seed)
        if agent_type == "RM":
            rm = RewardMachine(full_fsm)
            agent = DQNAgentRM(state_dim, len(full_fsm), action_dim)
        else:
            agent = DQNAgentShaped(state_dim, action_dim)

        for ep in range(n_episodes):
            if agent_type == "RM":
                obs, _ = env.reset()
                rm.reset()
                rm_state = rm.get_one_hot()
                state = obs.flatten().astype(np.float32)
            else:
                obs = env.reset()
                state = obs.flatten()

            cum_env_reward = 0.0
            done = False

            for t in range(max_steps):
                if agent_type == "RM":
                    action = agent.act(state, rm_state)
                    next_obs, env_reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    next_state = next_obs.flatten().astype(np.float32)
                    events = env.get_events(action)
                    rm_reward = rm.step(events)
                    total_reward = env_reward + rm_reward
                    next_rm_state = rm.get_one_hot()
                    agent.remember(state, rm_state, action, total_reward, next_state, next_rm_state, done)
                    was_comp = (("compromise" in str(events)) or ("compromise" in str(info).lower()))
                    w_st.writerow(["Advanced", "Reward Machine DQN", seed, ep+1, t+1, action, env_reward, list(events), was_comp, (info.get("target_state") if isinstance(info, dict) else None), info])
                    state = next_state; rm_state = next_rm_state
                else:
                    action = agent.act(state)
                    next_obs, env_reward, done, info = env.step(action)
                    next_state = next_obs.flatten()
                    agent.remember(state, action, env_reward, next_state, done)
                    was_comp = ("compromise" in str(info).lower())
                    w_st.writerow(["Advanced", "Shaped DQN", seed, ep+1, t+1, action, env_reward, (info.get("events") if isinstance(info, dict) else None), was_comp, (info.get("target_state") if isinstance(info, dict) else None), info])
                    state = next_state

                cum_env_reward += env_reward

                if len(agent.memory) > agent.batch_size:
                    agent.replay()

                if done or t == max_steps-1:
                    agent.update_target_model()
                    w_ep.writerow(["Advanced",
                                   "Reward Machine DQN" if agent_type=="RM" else "Shaped DQN",
                                   seed, ep+1, t+1, cum_env_reward, done])
                    break

    f_ep.close(); f_st.close()

if __name__ == "__main__":
    seeds = [6, 8, 13, 17, 21, 27, 31, 38, 43, 76]
    # You can reduce n_episodes for a quick pass, e.g., n_episodes=300
    run_agent("RM", seeds, n_episodes=1000, max_steps=50)
    run_agent("Shaped", seeds, n_episodes=1000, max_steps=50)
    print("Behavior audit logs written to behavior_audit/{RM,Shaped}_*.csv")
