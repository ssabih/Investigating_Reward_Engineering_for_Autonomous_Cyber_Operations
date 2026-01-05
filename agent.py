import gymnasium as gym
import time
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
register(id='TwoSubnet-v0', entry_point='envs:TwoSubnetEnv',)
env = gym.make('TwoSubnet-v0')
LINE_BREAK = "-"*60
rewardList = []

for episode in range(20):
    total_steps = 0
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action for testing
        state, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        #time.sleep(1)
        # env.render()
        # print(f"Episode: {episode},Steps: {total_steps}, Action: {action}, Reward: {reward}, Done: {done}")
        # print(env.state.flatten().size)
        # print(env.action_space.n)
        if done:
            rewardList.append(total_reward)
            break
    print(LINE_BREAK)
    print(f"Episode: {episode},Steps: {total_steps}, Reward: {total_reward}, Done: {done}")

plt.plot(rewardList)
plt.show()