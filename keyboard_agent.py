import gymnasium as gym
from gymnasium.envs.registration import register
register(id='TwoSubnet-v0', entry_point='envs:TwoSubnetEnv',)
env = gym.make('TwoSubnet-v0')

LINE_BREAK = "-"*60
LINE_BREAK2 = "="*60


def choose_action(env):
    print("CHOOSE ACTION")
    print(LINE_BREAK2)
    idx = int(input("Choose action number: "))
    return idx

def run_keyboard_agent(env):
  
    print(LINE_BREAK2)
    print("STARTING EPISODE")
    print(LINE_BREAK2)

    o = env.reset()
    env.render()
    total_steps = 0
    total_reward = 0
    done = False
    step_limit_reached = False
    while not done and not step_limit_reached:
        a = choose_action(env)
        o, r, done, _ = env.step(a)
        total_reward += r
        total_steps += 1
        print("\n" + LINE_BREAK2)
        print("OBSERVATION RECIEVED")
        print(LINE_BREAK2)
        env.render()
        print(f"Reward={r}")
        print(f"Done={done}")
        print(f"Step limit reached={step_limit_reached}")
        print(LINE_BREAK)
    return total_reward, total_steps, done


if __name__ == "__main__":
    total_reward, steps, goal = run_keyboard_agent(env)
    print(LINE_BREAK2)
    print("EPISODE FINISHED")
    print(LINE_BREAK)
    print(f"Goal reached = {goal}")
    print(f"Total reward = {total_reward}")
    print(f"Steps taken = {steps}")


