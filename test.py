import gymnasium as gym
import ale_py
import torch
import numpy as np
from models import Agent

TARGET_UPDATE_PERIOD = 7500
NO_OP_MAX = 30


gym.register_envs(ale_py)

def train(agent, env, num_episodes, epsilon_start, epsilon_end):
    epsilon = epsilon_start
    step_count = 0
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_length = 0

        no_op_actions = np.random.randint(1, NO_OP_MAX + 1)
        for _ in range(no_op_actions):
            state, _, done, _, _ = env.step(0)  # Assuming action 0 is the no-op action
            if done:
                state = env.reset()

        while not done:

            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update_policy()
            state = next_state
            total_reward += reward
            episode_length += 1
            step_count += 1

            if step_count % TARGET_UPDATE_PERIOD == 0:
                agent.update_target()

        if step_count > 50000 and step_count < 1050000:
            epsilon -= (epsilon_start - epsilon_end) / 1000000

        print(f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon}")

if __name__ == "__main__":
    env = gym.make('ALE/Breakout-v5')
    agent = Agent(env.observation_space.shape, env.action_space.n)

    train(agent, env, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.01)
