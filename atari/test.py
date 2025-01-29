import gymnasium as gym
import ale_py
import torch
import numpy as np
from models import Agent
import random

TARGET_UPDATE_PERIOD = 7500
NO_OP_MAX = 30

gym.register_envs(ale_py)

TASKS = [
    'ALE/Breakout-v5', 'ALE/Pong-v5', 'ALE/SpaceInvaders-v5', 
    'ALE/Enduro-v5', 'ALE/Seaquest-v5', 'ALE/BeamRider-v5', 
    'ALE/Qbert-v5', 'ALE/Assault-v5', 'ALE/RoadRunner-v5', 
    'ALE/UpNDown-v5'
]

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

        print(f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon}, Steps: {step_count}, Frames: {agent.frames_seen}, Episode Length: {episode_length}")

def evaluate(agent, envs):
    results = {}
    for task, env in envs.items():
        total_reward = 0
        for _ in range(5):  # Evaluate over 5 episodes
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state, epsilon=0.01)  # Use a low epsilon for evaluation
                state, reward, done, _, _ = env.step(action)
                total_reward += reward
        results[task] = total_reward / 5
    return results

if __name__ == "__main__":
    envs = {task: gym.make(task) for task in TASKS}
    agent = Agent(envs[TASKS[0]].observation_space.shape, envs[TASKS[0]].action_space.n, num_tasks=len(TASKS))

    for _ in range(10):  # Train for 10 segments
        task = random.choice(TASKS)
        agent.current_task = TASKS.index(task)
        env = envs[task]
        print(f"Training on task: {task}")
        train(agent, env, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.01)
        
        if agent.frames_seen > 20000000:
            states, _, _, _, _ = agent.memory.sample(len(agent.memory))
            criterion = torch.nn.MSELoss()
            agent.update_fisher(states, criterion)
        
        performance = evaluate(agent, envs)
        print(f"Performance after training on {task}: {performance}")
