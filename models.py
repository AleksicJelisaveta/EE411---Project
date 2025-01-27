import torch
import numpy as np
from collections import deque
import random

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from memory import ReplayBuffer
#from EWC_atari import ElasticWeightConsolidation

GRAD_NORM_CLIP = 50
CLIP_DELTA = 1.0
REPLAY_PERIOD = 4

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        input_shape = (input_shape[2], 84, 84)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.scale_images(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def scale_images(self, images):
        scaled_images = F.interpolate(images, size=(84, 84), mode='bilinear')
        return scaled_images

class Agent:
    def __init__(self, input_shape, num_actions, gamma=0.99, lr=0.00025, batch_size=32, memory_size=50000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, momentum=0, weight_decay=0.95)
        #self.ewc = ElasticWeightConsolidation(self.policy_net)

    def store_transition(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1, 1)  # Clip the reward to be within [-1, 1]
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        return states, actions, rewards, next_states, dones

    def update_policy(self):
        min_history = 50000
        if len(self.memory) < min_history:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        #loss = F.mse_loss(q_values, expected_q_values.detach()) + self.ewc.penalty(self.policy_net)
        loss = F.mse_loss(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-CLIP_DELTA, CLIP_DELTA)

        total_norm = 0
        for p in self.policy_net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > GRAD_NORM_CLIP:
            for p in self.policy_net.parameters():
                p.grad.data /= total_norm / GRAD_NORM_CLIP


        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.policy_net.fc2.out_features)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()