from torch import nn
import torch
import numpy as np
import gym
from collections import deque
import itertools
import random

random.seed(40)

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 500000
MIN_REPLAY_SIZE = 1024
EPSILON_START = 1
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQUENCY = 1000

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # print(obs_t)
        # print("**********************")
        # print(obs_t.unsqueeze(0))
        # print("**********************")
        q_value = self(obs_t.unsqueeze(0))
        # print(q_value)
        # print("**********************")
        max_q_index = torch.argmax(q_value, dim=1)[0]
        # print(max_q_index)
        action = max_q_index.detach().item()

        return action

env = gym.make('CartPole-v0')

# Agents
online_agent = Agent(env)
target_agent = Agent(env)

# Optimizer
optimizer = torch.optim.Adam(online_agent.parameters(), lr=5e4)
target_agent.load_state_dict(online_agent.state_dict())

# Memory
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

# Initialize replay buffer with random samples
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, rew, done, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    if done:
        obs = env.reset()

# Main Loop
obs = env.reset()
episode_reward = 0
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_agent.act(obs)

    new_obs, rew, done, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    episode_reward += rew
    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0
    
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) > 195:
            while True:
                action = online_agent.act(obs)
                obs, _, done, _ = env.step(action)
                env.render()
                if done:
                    env.reset()
                    

    
    # Gradient step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obs_batch = np.asarray([t[0] for t in transitions])
    action_batch = np.asarray([t[1] for t in transitions])
    reward_batch = np.asarray([t[2] for t in transitions])
    done_batch = np.asarray([t[3] for t in transitions])
    new_obs_batch = np.asarray([t[4] for t in transitions])
    
    obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32)
    action_batch_t = torch.as_tensor(action_batch, dtype=torch.int64).unsqueeze(-1)
    reward_batch_t = torch.as_tensor(reward_batch, dtype=torch.float32).unsqueeze(-1)
    done_batch_t = torch.as_tensor(done_batch, dtype=torch.float32).unsqueeze(-1)
    new_obs_batch_t = torch.as_tensor(new_obs_batch, dtype=torch.float32)


    #  Compute Target values
    target_q_vales = target_agent(new_obs_batch_t)
    max_target_q_vales = target_q_vales.max(dim=1, keepdims=True)[0]

    targets = reward_batch_t + GAMMA * (1 - done_batch_t) * max_target_q_vales

    # Compute Loss
    q_values = online_agent(obs_batch_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=action_batch_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % TARGET_UPDATE_FREQUENCY == 0:
        target_agent.load_state_dict(online_agent.state_dict())

    if step % 1000 == 0:
        print()
        reward = np.mean(rew_buffer)
        print(f"Step: {step} and the Average Reward {reward}".format(step, reward))
