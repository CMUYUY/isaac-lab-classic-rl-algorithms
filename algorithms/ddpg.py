"""
DDPG (Deep Deterministic Policy Gradient)
Isaac Lab Cartpole 环境
"""

import argparse
import random
import time
import os
import json
from datetime import datetime
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Isaac Lab imports
from isaaclab.app import AppLauncher

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--total-timesteps", type=int, default=100000, help="总训练步数")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
parser.add_argument("--num-envs", type=int, default=128, help="并行环境数量")
parser.add_argument("--buffer-size", type=int, default=1000000, help="重放缓冲区大小")
parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
parser.add_argument("--exploration-noise", type=float, default=0.1, help="探索噪声标准差")
parser.add_argument("--learning-starts", type=int, default=5000, help="开始学习的步数")
parser.add_argument("--seed", type=int, default=1, help="随机种子")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("\n" + "="*70)
print("DDPG - Isaac Cartpole")
print("="*70)

# 导入Isaac Lab环境
import isaaclab_tasks
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

# 创建环境
env_cfg = CartpoleEnvCfg()
env_cfg.scene.num_envs = args.num_envs
envs = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)

print(f"[INFO] 环境数: {args.num_envs}")
print(f"[INFO] 总训练步数: {args.total_timesteps:,}")
print(f"[INFO] 学习开始步数: {args.learning_starts:,}")
print("="*70 + "\n")

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda")

# Actor网络
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.unwrapped.single_observation_space["policy"].shape).prod()
        action_dim = np.prod(env.unwrapped.single_action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        
        # 动作缩放参数
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

# Critic网络(Q函数)
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.unwrapped.single_observation_space["policy"].shape).prod()
        action_dim = np.prod(env.unwrapped.single_action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 重放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, obs_shape, action_shape, device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def add(self, obs, next_obs, action, reward, done):
        batch_size = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.buffer_size
        
        self.observations[indices] = obs
        self.next_observations[indices] = next_obs
        self.actions[indices] = action
        self.rewards[indices] = reward
        self.dones[indices] = done
        
        self.ptr = (self.ptr + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.next_observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
        )

# 初始化网络
actor = Actor(envs).to(device)
qf = QNetwork(envs).to(device)
qf_target = QNetwork(envs).to(device)
qf_target.load_state_dict(qf.state_dict())
actor_target = Actor(envs).to(device)
actor_target.load_state_dict(actor.state_dict())

q_optimizer = optim.Adam(qf.parameters(), lr=args.learning_rate)
actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

# 重放缓冲区
obs_shape = envs.unwrapped.single_observation_space["policy"].shape
action_shape = envs.unwrapped.single_action_space.shape
rb = ReplayBuffer(args.buffer_size, obs_shape, action_shape, device)

# 训练循环
obs_dict, _ = envs.reset(seed=args.seed)
obs = obs_dict["policy"].to(device)

print(f"Setting seed: {args.seed}")
print("开始训练...\n")

start_time = time.time()
# Episode统计 - 使用list保存所有episode
episode_rewards = []
episode_lengths = []
current_episode_reward = torch.zeros(args.num_envs, device=device)
current_episode_length = torch.zeros(args.num_envs, device=device)

for global_step in range(args.total_timesteps):
    # 选择动作
    if global_step < args.learning_starts:
        actions = torch.tensor(
            np.array([envs.unwrapped.single_action_space.sample() for _ in range(args.num_envs)]),
            dtype=torch.float32,
            device=device
        )
    else:
        with torch.no_grad():
            actions = actor(obs)
            # 添加探索噪声
            noise = torch.randn_like(actions) * (actor.action_scale * args.exploration_noise)
            actions = (actions + noise).clamp(
                torch.tensor(envs.unwrapped.single_action_space.low, device=device),
                torch.tensor(envs.unwrapped.single_action_space.high, device=device)
            )

    # 执行动作
    next_obs_dict, rewards, terminations, truncations, infos = envs.step(actions)
    next_obs = next_obs_dict["policy"].to(device)
    rewards = rewards.to(device).unsqueeze(-1)
    dones = (terminations | truncations).to(device).float().unsqueeze(-1)

    # 存储转换
    rb.add(obs, next_obs, actions, rewards, dones)
    
    # 更新统计
    current_episode_reward += rewards.squeeze(-1)
    current_episode_length += 1
    
    # 处理episode结束
    if dones.any():
        done_indices = dones.squeeze(-1).nonzero(as_tuple=True)[0]
        for idx in done_indices:
            episode_rewards.append(current_episode_reward[idx].item())
            episode_lengths.append(current_episode_length[idx].item())
            current_episode_reward[idx] = 0
            current_episode_length[idx] = 0

    obs = next_obs

    # 训练
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)
        
        # 更新Critic
        with torch.no_grad():
            next_state_actions = actor_target(data[1])
            qf_next_target = qf_target(data[1], next_state_actions)
            next_q_value = data[3] + (1 - data[4]) * args.gamma * qf_next_target

        qf_a_values = qf(data[0], data[2])
        qf_loss = nn.functional.mse_loss(qf_a_values, next_q_value)

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # 更新Actor
        actor_loss = -qf(data[0], actor(data[0])).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(qf.parameters(), qf_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    # 打印进度
    if (global_step + 1) % 5000 == 0:
        elapsed_time = time.time() - start_time
        sps = (global_step + 1) / elapsed_time
        
        if len(episode_rewards) > 0:
            print(f"Step {global_step + 1:,}/{args.total_timesteps:,} | "
                  f"SPS: {int(sps):,} | "
                  f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                  f"Avg Length: {np.mean(episode_lengths):.1f}")
        else:
            print(f"Step {global_step + 1:,}/{args.total_timesteps:,} | SPS: {int(sps):,}")

# 训练完成
elapsed_time = time.time() - start_time
avg_sps = int(args.total_timesteps/elapsed_time)
print("\n" + "="*70)
print("训练完成!")
print("="*70)
print(f"总时间: {elapsed_time/60:.1f} 分钟")
print(f"平均SPS: {avg_sps:,}")
if len(episode_rewards) > 0:
    final_reward = np.mean(episode_rewards)
    final_length = np.mean(episode_lengths)
    print(f"最终平均奖励: {final_reward:.2f}")
    print(f"最终平均长度: {final_length:.1f}")
    print(f"总episode数: {len(episode_rewards)}")
else:
    final_reward = 0.0
    final_length = 0.0
print("="*70)

# 保存训练结果
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "algorithm": "DDPG",
    "timestamp": timestamp,
    "total_timesteps": args.total_timesteps,
    "total_time_seconds": elapsed_time,
    "total_time_minutes": elapsed_time / 60,
    "average_sps": avg_sps,
    "final_avg_reward": float(final_reward),
    "final_avg_length": float(final_length),
    "total_episodes": len(episode_rewards),
    "all_episode_rewards": [float(r) for r in episode_rewards],
    "all_episode_lengths": [float(l) for l in episode_lengths],
    "hyperparameters": {
        "learning_rate": args.learning_rate,
        "num_envs": args.num_envs,
        "buffer_size": args.buffer_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "exploration_noise": args.exploration_noise,
    }
}

result_file = f"results/ddpg_{timestamp}.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {result_file}")

envs.close()
simulation_app.close()

