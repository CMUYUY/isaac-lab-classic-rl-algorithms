"""
SAC (Soft Actor-Critic)
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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--total-timesteps", type=int, default=100000, help="总训练步数")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
parser.add_argument("--num-envs", type=int, default=128, help="并行环境数量")
parser.add_argument("--buffer-size", type=int, default=1000000, help="重放缓冲区大小")
parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
parser.add_argument("--learning-starts", type=int, default=5000, help="开始学习的步数")
parser.add_argument("--alpha", type=float, default=0.2, help="熵正则化系数")
parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="自动调整alpha")
parser.add_argument("--seed", type=int, default=1, help="随机种子")

from distutils.util import strtobool

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("\n" + "="*70)
print("SAC - Isaac Cartpole")
print("="*70)

import isaaclab_tasks
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

# 创建环境
env_cfg = CartpoleEnvCfg()
env_cfg.scene.num_envs = args.num_envs
envs = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)

print(f"[INFO] 环境数: {args.num_envs}")
print(f"[INFO] 总训练步数: {args.total_timesteps:,}")
print(f"[INFO] 自动调整alpha: {args.autotune}")
print("="*70 + "\n")

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda")

LOG_STD_MAX = 2
LOG_STD_MIN = -5

# Actor网络(Gaussian Policy)
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.unwrapped.single_observation_space["policy"].shape).prod()
        action_dim = np.prod(env.unwrapped.single_action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        
        # Cartpole动作范围固定为[-1, 1]
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # 应用tanh变换的修正
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        # 返回detach的数据,避免计算图累积
        return (
            self.observations[indices].detach(),
            self.next_observations[indices].detach(),
            self.actions[indices].detach(),
            self.rewards[indices].detach(),
            self.dones[indices].detach(),
        )

# 初始化网络
actor = Actor(envs).to(device)
qf1 = QNetwork(envs).to(device)
qf2 = QNetwork(envs).to(device)
qf1_target = QNetwork(envs).to(device)
qf2_target = QNetwork(envs).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

# 自动调整alpha
if args.autotune:
    target_entropy = -torch.prod(torch.Tensor(envs.unwrapped.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
else:
    alpha = args.alpha

print(f"[INFO] 初始alpha: {alpha:.4f}")
if args.autotune:
    print(f"[INFO] 目标熵: {target_entropy:.2f}")

# 重放缓冲区
obs_shape = envs.unwrapped.single_observation_space["policy"].shape
action_shape = envs.unwrapped.single_action_space.shape
rb = ReplayBuffer(args.buffer_size, obs_shape, action_shape, device)

# 训练循环
obs_dict, _ = envs.reset(seed=args.seed)
obs = obs_dict["policy"].to(device)

print(f"\nSetting seed: {args.seed}")
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
        actions, _, _ = actor.get_action(obs)

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
            next_state_actions, next_state_log_pi, _ = actor.get_action(data[1])
            qf1_next_target = qf1_target(data[1], next_state_actions)
            qf2_next_target = qf2_target(data[1], next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = data[3] + (1 - data[4]) * args.gamma * min_qf_next_target

        qf1_a_values = qf1(data[0], data[2])
        qf2_a_values = qf2(data[0], data[2])
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        
        # 合并损失并更新
        q_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        q_optimizer.step()

        # 更新Actor
        pi, log_pi, _ = actor.get_action(data[0])
        qf1_pi = qf1(data[0], pi)
        qf2_pi = qf2(data[0], pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 更新alpha
        if args.autotune:
            # 重新采样动作(避免共享计算图)
            _, log_pi_alpha, _ = actor.get_action(data[0])
            alpha_loss = (-log_alpha.exp() * (log_pi_alpha + target_entropy)).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()
            alpha = log_alpha.exp().item()

        # 软更新目标网络
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    # 打印进度
    if (global_step + 1) % 5000 == 0:
        elapsed_time = time.time() - start_time
        sps = (global_step + 1) / elapsed_time
        
        if len(episode_rewards) > 0:
            print(f"Step {global_step + 1:,}/{args.total_timesteps:,} | "
                  f"SPS: {int(sps):,} | "
                  f"Alpha: {alpha:.4f} | "
                  f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                  f"Avg Length: {np.mean(episode_lengths):.1f}")
        else:
            print(f"Step {global_step + 1:,}/{args.total_timesteps:,} | SPS: {int(sps):,} | Alpha: {alpha:.4f}")

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
print(f"最终alpha: {alpha:.4f}")
print("="*70)

# 保存训练结果
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "algorithm": "SAC",
    "timestamp": timestamp,
    "total_timesteps": args.total_timesteps,
    "total_time_seconds": elapsed_time,
    "total_time_minutes": elapsed_time / 60,
    "average_sps": avg_sps,
    "final_avg_reward": float(final_reward),
    "final_avg_length": float(final_length),
    "total_episodes": len(episode_rewards),
    "final_alpha": float(alpha),
    "all_episode_rewards": [float(r) for r in episode_rewards],
    "all_episode_lengths": [float(l) for l in episode_lengths],
    "hyperparameters": {
        "learning_rate": args.learning_rate,
        "num_envs": args.num_envs,
        "buffer_size": args.buffer_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "alpha": args.alpha if not args.autotune else "auto",
    }
}

result_file = f"results/sac_{timestamp}.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {result_file}")

envs.close()
simulation_app.close()

