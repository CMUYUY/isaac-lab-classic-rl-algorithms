"""
TD3 (Twin Delayed DDPG)
Isaac Lab Cartpole 环境
"""

import argparse
import random
import time
import os
import json
from datetime import datetime
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Isaac Lab imports
from isaaclab.app import AppLauncher

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1, help="随机种子")
parser.add_argument("--total-timesteps", type=int, default=100000, help="总训练步数")
parser.add_argument("--num-envs", type=int, default=128, help="并行环境数量")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
parser.add_argument("--buffer-size", type=int, default=100000, help="回放缓冲区大小")
parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
parser.add_argument("--policy-noise", type=float, default=0.2, help="策略噪声")
parser.add_argument("--exploration-noise", type=float, default=0.1, help="探索噪声")
parser.add_argument("--learning-starts", type=int, default=5000, help="开始学习的步数")
parser.add_argument("--policy-frequency", type=int, default=2, help="策略更新频率")
parser.add_argument("--noise-clip", type=float, default=0.5, help="噪声裁剪")

# 启动Isaac Sim
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入 Isaac Lab 任务模块  
import isaaclab_tasks
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

print("\n" + "="*70)
print(f"TD3 - Isaac Cartpole")
print("="*70)
print(f"[INFO] 并行环境数: {args.num_envs}")
print(f"[INFO] 总训练步数: {args.total_timesteps:,}")
print("="*70 + "\n")

# 创建环境
env_cfg = CartpoleEnvCfg()
env_cfg.scene.num_envs = args.num_envs
envs = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)

print(f"[INFO] 环境已创建")
print(f"   观测空间: {envs.unwrapped.single_observation_space['policy'].shape}")
print(f"   动作空间: {envs.unwrapped.single_action_space.shape}\n")

# Q网络
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

# Actor网络
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.unwrapped.single_observation_space["policy"].shape).prod()
        action_dim = np.prod(env.unwrapped.single_action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        
        # Cartpole动作范围固定为[-1, 1]
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

# 简化回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, obs_shape, action_shape, device):
        self.buffer_size = buffer_size
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
qf1 = QNetwork(envs).to(device)
qf2 = QNetwork(envs).to(device)
qf1_target = QNetwork(envs).to(device)
qf2_target = QNetwork(envs).to(device)
target_actor = Actor(envs).to(device)
target_actor.load_state_dict(actor.state_dict())
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)

# 回放缓冲区
obs_shape = envs.unwrapped.single_observation_space["policy"].shape
action_shape = envs.unwrapped.single_action_space.shape
rb = ReplayBuffer(args.buffer_size, obs_shape, action_shape, device)

# Episode统计 - 手动追踪
episode_rewards_buffer = torch.zeros(args.num_envs, device=device)
episode_lengths_buffer = torch.zeros(args.num_envs, device=device)
completed_episode_rewards = []
completed_episode_lengths = []

# 训练
start_time = time.time()
obs, _ = envs.reset(seed=args.seed)
obs = obs["policy"].to(device)

print("开始训练...\n")

for global_step in range(args.total_timesteps):
    # 探索或利用
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
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    next_obs = next_obs["policy"].to(device)
    rewards_tensor = rewards.unsqueeze(-1)
    dones = torch.logical_or(terminations, truncations).unsqueeze(-1).float()

    # 手动追踪episode统计
    episode_rewards_buffer += rewards
    episode_lengths_buffer += 1
    
    # 检测episode结束
    done_mask = torch.logical_or(terminations, truncations)
    for env_idx in range(args.num_envs):
        if done_mask[env_idx]:
            completed_episode_rewards.append(episode_rewards_buffer[env_idx].item())
            completed_episode_lengths.append(episode_lengths_buffer[env_idx].item())
            episode_rewards_buffer[env_idx] = 0
            episode_lengths_buffer[env_idx] = 0

    # 存储
    rb.add(obs, next_obs, actions, rewards_tensor, dones)
    obs = next_obs

    # 训练
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)
        
        with torch.no_grad():
            clipped_noise = (torch.randn_like(data[2]) * args.policy_noise).clamp(
                -args.noise_clip, args.noise_clip
            ) * actor.action_scale
            
            next_state_actions = (target_actor(data[1]) + clipped_noise).clamp(
                torch.tensor(envs.unwrapped.single_action_space.low, device=device),
                torch.tensor(envs.unwrapped.single_action_space.high, device=device)
            )
            qf1_next_target = qf1_target(data[1], next_state_actions)
            qf2_next_target = qf2_target(data[1], next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data[3] + (1 - data[4]) * args.gamma * min_qf_next_target

        qf1_a_values = qf1(data[0], data[2])
        qf2_a_values = qf2(data[0], data[2])
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:
            actor_loss = -qf1(data[0], actor(data[0])).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # 软更新
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    if global_step % 5000 == 0 and global_step > 0:
        sps = int(global_step / (time.time() - start_time))
        if len(completed_episode_rewards) > 0:
            print(f"Step {global_step:,}/{args.total_timesteps:,} | SPS: {sps:,} | "
                  f"Avg Reward: {np.mean(completed_episode_rewards[-100:]):.2f} | "
                  f"Avg Length: {np.mean(completed_episode_lengths[-100:]):.1f}")
        else:
            print(f"Step {global_step:,}/{args.total_timesteps:,} | SPS: {sps:,}")

print("\n" + "="*70)
print("训练完成!")
print("="*70)
total_time = time.time() - start_time
avg_sps = int(args.total_timesteps / total_time)
print(f"总时间: {total_time/60:.1f} 分钟")
print(f"平均SPS: {avg_sps:,}")
if len(completed_episode_rewards) > 0:
    final_reward = np.mean(completed_episode_rewards[-100:])
    final_length = np.mean(completed_episode_lengths[-100:])
    print(f"最终平均奖励: {final_reward:.2f}")
    print(f"最终平均长度: {final_length:.1f}")
    print(f"总episode数: {len(completed_episode_rewards)}")
else:
    final_reward = 0.0
    final_length = 0.0
print("="*70)

# 保存训练结果
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "algorithm": "TD3",
    "timestamp": timestamp,
    "total_timesteps": args.total_timesteps,
    "total_time_seconds": total_time,
    "total_time_minutes": total_time / 60,
    "average_sps": avg_sps,
    "final_avg_reward": float(final_reward),
    "final_avg_length": float(final_length),
    "total_episodes": len(completed_episode_rewards),
    "all_episode_rewards": [float(r) for r in completed_episode_rewards],
    "all_episode_lengths": [float(l) for l in completed_episode_lengths],
    "hyperparameters": {
        "learning_rate": args.learning_rate,
        "num_envs": args.num_envs,
        "buffer_size": args.buffer_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "exploration_noise": args.exploration_noise,
    }
}

result_file = f"results/td3_{timestamp}.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {result_file}")

envs.close()
simulation_app.close()
