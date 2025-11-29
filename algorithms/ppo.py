"""
PPO (Proximal Policy Optimization)
Isaac Lab Cartpole 环境
"""

import argparse
import random
import time
import os
import json
from datetime import datetime
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Isaac Lab imports
from isaaclab.app import AppLauncher

# 添加参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--total-timesteps", type=int, default=500000, help="总训练步数")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
parser.add_argument("--num-envs", type=int, default=128, help="并行环境数量")
parser.add_argument("--num-steps", type=int, default=16, help="每次更新的步数")
parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--num-minibatches", type=int, default=8, help="minibatch数量")
parser.add_argument("--update-epochs", type=int, default=4, help="更新epoch数")
parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip系数")
parser.add_argument("--ent-coef", type=float, default=0.01, help="熵系数")
parser.add_argument("--vf-coef", type=float, default=0.5, help="价值函数系数")
parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪")
parser.add_argument("--seed", type=int, default=1, help="随机种子")

# Isaac Sim参数
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("="*70)
print("PPO - Isaac Cartpole")
print("="*70)

# 导入Isaac Lab环境
import isaaclab_tasks
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

# 创建环境配置
env_cfg = CartpoleEnvCfg()
env_cfg.scene.num_envs = args.num_envs

print(f"[INFO] 并行环境数: {env_cfg.scene.num_envs}")

# 创建环境
envs = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda")
args.num_envs = envs.unwrapped.num_envs  # 使用实际的环境数量
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
num_updates = args.total_timesteps // args.batch_size

print(f"[INFO] 实际环境数: {args.num_envs}")
print(f"[INFO] Batch size: {args.batch_size}")
print(f"[INFO] 更新次数: {num_updates}")

# PPO Agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.unwrapped.single_observation_space["policy"].shape
        act_shape = envs.unwrapped.single_action_space.shape
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(act_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(act_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# 初始化
agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# 存储
obs_storage = torch.zeros((args.num_steps, args.num_envs) + envs.unwrapped.single_observation_space["policy"].shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.unwrapped.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# Episode统计 - 手动追踪每个环境的累积奖励
episode_rewards_buffer = torch.zeros(args.num_envs, device=device)
episode_lengths_buffer = torch.zeros(args.num_envs, device=device)
completed_episode_rewards = []
completed_episode_lengths = []

# 训练循环
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=args.seed)
next_obs = next_obs["policy"]  # 获取policy观测
next_done = torch.zeros(args.num_envs).to(device)

print("\n" + "="*70)
print("开始训练...")
print("="*70)

for update in range(1, num_updates + 1):
    # 收集轨迹
    for step in range(args.num_steps):
        global_step += args.num_envs
        obs_storage[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_obs_dict, reward, terminations, truncations, infos = envs.step(action)
        next_obs = next_obs_dict["policy"]
        done = torch.logical_or(terminations, truncations)
        rewards[step] = reward.view(-1)
        next_done = done

        # 手动追踪episode统计
        episode_rewards_buffer += reward.view(-1)
        episode_lengths_buffer += 1
        
        # 检测episode结束
        for env_idx in range(args.num_envs):
            if done[env_idx]:
                completed_episode_rewards.append(episode_rewards_buffer[env_idx].item())
                completed_episode_lengths.append(episode_lengths_buffer[env_idx].item())
                episode_rewards_buffer[env_idx] = 0
                episode_lengths_buffer[env_idx] = 0

    # GAE计算
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()  # 转换为float
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1].float()  # 转换为float
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # PPO更新
    b_obs = obs_storage.reshape((-1,) + envs.unwrapped.single_observation_space["policy"].shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.unwrapped.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    b_inds = np.arange(args.batch_size)
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    # 打印进度
    if update % 5 == 0:
        sps = int(global_step / (time.time() - start_time))
        if len(completed_episode_rewards) > 0:
            print(f"Update {update}/{num_updates} | Step {global_step:,}/{args.total_timesteps:,} | "
                  f"SPS: {sps:,} | Avg Reward: {np.mean(completed_episode_rewards[-100:]):.2f} | "
                  f"Avg Length: {np.mean(completed_episode_lengths[-100:]):.1f}")
        else:
            print(f"Update {update}/{num_updates} | Step {global_step:,}/{args.total_timesteps:,} | SPS: {sps:,}")

print("\n" + "="*70)
print("训练完成!")
print("="*70)
total_time = time.time() - start_time
avg_sps = int(global_step / total_time)
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
    "algorithm": "PPO",
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
        "num_steps": args.num_steps,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_coef": args.clip_coef,
    }
}

result_file = f"results/ppo_{timestamp}.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {result_file}")

envs.close()
simulation_app.close()
