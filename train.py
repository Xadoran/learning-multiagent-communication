import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.coop_nav_env import CooperativeNavEnv
from envs.lever_game_env import LeverGameEnv
from models.commnet import CommNetPolicy
from utils import compute_returns, plot_rewards


def rollout_episode(env: CooperativeNavEnv, policy: CommNetPolicy, gamma: float):
    obs = env.reset()
    rewards: List[float] = []
    logprob_totals: List[torch.Tensor] = []
    baselines: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []

    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, J, obs_dim]
        logits, baseline = policy.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        actions_np = actions.squeeze(0).numpy()

        obs, reward, done, _info = env.step(actions_np)
        rewards.append(reward)
        logprob_totals.append(logprobs.sum(dim=1).squeeze(0))  # scalar
        baselines.append(baseline.squeeze(0))
        entropies.append(dist_entropy.mean().squeeze(0))

    returns = compute_returns(rewards, gamma)  # [T]
    baselines_t = torch.stack(baselines)  # [T]
    logprob_totals_t = torch.stack(logprob_totals)  # [T]
    advantages = returns - baselines_t.detach()
    if advantages.numel() > 1 and advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    entropies_t = torch.stack(entropies)  # [T]
    return returns, advantages, baselines_t, logprob_totals_t, entropies_t, float(np.sum(rewards))


def train_policy(
    episodes: int,
    env: CooperativeNavEnv,
    policy: CommNetPolicy,
    lr: float = 3e-4,
    gamma: float = 0.99,
    alpha_value: float = 0.03,
    entropy_beta: float = 0.01,
):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    episode_returns: List[float] = []

    for ep in range(episodes):
        returns, advantages, baselines, logprob_totals, entropies, ep_ret = rollout_episode(
            env, policy, gamma
        )
        policy_loss = -(logprob_totals * advantages).sum() - entropy_beta * entropies.mean()
        value_loss = mse_loss(baselines, returns)
        loss = policy_loss + alpha_value * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        episode_returns.append(ep_ret)
        if (ep + 1) % 100 == 0:
            avg_last = np.mean(episode_returns[-100:])
            print(f"Episode {ep+1}/{episodes} | avg return (last 100): {avg_last:.3f}")
    return episode_returns


def train_lever_supervised(
    episodes: int,
    env: LeverGameEnv,
    policy: CommNetPolicy,
    lr: float = 3e-4,
    entropy_beta: float = 0.01,
):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    episode_returns: List[float] = []

    for ep in range(episodes):
        obs = env.reset()
        target_np = env.optimal_actions()
        target = torch.tensor(target_np, dtype=torch.long)  # [J]
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1,J,obs_dim]

        logits, _ = policy.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean()

        loss = ce_loss(logits.view(-1, policy.action_dim), target) - entropy_beta * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Reward for optimal assignment should be 1.0
        ep_ret = len(set(target_np.tolist())) / float(env.num_levers)
        episode_returns.append(ep_ret)
        if (ep + 1) % 100 == 0:
            avg_last = np.mean(episode_returns[-100:])
            print(f"[Lever supervised] Episode {ep+1}/{episodes} | avg return (last 100): {avg_last:.3f}")
    return episode_returns


def main():
    parser = argparse.ArgumentParser(description="Train CommNet vs no-comm baseline.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--env", choices=["nav", "lever"], default="nav")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--comm_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha_value", type=float, default=0.03)
    parser.add_argument("--entropy_beta", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--vision", type=float, default=0.4)
    parser.add_argument("--spawn_span", type=float, default=0.4)
    parser.add_argument("--num_levers", type=int, default=None)
    parser.add_argument("--pool_size", type=int, default=500)
    parser.add_argument("--lever_supervised", action="store_true", help="Use supervised lever policy (optimal assignment) instead of RL")
    parser.add_argument("--lever_identity_encoder", action="store_true", help="Use identity encoder for lever game (hidden_dim=obs_dim)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_path", type=str, default="plots/commnet_vs_nocomm.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.env == "nav":
        env = CooperativeNavEnv(
            num_agents=args.agents,
            max_steps=args.max_steps,
            vision_radius=args.vision,
            spawn_span=args.spawn_span,
            seed=args.seed,
        )
    else:
        env = LeverGameEnv(
            pool_size=args.pool_size,
            active_agents=args.agents,
            num_levers=args.num_levers,
            seed=args.seed,
        )

    # Select training function
    use_supervised_lever = args.env == "lever" and args.lever_supervised
    train_fn = train_lever_supervised if use_supervised_lever else train_policy

    # CommNet
    comm_policy = CommNetPolicy(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden,
        K=args.comm_steps,
        use_comm=True,
        identity_encoder=(args.env == "lever" and args.lever_identity_encoder),
    )
    if use_supervised_lever:
        comm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=comm_policy,
            lr=args.lr,
            entropy_beta=args.entropy_beta,
        )
    else:
        comm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=comm_policy,
            lr=args.lr,
            gamma=args.gamma,
            alpha_value=args.alpha_value,
            entropy_beta=args.entropy_beta,
        )

    # No-communication baseline
    nocomm_policy = CommNetPolicy(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden,
        K=args.comm_steps,
        use_comm=False,
        identity_encoder=(args.env == "lever" and args.lever_identity_encoder),
    )
    if use_supervised_lever:
        nocomm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=nocomm_policy,
            lr=args.lr,
            entropy_beta=args.entropy_beta,
        )
    else:
        nocomm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=nocomm_policy,
            lr=args.lr,
            gamma=args.gamma,
            alpha_value=args.alpha_value,
            entropy_beta=args.entropy_beta,
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(comm_policy.state_dict(), "checkpoints/commnet.pt")
    torch.save(nocomm_policy.state_dict(), "checkpoints/nocomm.pt")

    plot_rewards(
        [
            ("CommNet", comm_returns),
            ("NoComm", nocomm_returns),
        ],
        out_path=args.plot_path,
    )
    print(f"Saved plot to {args.plot_path}")


if __name__ == "__main__":
    main()
