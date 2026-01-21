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
    batch_size: int = 256,
):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    episode_returns: List[float] = []

    pool_size = env.pool_size
    active_agents = env.active_agents
    num_levers = env.num_levers

    for ep in range(episodes):
        # Sample a batch of episodes by drawing active IDs without replacement
        scores = env.rng.random((batch_size, pool_size))
        ids = np.argsort(scores, axis=1)[:, :active_agents]  # [B, J]

        obs = np.zeros((batch_size, active_agents, pool_size), dtype=np.float32)
        batch_idx = np.arange(batch_size)[:, None]
        agent_idx = np.arange(active_agents)[None, :]
        obs[batch_idx, agent_idx, ids] = 1.0

        # Optimal assignment: sort IDs and assign ranks to levers
        order = np.argsort(ids, axis=1)  # [B, J]
        targets = np.zeros((batch_size, active_agents), dtype=np.int64)
        for b in range(batch_size):
            for rank, idx in enumerate(order[b]):
                targets[b, idx] = rank % num_levers

        obs_t = torch.tensor(obs, dtype=torch.float32)
        target_t = torch.tensor(targets, dtype=torch.long)

        logits, _ = policy.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean()
        loss = ce_loss(logits.view(-1, policy.action_dim), target_t.view(-1)) - entropy_beta * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Perfect assignment yields 1.0 reward
        episode_returns.append(1.0)
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
    parser.add_argument("--collision_penalty", type=float, default=0.2)
    parser.add_argument("--success_bonus", type=float, default=20.0)
    parser.add_argument("--progress_scale", type=float, default=2.0)
    parser.add_argument("--step_size", type=float, default=0.25)
    parser.add_argument("--goal_eps", type=float, default=0.2)
    parser.add_argument("--time_penalty", type=float, default=0.0)
    parser.add_argument("--distance_weight", type=float, default=1.0)
    parser.add_argument("--num_levers", type=int, default=None)
    parser.add_argument("--pool_size", type=int, default=500)
    parser.add_argument("--lever_supervised", action="store_true", help="Use supervised lever policy (optimal assignment) instead of RL")
    parser.add_argument("--lever_identity_encoder", action="store_true", help="Use identity encoder for lever game (hidden_dim=obs_dim)")
    parser.add_argument("--lever_batch", type=int, default=256, help="Batch size for supervised lever training")
    parser.add_argument("--comm_mlp", action="store_true", help="Use 2-layer MLP for comm blocks (more expressive)")
    parser.add_argument("--comm_mlp_hidden", type=int, default=None, help="Hidden size for comm MLP blocks")
    parser.add_argument("--lever_rank_features", action="store_true", help="Add prefix-sum rank features for lever game (requires identity encoder)")
    parser.add_argument("--no_encoder_activation", action="store_true", help="Disable encoder tanh activation (useful for lever)")
    parser.add_argument("--no_comm_activation", action="store_true", help="Disable comm layer tanh activation (useful for lever)")
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
            collision_penalty=args.collision_penalty,
            success_bonus=args.success_bonus,
            progress_scale=args.progress_scale,
            step_size=args.step_size,
            goal_eps=args.goal_eps,
            time_penalty=args.time_penalty,
            distance_weight=args.distance_weight,
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
        comm_mlp=args.comm_mlp,
        comm_mlp_hidden=args.comm_mlp_hidden,
        lever_rank_features=(args.env == "lever" and args.lever_rank_features),
        encoder_activation=not args.no_encoder_activation,
        comm_activation=not args.no_comm_activation,
    )
    if use_supervised_lever:
        comm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=comm_policy,
            lr=args.lr,
            entropy_beta=args.entropy_beta,
            batch_size=args.lever_batch,
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
        comm_mlp=args.comm_mlp,
        comm_mlp_hidden=args.comm_mlp_hidden,
        lever_rank_features=(args.env == "lever" and args.lever_rank_features),
        encoder_activation=not args.no_encoder_activation,
        comm_activation=not args.no_comm_activation,
    )
    if use_supervised_lever:
        nocomm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=nocomm_policy,
            lr=args.lr,
            entropy_beta=args.entropy_beta,
            batch_size=args.lever_batch,
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
