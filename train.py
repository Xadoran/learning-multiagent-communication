import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.coop_nav_env import CooperativeNavEnv
from envs.lever_game_env import LeverGameEnv
from envs.traffic_junction_env import TrafficJunctionEnv
from envs.predator_prey_env import PredatorPreyEnv
from models.commnet import CommNetPolicy
from utils import compute_returns, plot_rewards


def rollout_episode(env, policy: CommNetPolicy, gamma: float):
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
    env,
    policy: CommNetPolicy,
    lr: float = 3e-4,
    gamma: float = 0.99,
    alpha_value: float = 0.03,
    entropy_beta: float = 0.01,
    batch_size: int = 1,
    checkpoint_path: str | None = None,
    save_checkpoint_every: int | None = None,
    checkpoint_prefix: str = "checkpoint",
):
    """
    Train policy using REINFORCE with baseline.
    
    Args:
        episodes: Total number of episodes to train
        env: Environment instance
        policy: CommNet policy network
        lr: Learning rate
        gamma: Discount factor
        alpha_value: Weight for value loss
        entropy_beta: Entropy bonus coefficient
        batch_size: Number of episodes to collect before updating (default: 1 for single-episode updates)
        checkpoint_path: Path to checkpoint file to resume from (optional)
        save_checkpoint_every: Save checkpoint every N episodes (optional)
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    episode_returns: List[float] = []
    start_episode = 0

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint.get("episode", 0)
        episode_returns = checkpoint.get("episode_returns", [])
        print(f"Resumed from episode {start_episode}, {len(episode_returns)} episodes in history")

    for ep in range(start_episode, episodes):
        # Collect batch of episodes
        batch_returns = []
        batch_advantages = []
        batch_baselines = []
        batch_logprob_totals = []
        batch_entropies = []
        batch_ep_returns = []

        for _ in range(batch_size):
            returns, advantages, baselines, logprob_totals, entropies, ep_ret = rollout_episode(
                env, policy, gamma
            )
            batch_returns.append(returns)
            batch_advantages.append(advantages)
            batch_baselines.append(baselines)
            batch_logprob_totals.append(logprob_totals)
            batch_entropies.append(entropies)
            batch_ep_returns.append(ep_ret)

        # Aggregate losses across batch
        # Average policy loss across episodes
        policy_losses = []
        value_losses = []
        for i in range(batch_size):
            policy_loss = -(batch_logprob_totals[i] * batch_advantages[i]).sum() - entropy_beta * batch_entropies[i].mean()
            value_loss = mse_loss(batch_baselines[i], batch_returns[i])
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        # Average losses across batch
        total_policy_loss = sum(policy_losses) / batch_size
        total_value_loss = sum(value_losses) / batch_size
        loss = total_policy_loss + alpha_value * total_value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Record all episode returns
        episode_returns.extend(batch_ep_returns)
        
        if (ep + 1) % max(1, 100 // batch_size) == 0:
            avg_last = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            total_episodes = len(episode_returns)
            print(f"Episode {ep+1}/{episodes} (total: {total_episodes}) | avg return (last 100): {avg_last:.3f} | batch_size: {batch_size}")
        
        # Save checkpoint periodically if requested
        if save_checkpoint_every and (ep + 1) % save_checkpoint_every == 0:
            checkpoint = {
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": ep + 1,
                "episode_returns": episode_returns,
            }
            if checkpoint_path:
                # If resuming from checkpoint, save to same path
                checkpoint_file = checkpoint_path
            else:
                # Otherwise, save with episode number
                checkpoint_file = f"checkpoints/{checkpoint_prefix}_ep{ep+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_file) if os.path.dirname(checkpoint_file) else ".", exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
            print(f"Saved checkpoint to {checkpoint_file} at episode {ep+1}")
    
    return episode_returns


def train_lever_supervised(
    episodes: int,
    env: LeverGameEnv,
    policy: CommNetPolicy,
    lr: float = 3e-4,
    entropy_beta: float = 0.01,
    batch_size: int = 256,
    checkpoint_path: str | None = None,
    save_checkpoint_every: int | None = None,
    checkpoint_prefix: str = "checkpoint",
):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    episode_returns: List[float] = []
    start_episode = 0

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint.get("episode", 0)
        episode_returns = checkpoint.get("episode_returns", [])
        print(f"Resumed from episode {start_episode}, {len(episode_returns)} episodes in history")

    pool_size = env.pool_size
    active_agents = env.active_agents
    num_levers = env.num_levers

    for ep in range(start_episode, episodes):
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
        
        # Save checkpoint periodically if requested
        if save_checkpoint_every and (ep + 1) % save_checkpoint_every == 0:
            checkpoint = {
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": ep + 1,
                "episode_returns": episode_returns,
            }
            if checkpoint_path:
                # If resuming from checkpoint, save to same path
                checkpoint_file = checkpoint_path
            else:
                # Otherwise, save with episode number
                checkpoint_file = f"checkpoints/{checkpoint_prefix}_ep{ep+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_file) if os.path.dirname(checkpoint_file) else ".", exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
            print(f"Saved checkpoint to {checkpoint_file} at episode {ep+1}")
    
    return episode_returns


def main():
    parser = argparse.ArgumentParser(description="Train CommNet vs no-comm baseline.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--env", choices=["nav", "lever", "traffic", "prey"], default="nav")
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
    # Traffic junction args
    parser.add_argument("--grid_size", type=int, default=7, help="Grid size for traffic/prey environments")
    parser.add_argument("--vision_range", type=int, default=2, help="Vision range for traffic/prey environments")
    parser.add_argument("--spawn_mode", type=str, default="corners", choices=["corners", "random"], help="Spawn mode for traffic junction")
    # Predator-prey args
    parser.add_argument("--num_prey", type=int, default=1, help="Number of prey in predator-prey environment")
    parser.add_argument("--catch_radius", type=float, default=1.0, help="Catch radius for predator-prey")
    parser.add_argument("--prey_move_prob", type=float, default=0.5, help="Probability prey moves each step")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for RL training (number of episodes per update, default: 1)")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--save_checkpoint_every", type=int, default=None, help="Save checkpoint every N episodes (optional)")
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
    elif args.env == "lever":
        env = LeverGameEnv(
            pool_size=args.pool_size,
            active_agents=args.agents,
            num_levers=args.num_levers,
            seed=args.seed,
        )
    elif args.env == "traffic":
        env = TrafficJunctionEnv(
            grid_size=args.grid_size,
            num_agents=args.agents,
            max_steps=args.max_steps,
            vision_range=args.vision_range,
            collision_penalty=args.collision_penalty,
            success_bonus=args.success_bonus,
            time_penalty=args.time_penalty,
            spawn_mode=args.spawn_mode,
            seed=args.seed,
        )
    elif args.env == "prey":
        env = PredatorPreyEnv(
            grid_size=args.grid_size,
            num_predators=args.agents,
            num_prey=args.num_prey,
            max_steps=args.max_steps,
            vision_range=args.vision_range,
            catch_radius=args.catch_radius,
            success_bonus=args.success_bonus,
            time_penalty=args.time_penalty,
            prey_move_prob=args.prey_move_prob,
            collision_penalty=args.collision_penalty,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

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
    
    # Determine checkpoint paths
    comm_checkpoint = args.resume_checkpoint if args.resume_checkpoint else None
    if comm_checkpoint and not os.path.exists(comm_checkpoint):
        print(f"Warning: Checkpoint {comm_checkpoint} not found, starting from scratch")
        comm_checkpoint = None
    
    if use_supervised_lever:
        comm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=comm_policy,
            lr=args.lr,
            entropy_beta=args.entropy_beta,
            batch_size=args.lever_batch,
            checkpoint_path=comm_checkpoint,
            save_checkpoint_every=args.save_checkpoint_every,
            checkpoint_prefix="commnet",
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
            batch_size=args.batch_size,
            checkpoint_path=comm_checkpoint,
            save_checkpoint_every=args.save_checkpoint_every,
            checkpoint_prefix="commnet",
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
    
    # For nocomm, use separate checkpoint path if provided (or None)
    nocomm_checkpoint = None
    if args.resume_checkpoint:
        # Try to find nocomm checkpoint by replacing commnet with nocomm in path
        nocomm_checkpoint = args.resume_checkpoint.replace("commnet", "nocomm")
        if not os.path.exists(nocomm_checkpoint):
            nocomm_checkpoint = None
    
    if use_supervised_lever:
        nocomm_returns = train_fn(
            episodes=args.episodes,
            env=env,
            policy=nocomm_policy,
            lr=args.lr,
            entropy_beta=args.entropy_beta,
            batch_size=args.lever_batch,
            checkpoint_path=nocomm_checkpoint,
            save_checkpoint_every=args.save_checkpoint_every,
            checkpoint_prefix="nocomm",
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
            batch_size=args.batch_size,
            checkpoint_path=nocomm_checkpoint,
            save_checkpoint_every=args.save_checkpoint_every,
            checkpoint_prefix="nocomm",
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
