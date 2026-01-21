import argparse
import numpy as np
import torch

from envs.coop_nav_env import CooperativeNavEnv
from envs.lever_game_env import LeverGameEnv
from models.commnet import CommNetPolicy


@torch.no_grad()
def evaluate(policy: CommNetPolicy, env, episodes: int = 50, greedy: bool = True):
    returns = []
    successes = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = policy.forward(obs_t)
            if greedy:
                actions = logits.argmax(dim=-1).squeeze(0).numpy()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample().squeeze(0).numpy()
            obs, reward, done, info = env.step(actions)
            ep_ret += reward
        returns.append(ep_ret)
        successes += int(info.get("success", False))
    return float(np.mean(returns)), successes / episodes


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved CommNet / NoComm policies.")
    parser.add_argument("--env", choices=["nav", "lever"], default="nav")
    parser.add_argument("--policy", type=str, choices=["commnet", "nocomm"], default="commnet")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--comm_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--vision", type=float, default=0.75)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true", help="Use argmax actions instead of sampling")
    parser.add_argument("--lever_identity_encoder", action="store_true", help="Use identity encoder for lever game")
    parser.add_argument("--comm_mlp", action="store_true", help="Use 2-layer MLP for comm blocks")
    parser.add_argument("--comm_mlp_hidden", type=int, default=None, help="Hidden size for comm MLP blocks")
    parser.add_argument("--lever_rank_features", action="store_true", help="Add prefix-sum rank features for lever game")
    parser.add_argument("--lever_oracle", action="store_true", help="Use deterministic lever oracle (comm-only) for evaluation")
    parser.add_argument("--no_encoder_activation", action="store_true", help="Disable encoder tanh activation")
    parser.add_argument("--no_comm_activation", action="store_true", help="Disable comm layer tanh activation")
    args = parser.parse_args()

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

    policy = CommNetPolicy(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden,
        K=args.comm_steps,
        use_comm=args.policy == "commnet",
        identity_encoder=(args.env == "lever" and args.lever_identity_encoder),
        comm_mlp=args.comm_mlp,
        comm_mlp_hidden=args.comm_mlp_hidden,
        lever_rank_features=(args.env == "lever" and args.lever_rank_features),
        lever_oracle=(args.env == "lever" and args.lever_oracle),
        encoder_activation=not args.no_encoder_activation,
        comm_activation=not args.no_comm_activation,
    )

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = f"checkpoints/{args.policy}.pt"
    if not args.lever_oracle:
        state = torch.load(ckpt_path, map_location="cpu")
        policy.load_state_dict(state)

    mean_return, success_rate = evaluate(policy, env, episodes=args.episodes, greedy=args.greedy)
    print(f"{args.policy} | mean return: {mean_return:.3f} | success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()
