import argparse
import numpy as np
import torch

from envs.coop_nav_env import CooperativeNavEnv
from models.commnet import CommNetPolicy


@torch.no_grad()
def evaluate(policy: CommNetPolicy, env: CooperativeNavEnv, episodes: int = 50):
    returns = []
    successes = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = policy.forward(obs_t)
            actions = logits.argmax(dim=-1).squeeze(0).numpy()
            obs, reward, done, info = env.step(actions)
            ep_ret += reward
        returns.append(ep_ret)
        successes += int(info.get("success", False))
    return float(np.mean(returns)), successes / episodes


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved CommNet / NoComm policies.")
    parser.add_argument("--policy", type=str, choices=["commnet", "nocomm"], default="commnet")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--comm_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--vision", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CooperativeNavEnv(
        num_agents=args.agents,
        max_steps=args.max_steps,
        vision_radius=args.vision,
        seed=args.seed,
    )

    policy = CommNetPolicy(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden,
        K=args.comm_steps,
        use_comm=args.policy == "commnet",
    )

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = f"checkpoints/{args.policy}.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(state)

    mean_return, success_rate = evaluate(policy, env, episodes=args.episodes)
    print(f"{args.policy} | mean return: {mean_return:.3f} | success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()

