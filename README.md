# CommNet Reproduction (NIPS 2016)

Minimal reproduction of "Learning Multiagent Communication with Backpropagation" with environments from the paper. Implements:
- Cooperative navigation environment (`envs/coop_nav_env.py`)
- Lever-pulling coordination game (`envs/lever_game_env.py`)
- Traffic junction environment (`envs/traffic_junction_env.py`)
- Combat task environment (`envs/combat_env.py`)
- CommNet policy with continuous differentiable communication (`models/commnet.py`)
- No-communication baseline (same architecture, comm disabled)
- REINFORCE with learned baseline, training/eval scripts, plotting utilities

## Setup
Use Python 3.10+ and install dependencies (NumPy pinned <2.0 to avoid ABI issues with matplotlib):
```bash
python -m venv .venv
.venv\Scripts\activate      # or source .venv/bin/activate on Unix
pip install --upgrade pip
pip install -r requirements.txt
```

## Train
Trains CommNet and NoComm sequentially, saves checkpoints and a reward plot.
```bash
.venv/Scripts/python train.py --env nav --episodes 2000 --agents 3 --comm_steps 2 --plot_path plots/commnet_vs_nocomm.png
```
Key flags:
- `--env`: which environment (`nav` for navigation, `lever` for lever game, `traffic` for traffic junction, `combat` for combat task)
- `--episodes`: training episodes per model
- `--agents`: number of agents J
- `--comm_steps`: communication layers K
- `--max_steps`: episode horizon
- `--hidden`: hidden size
- `--vision`: vision radius for partial observations (lower -> harder, more need for comm)
- `--spawn_span`: half-width of spawn/goal box (larger -> harder, more spread)
- `--collision_penalty`: penalty per collision (nav)
- `--success_bonus`: bonus when all agents reach goal (nav)
- `--progress_scale`: scale for progress shaping (nav)
- `--distance_weight`: weight of mean distance term (nav)
- `--step_size`: move step size (nav)
- `--goal_eps`: success threshold (nav)
- `--time_penalty`: per-step penalty (nav)
- `--alpha_value`: weight for value loss (baseline)
- `--entropy_beta`: entropy bonus for exploration
- `--pool_size`: lever game pool size (number of possible agent IDs)
- `--num_levers`: lever game number of levers (defaults to agents)
- `--lever_supervised`: for lever game, use supervised optimal assignment instead of RL
- `--lever_identity_encoder`: for lever game, use identity encoder (hidden_dim=obs_dim) to preserve agent ID mask
- `--lever_batch`: supervised lever training batch size (more samples per update)
- `--comm_mlp`: use a 2-layer MLP for comm blocks (more expressive)
- `--comm_mlp_hidden`: hidden size for the comm MLP blocks
- `--lever_rank_features`: add prefix-sum rank features for the lever game (requires identity encoder)
- `--lever_oracle`: use deterministic lever oracle during evaluation (comm-only, diagnostic)
- `--no_encoder_activation`: disable encoder tanh (useful for lever)
- `--no_comm_activation`: disable comm layer tanh (useful for lever)
- `--traffic_grid_size`: grid size for traffic environment (default: 7)
- `--vision_range`: vision range for traffic environment (default: 2)
- `--spawn_mode`: spawn mode for traffic junction (`corners` or `random`, default: `corners`)
- `--combat_grid_size`: grid size for combat environment (default: 15)
- `--combat_max_steps`: combat horizon (default: 40)
- `--combat_spawn_square`: spawn square side length (default: 5; paper-parallel)
- `--combat_visual_range`: combat visual range (3x3 => 1)
- `--combat_firing_range`: combat firing range (3x3 => 1)
- `--combat_hp_max`: HP per agent (default: 3; paper-parallel)
- `--combat_cooldown_steps`: cooldown after attack (default: 1; paper-parallel)
- `--batch_size`: batch size for RL training (number of episodes per update, default: 1 for single-episode updates)
- `--resume_checkpoint`: path to checkpoint file to resume training from (optional)
- `--save_checkpoint_every`: save checkpoint every N episodes (optional, enables periodic checkpoint saving)

Outputs:
- Checkpoints: `checkpoints/commnet.pt`, `checkpoints/nocomm.pt` (final checkpoints)
- Periodic checkpoints: `checkpoints/commnet_ep{N}.pt`, `checkpoints/nocomm_ep{N}.pt` (if `--save_checkpoint_every` is set)
- Plot: `plots/commnet_vs_nocomm.png` (moving-average returns)

### Batch Training
For faster and more stable training, use batch training to collect multiple episodes before updating:
```bash
.venv/Scripts/python train.py --env nav --episodes 2000 --batch_size 8
```
This collects 8 episodes before each gradient update, averaging losses across the batch for more stable training.

### Checkpoint Resuming
Resume training from a saved checkpoint:
```bash
.venv/Scripts/python train.py --env nav --episodes 2000 --resume_checkpoint checkpoints/commnet_ep500.pt
```
The checkpoint includes policy state, optimizer state, episode count, and returns history.

### Periodic Checkpoint Saving
Save checkpoints periodically during training:
```bash
.venv/Scripts/python train.py --env nav --episodes 2000 --save_checkpoint_every 500
```
This saves checkpoints every 500 episodes to `checkpoints/commnet_ep{N}.pt` and `checkpoints/nocomm_ep{N}.pt`.

## Evaluate
Run a saved policy. By default actions are sampled; add `--greedy` to use argmax actions.
```bash
.venv/Scripts/python eval.py --env nav --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50
.venv/Scripts/python eval.py --env nav --policy nocomm --checkpoint checkpoints/nocomm.pt --episodes 50

# Lever game
.venv/Scripts/python eval.py --env lever --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50 --greedy
.venv/Scripts/python eval.py --env lever --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50

# Traffic junction
.venv/Scripts/python eval.py --env traffic --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50 --greedy

# Combat
.venv/Scripts/python eval.py --env combat --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50 --greedy --agents 5
```
Reports mean return and success rate (all agents within goal tolerance).

## Environment details
- Navigation (`envs/coop_nav_env.py`):
  - 2D square workspace, agents spawn/goals in [-0.4, 0.4]^2, discrete moves {stay, up, down, left, right} with step size 0.25 (defaults).
  - Observations per agent: own position (2), shared goal (2 by default), relative positions of other agents within vision radius (masked to zero otherwise). Default vision radius is 0.4 to make communication valuable.
  - Shared reward each step: `-distance_weight*mean_dist + progress_scale*(prev_mean_dist - mean_dist) - time_penalty - collision_penalty` + success bonus when all agents are within `goal_eps`.
  - Episode ends on success or when `max_steps` reached (default 40). Defaults are tuned to keep the task learnable while partial observability still matters.

- Lever game (`envs/lever_game_env.py`):
  - Pool of N agent IDs (pool_size); each episode samples M active agents (agents). Each active agent observes only its own ID (one-hot over pool_size).
  - Each agent chooses a lever simultaneously; reward = (# distinct levers chosen) / m (num_levers). Episode is one step.

- Traffic junction (`envs/traffic_junction_env.py`):
  - Grid-based environment where agents (cars) must coordinate to cross intersections without collisions.
  - Agents spawn at corners/edges and need to reach opposite corners/edges.
  - Observations: local grid view around agent (vision_range×vision_range) + normalized position + goal position.
  - Actions: discrete moves {stay, up, down, left, right} on grid.
  - Reward: success bonus when all agents reach goals, minus collision penalties and time penalty.
  - Collision handling: if multiple agents try to move to same cell, none move (stay in place).
  - Episode ends when all agents reach goals or max_steps reached.

- Combat (`envs/combat_env.py`):
  - Two-team grid-based combat task from the paper (model-controlled team vs hard-coded bot team).
  - Each team has m agents (use `--agents` to set m). Default paper-parallel settings: 15×15 grid, 40 steps, 5×5 spawn square.
  - Actions per model agent: {noop, move (4 dirs), attack enemy j}. Attacks succeed only if target is within firing range (default 3×3 neighborhood) and attacker is not in cooldown.
  - Bot policy (paper-parallel): attack nearest enemy if in range; else move toward nearest visible enemy (with shared vision among bots).
  - Reward shaping (paper): per-step reward proportional to negative total enemy HP; terminal penalty on lose/draw.

## Recommended runs
Navigation (moderate difficulty, fair comparison):
```bash
.venv/Scripts/python train.py --env nav --episodes 2000 --agents 3 --comm_steps 2 --hidden 64 --vision 0.3 --spawn_span 0.6 --collision_penalty 0.2 --success_bonus 20 --progress_scale 2.0 --distance_weight 0.2 --step_size 0.25 --goal_eps 0.2 --entropy_beta 0.02 --batch_size 4 --save_checkpoint_every 500 --plot_path plots/commnet_vs_nocomm.png
.venv/Scripts/python eval.py --env nav --policy commnet --checkpoint checkpoints/commnet.pt --agents 3 --hidden 64 --comm_steps 2 --vision 0.3 --spawn_span 0.6 --collision_penalty 0.2 --success_bonus 20 --progress_scale 2.0 --distance_weight 0.2 --step_size 0.25 --goal_eps 0.2
.venv/Scripts/python eval.py --env nav --policy nocomm --checkpoint checkpoints/nocomm.pt --agents 3 --hidden 64 --comm_steps 2 --vision 0.3 --spawn_span 0.6 --collision_penalty 0.2 --success_bonus 20 --progress_scale 2.0 --distance_weight 0.2 --step_size 0.25 --goal_eps 0.2
```

Lever (rank features, greedy eval):
```bash
.venv/Scripts/python train.py --env lever --agents 5 --pool_size 50 --num_levers 5 --hidden 128 --comm_steps 2 --episodes 400 --lever_supervised --lever_batch 256 --lever_identity_encoder --lever_rank_features --no_encoder_activation --comm_mlp --comm_mlp_hidden 256 --plot_path plots/lever_commnet_vs_nocomm.png
.venv/Scripts/python eval.py --env lever --policy commnet --checkpoint checkpoints/commnet.pt --agents 5 --pool_size 50 --num_levers 5 --hidden 128 --comm_steps 2 --lever_identity_encoder --lever_rank_features --no_encoder_activation --comm_mlp --comm_mlp_hidden 256 --greedy
.venv/Scripts/python eval.py --env lever --policy nocomm --checkpoint checkpoints/nocomm.pt --agents 5 --pool_size 50 --num_levers 5 --hidden 128 --comm_steps 2 --lever_identity_encoder --lever_rank_features --no_encoder_activation --comm_mlp --comm_mlp_hidden 256 --greedy
```

Traffic junction:
```bash
.venv/Scripts/python train.py --env traffic --episodes 2000 --agents 5 --traffic_grid_size 7 --vision_range 2 --max_steps 40 --collision_penalty 0.5 --success_bonus 10 --time_penalty 0.1 --spawn_mode corners --plot_path plots/traffic_commnet_vs_nocomm.png
.venv/Scripts/python eval.py --env traffic --policy commnet --checkpoint checkpoints/commnet.pt --agents 5 --traffic_grid_size 7 --vision_range 2 --max_steps 40 --collision_penalty 0.5 --success_bonus 10 --time_penalty 0.1 --spawn_mode corners --greedy
```

Combat:
```bash
.venv/Scripts/python train.py --env combat --episodes 2000 --agents 5 --combat_grid_size 15 --combat_max_steps 40 --combat_spawn_square 5 --combat_visual_range 1 --combat_firing_range 1 --combat_hp_max 3 --combat_cooldown_steps 1 --plot_path plots/combat_commnet_vs_nocomm.png
.venv/Scripts/python eval.py --env combat --policy commnet --checkpoint checkpoints/commnet.pt --agents 5 --combat_grid_size 15 --combat_max_steps 40 --combat_spawn_square 5 --combat_visual_range 1 --combat_firing_range 1 --combat_hp_max 3 --combat_cooldown_steps 1 --greedy
```

## Troubleshooting
- NumPy ABI: if you see errors like "module compiled for NumPy 1.x cannot run on NumPy 2.x", reinstall with `pip install 'numpy<2' --force-reinstall`.
- For speed, lower `--episodes`, `--agents`, or `--max_steps` during smoke testing.
