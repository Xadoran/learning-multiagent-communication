# Learning Multiagent Communication (CommNet) — NIPS 2016 reproduction

Minimal (but practical) reproduction of **“Learning Multiagent Communication with Backpropagation”** (CommNet) with a small CommNet policy, a no-communication ablation, and several paper-style environments.

This repo includes:

- **Environments**: `nav` (`envs/coop_nav_env.py`), `lever` (`envs/lever_game_env.py`), `traffic` (`envs/traffic_junction_env.py`), `combat` (`envs/combat_env.py`)
- **Models**: CommNet policy + value baseline (`models/commnet.py`), with a **NoComm** baseline via `use_comm=False`
- **Training**: REINFORCE + learned baseline (`train.py`) with optional batch updates and checkpoint resume
- **Evaluation**: episode return + success rate (`eval.py`)
- **Experiment logging**: run folders with `run_config.json` + `metrics.csv` under `runs/` (`analysis/logging_utils.py`)
- **Analysis/reporting**: `analysis/commnet_results.ipynb`, plots under `plots/`, and the writeup in `iar-final-report.tex` / `iar_final_report.pdf`

## Setup

Use **Python 3.10+**. NumPy is pinned `<2.0` to avoid ABI issues with some matplotlib builds.

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quickstart

Train CommNet and NoComm on cooperative navigation, then evaluate both:

```bash
python3 train.py --env nav --episodes 2000 --agents 3 --comm_steps 2 --plot_path plots/nav_commnet_vs_nocomm.png
python3 eval.py  --env nav --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50 --greedy
python3 eval.py  --env nav --policy nocomm  --checkpoint checkpoints/nocomm.pt  --episodes 50 --greedy
```

## Training (`train.py`)

`train.py` trains **CommNet** and **NoComm** sequentially (and can repeat that for multiple seeds).

```bash
python3 train.py --env nav --episodes 2000 --agents 3 --comm_steps 2
```

### Common flags

- **`--env`**: `nav | lever | traffic | combat`
- **`--episodes`**: episodes per model
- **`--agents`**: number of agents \(J\) (also used as `nmax` in traffic, and as \(m\) per team in combat)
- **`--hidden`**: hidden size (ignored if `--lever_identity_encoder` is used)
- **`--comm_steps`**: number of comm blocks \(K\)
- **`--batch_size`**: RL batch size (episodes per gradient update; RL envs only)
- **`--resume_checkpoint`**: resume from a **training checkpoint** (see “Checkpoints” below)
- **`--save_checkpoint_every`**: periodically write training checkpoints
- **`--seeds`**: comma-separated seeds (e.g. `0,1,2,3,4`) to run multiple independent trainings

### Outputs

- **Final “legacy” checkpoints (single-seed only)**:
  - `checkpoints/commnet.pt` (PyTorch `state_dict`)
  - `checkpoints/nocomm.pt` (PyTorch `state_dict`)
- **Training checkpoints (when `--save_checkpoint_every` is set)**:
  - `checkpoints/commnet_ep{N}.pt`, `checkpoints/nocomm_ep{N}.pt` (dict containing policy + optimizer + history)
- **Run logs (always)**:
  - `runs/{env}/{model}/{run_id}/run_config.json`
  - `runs/{env}/{model}/{run_id}/metrics.csv`
- **Reward plot (single-seed only)**: `--plot_path` (moving-average returns)

### Checkpoints: `state_dict` vs training checkpoint

- **`checkpoints/commnet.pt` / `checkpoints/nocomm.pt`** are _only_ model weights (usable for `eval.py`).
- **`checkpoints/*_ep{N}.pt`** are _training checkpoints_ (policy + optimizer + episode counter + return history) and are what `--resume_checkpoint` expects.

## Evaluation (`eval.py`)

Evaluate a saved policy checkpoint and report **mean return** and **success rate**:

```bash
python3 eval.py --env nav --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50 --greedy
```

Notes:

- If you omit `--checkpoint`, `eval.py` defaults to `checkpoints/{policy}.pt`.
- `--greedy` uses argmax actions; without it, actions are sampled.

## Environments

### Navigation (`envs/coop_nav_env.py`)

- **Actions**: discrete \(\{ \text{stay, up, down, left, right} \}\)
- **Observations**: own pos (2) + goal (2) + masked relative positions of other agents
- **Partial observability**: controlled by `--vision` (smaller => harder, more need for comm)
- **Success**: all agents within `--goal_eps` of goal

### Lever (`envs/lever_game_env.py`)

- **One-step coordination game**: each agent observes only its ID (one-hot over `--pool_size`)
- **Reward**: `distinct_levers / num_levers`
- Optional **paper-style supervised training**: `--lever_supervised` (uses the deterministic optimal assignment)

### Traffic Junction (`envs/traffic_junction_env.py`) _(paper-parallel)_

Implements the CommNet traffic task in a **fixed-slot** form:

- **Grid**: `--traffic_grid_size` (paper default is 14)
- **Slots**: `nmax == --agents` (inactive slots emit zero observations)
- **Actions**: `BRAKE` (0) or `GAS` (1)
- **Arrivals**: each direction spawns a new car with probability `--traffic_p_arrive` (capped by `nmax`)
- **Reward** (paper-style): \(r(t) = C*t \cdot r*{coll} + \sum*i \tau_i \cdot r*{time}\)
- **Success**: after horizon, if **no collision occurred at any time**

### Combat (`envs/combat_env.py`) _(paper-parallel)_

- **Two teams**, model-controlled vs hard-coded bot policy
- **Actions**: noop + 4 moves + \(m\) attack actions
- **Reward**: \(-0.1 \cdot\) (total enemy HP), plus terminal \(-1\) on lose/draw

## Recommended runs

### Navigation (moderate difficulty)

```bash
python3 train.py --env nav --episodes 2000 --agents 3 --comm_steps 2 --hidden 64 --vision 0.3 --spawn_span 0.6 --collision_penalty 0.2 --success_bonus 20 --progress_scale 2.0 --distance_weight 0.2 --step_size 0.25 --goal_eps 0.2 --entropy_beta 0.02 --batch_size 4 --save_checkpoint_every 500 --plot_path plots/nav_commnet_vs_nocomm.png
python3 eval.py  --env nav --policy commnet --checkpoint checkpoints/commnet.pt --agents 3 --hidden 64 --comm_steps 2 --vision 0.3 --spawn_span 0.6 --collision_penalty 0.2 --success_bonus 20 --progress_scale 2.0 --distance_weight 0.2 --step_size 0.25 --goal_eps 0.2 --greedy
python3 eval.py  --env nav --policy nocomm  --checkpoint checkpoints/nocomm.pt  --agents 3 --hidden 64 --comm_steps 2 --vision 0.3 --spawn_span 0.6 --collision_penalty 0.2 --success_bonus 20 --progress_scale 2.0 --distance_weight 0.2 --step_size 0.25 --goal_eps 0.2 --greedy
```

### Lever (supervised + rank features, greedy eval)

```bash
python3 train.py --env lever --agents 5 --pool_size 50 --num_levers 5 --hidden 128 --comm_steps 2 --episodes 400 --lever_supervised --lever_batch 256 --lever_identity_encoder --lever_rank_features --no_encoder_activation --comm_mlp --comm_mlp_hidden 256 --plot_path plots/lever_commnet_vs_nocomm.png
python3 eval.py  --env lever --policy commnet --checkpoint checkpoints/commnet.pt --agents 5 --pool_size 50 --num_levers 5 --hidden 128 --comm_steps 2 --lever_identity_encoder --lever_rank_features --no_encoder_activation --comm_mlp --comm_mlp_hidden 256 --greedy
python3 eval.py  --env lever --policy nocomm  --checkpoint checkpoints/nocomm.pt  --agents 5 --pool_size 50 --num_levers 5 --hidden 128 --comm_steps 2 --lever_identity_encoder --lever_rank_features --no_encoder_activation --comm_mlp --comm_mlp_hidden 256 --greedy
```

### Traffic Junction (paper-parallel defaults)

```bash
python3 train.py --env traffic --episodes 2000 --agents 10 --traffic_grid_size 14 --max_steps 40 --traffic_p_arrive 0.2 --traffic_r_coll -10.0 --traffic_r_time -0.01 --vision_range 1 --plot_path plots/traffic_commnet_vs_nocomm.png
python3 eval.py  --env traffic --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50 --agents 10 --traffic_grid_size 14 --max_steps 40 --traffic_p_arrive 0.2 --traffic_r_coll -10.0 --traffic_r_time -0.01 --vision_range 1 --greedy
```

### Combat (paper-parallel defaults)

```bash
python3 train.py --env combat --episodes 2000 --agents 5 --combat_grid_size 15 --combat_max_steps 40 --combat_spawn_square 5 --combat_visual_range 1 --combat_firing_range 1 --combat_hp_max 3 --combat_cooldown_steps 1 --plot_path plots/combat_commnet_vs_nocomm.png
python3 eval.py  --env combat --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50 --agents 5 --combat_grid_size 15 --combat_max_steps 40 --combat_spawn_square 5 --combat_visual_range 1 --combat_firing_range 1 --combat_hp_max 3 --combat_cooldown_steps 1 --greedy
```

## Tests

The tests are runnable as plain Python scripts (no pytest dependency required):

```bash
python3 tests/test_coop_nav_env.py
python3 tests/test_lever_game_env.py
python3 tests/test_traffic_junction_env.py
python3 tests/test_combat_env.py
```
