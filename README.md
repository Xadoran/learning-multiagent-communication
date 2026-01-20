# CommNet Reproduction (NIPS 2016)

Minimal reproduction of "Learning Multiagent Communication with Backpropagation" using a simple cooperative navigation task. Implements:
- Custom cooperative navigation environment (`envs/coop_nav_env.py`)
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
python train.py --episodes 2000 --agents 3 --comm_steps 2 --plot_path plots/commnet_vs_nocomm.png
```
Key flags:
- `--episodes`: training episodes per model
- `--agents`: number of agents J
- `--comm_steps`: communication layers K
- `--max_steps`: episode horizon
- `--hidden`: hidden size
- `--vision`: vision radius for partial observations (lower -> harder, more need for comm)
- `--spawn_span`: half-width of spawn/goal box (larger -> harder, more spread)
- `--alpha_value`: weight for value loss (baseline)
- `--entropy_beta`: entropy bonus for exploration

Outputs:
- Checkpoints: `checkpoints/commnet.pt`, `checkpoints/nocomm.pt`
- Plot: `plots/commnet_vs_nocomm.png` (moving-average returns)

## Evaluate
Run a saved policy (greedy actions by argmax):
```bash
python eval.py --policy commnet --checkpoint checkpoints/commnet.pt --episodes 50
python eval.py --policy nocomm --checkpoint checkpoints/nocomm.pt --episodes 50
```
Reports mean return and success rate (all agents within goal tolerance).

## Environment details
- 2D square workspace, agents spawn/goals in [-0.4, 0.4]^2, discrete moves {stay, up, down, left, right} with step size 0.25 (defaults).
- Observations per agent: own position (2), shared goal (2 by default), relative positions of other agents within vision radius (masked to zero otherwise). Default vision radius is 0.4 to make communication valuable.
- Shared reward each step: `progress_scale*(prev_mean_dist - mean_dist) - time_penalty - collision_penalty` + success bonus when all agents are within `goal_eps`.
- Episode ends on success or when `max_steps` reached (default 40). Defaults are tuned to keep the task learnable while partial observability still matters.

## Troubleshooting
- NumPy ABI: if you see errors like "module compiled for NumPy 1.x cannot run on NumPy 2.x", reinstall with `pip install 'numpy<2' --force-reinstall`.
- For speed, lower `--episodes`, `--agents`, or `--max_steps` during smoke testing.
