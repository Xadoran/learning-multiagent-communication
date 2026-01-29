from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _utc_timestamp_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def create_run_dir(
    env: str,
    model: str,
    run_id: str,
    runs_root: str | Path = "runs",
) -> Path:
    """
    Create a run directory:
      runs/{env}/{model}/{run_id}/
    """
    run_dir = Path(runs_root) / str(env) / str(model) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def default_run_id(
    *,
    seed: int,
    K: int | None = None,
    vision: float | None = None,
    vision_range: int | None = None,
) -> str:
    """
    Human-readable, sortable run id.
    """
    parts: list[str] = [f"{_utc_timestamp_compact()}", f"seed{seed}"]
    if K is not None:
        parts.append(f"K{int(K)}")
    if vision is not None:
        parts.append(f"vision{vision:g}")
    if vision_range is not None:
        parts.append(f"vr{int(vision_range)}")
    return "_".join(parts)


def save_run_config(run_dir: str | Path, config: dict[str, Any]) -> None:
    run_dir_p = Path(run_dir)
    run_dir_p.mkdir(parents=True, exist_ok=True)
    path = run_dir_p / "run_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


@dataclass
class RunLogger:
    run_dir: Path
    extra_columns: list[str]

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"

        self.columns = [
            "episode_idx",
            "episode_return",
            "success",
            "episode_len",
            *self.extra_columns,
        ]

        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()

    def append_episode(
        self,
        *,
        episode_idx: int,
        episode_return: float,
        success: int,
        episode_len: int,
        extras: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "episode_idx": int(episode_idx),
            "episode_return": float(episode_return),
            "success": int(success),
            "episode_len": int(episode_len),
        }
        extras = extras or {}
        for k in self.extra_columns:
            v = extras.get(k, "")
            # Keep CSV friendly (avoid dicts/lists); JSON-encode non-scalars.
            if isinstance(v, (dict, list, tuple)):
                row[k] = json.dumps(v)
            else:
                row[k] = v

        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)


def extra_columns_for_env(env: str) -> list[str]:
    """
    Optional, env-specific diagnostics (best-effort).
    """
    if env == "nav":
        return ["mean_dist", "collisions"]
    if env == "traffic":
        return ["collisions_t", "failure", "cars_present"]
    if env == "lever":
        return ["distinct"]
    if env == "combat":
        return ["outcome", "model_alive", "bot_alive", "enemy_total_hp"]
    return []

