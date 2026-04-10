from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "outputs" / "archive" / "raw_runs" / "20260410_trot_stock_vs_custom_same_scenarios"
OUT_DIR = ROOT / "outputs" / "report_progress_explainer" / "trot_stock_vs_custom_same_scenarios_20260410"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOCK_COLOR = "#2C7FB8"
CUSTOM_COLOR = "#D7301F"


def _load_episode(name: str) -> tuple[dict, dict]:
    ep_dir = RUN_ROOT / name / "episode_000"
    data = np.load(ep_dir / "run_log.npz", allow_pickle=True)
    summary = json.loads((ep_dir / "summary.json").read_text(encoding="utf-8"))
    return {key: data[key] for key in data.files}, summary


def build() -> Path:
    stock_straight, stock_straight_s = _load_episode("stock_straight_20s")
    custom_straight, custom_straight_s = _load_episode("custom_straight_20s")
    stock_turn, stock_turn_s = _load_episode("stock_turn_10s")
    custom_turn, custom_turn_s = _load_episode("custom_turn_10s")
    stock_disturb, stock_disturb_s = _load_episode("stock_disturb_4s")
    custom_disturb, custom_disturb_s = _load_episode("custom_disturb_4s")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    # Straight: vx tracking
    ax = axes[0]
    ts = np.asarray(stock_straight["time"], dtype=float)
    tc = np.asarray(custom_straight["time"], dtype=float)
    vx_s = np.asarray(stock_straight["base_lin_vel"], dtype=float)[:, 0]
    vx_c = np.asarray(custom_straight["base_lin_vel"], dtype=float)[:, 0]
    ax.plot(ts, vx_s, color=STOCK_COLOR, linewidth=2.0, label="stock")
    ax.plot(tc, vx_c, color=CUSTOM_COLOR, linewidth=2.0, label="custom")
    ax.axhline(0.12, color="0.4", linestyle="--", linewidth=1.2, label="cmd vx")
    ax.set_title("Straight 20 s\nForward velocity")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("vx [m/s]")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=9, loc="lower right")
    ax.text(
        0.02,
        0.96,
        f"mean vx: stock {stock_straight_s['mean_vx']:.3f} | custom {custom_straight_s['mean_vx']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )

    # Turn: XY path
    ax = axes[1]
    pos_s = np.asarray(stock_turn["base_pos"], dtype=float)
    pos_c = np.asarray(custom_turn["base_pos"], dtype=float)
    ax.plot(pos_s[:, 0], pos_s[:, 1], color=STOCK_COLOR, linewidth=2.0, label="stock")
    ax.plot(pos_c[:, 0], pos_c[:, 1], color=CUSTOM_COLOR, linewidth=2.0, label="custom")
    ax.scatter(pos_s[0, 0], pos_s[0, 1], color=STOCK_COLOR, s=28)
    ax.scatter(pos_c[0, 0], pos_c[0, 1], color=CUSTOM_COLOR, s=28)
    ax.set_title("Turn 10 s\nXY path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=9, loc="best")
    ax.text(
        0.02,
        0.96,
        f"mean|roll|: stock {stock_turn_s['mean_abs_roll']:.3f} | custom {custom_turn_s['mean_abs_roll']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )

    # Disturbance: XY path + pulse markers in title note
    ax = axes[2]
    pos_s = np.asarray(stock_disturb["base_pos"], dtype=float)
    pos_c = np.asarray(custom_disturb["base_pos"], dtype=float)
    ax.plot(pos_s[:, 0], pos_s[:, 1], color=STOCK_COLOR, linewidth=2.0, label="stock")
    ax.plot(pos_c[:, 0], pos_c[:, 1], color=CUSTOM_COLOR, linewidth=2.0, label="custom")
    ax.scatter(pos_s[0, 0], pos_s[0, 1], color=STOCK_COLOR, s=28)
    ax.scatter(pos_c[0, 0], pos_c[0, 1], color=CUSTOM_COLOR, s=28)
    ax.set_title("Disturbance 4 s\nXY path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=9, loc="best")
    ax.text(
        0.02,
        0.96,
        f"mean|pitch|: stock {stock_disturb_s['mean_abs_pitch']:.3f} | custom {custom_disturb_s['mean_abs_pitch']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )

    fig.suptitle(
        "Trot trajectory/tracking overview | matched local stock vs custom benchmarks",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.02,
        0.02,
        "Straight uses vx tracking because XY path is nearly degenerate; turn/disturbance use XY path for motion comparison.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))

    out_path = OUT_DIR / "trot_tracking_overview.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(build())
