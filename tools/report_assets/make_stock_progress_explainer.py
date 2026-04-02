from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "report_progress_explainer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUCCESS_DIR = ROOT / "outputs" / "stock_stack_runs" / "stock_sampling_compare" / "episode_000"
FAIL_DIR = ROOT / "outputs" / "stock_stack_runs" / "stock_front_touchdown_confirm_probe_j_rearfast" / "episode_000"

LEG_NAMES = ["FL", "FR", "RL", "RR"]
LEG_IDX = {name: i for i, name in enumerate(LEG_NAMES)}


def _load_run(ep_dir: Path) -> dict:
    data = np.load(ep_dir / "run_log.npz", allow_pickle=True)
    summary = json.loads((ep_dir / "summary.json").read_text(encoding="utf-8"))
    out = {k: data[k] for k in data.files}
    out["summary"] = summary
    return out


def _contact_to_band(values: np.ndarray, offset: float, height: float = 0.8) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    return offset + height * vals


def _legend_text(summary: dict) -> str:
    meta = summary.get("meta", {})
    invalid = meta.get("invalid_contact_keys") or ["none"]
    invalid_txt = ", ".join(invalid)
    return (
        f"type={meta.get('controller_type', 'unknown')}\n"
        f"gait={meta.get('gait', 'unknown')}\n"
        f"duration={summary.get('duration_s', 0.0):.3f}s\n"
        f"mean_vx={summary.get('mean_vx', 0.0):.3f}\n"
        f"mean_base_z={summary.get('mean_base_z', 0.0):.3f}\n"
        f"front_actual_swing={summary.get('front_actual_swing_realization_mean', 0.0):.3f}\n"
        f"invalid={invalid_txt}"
    )


def make_figure() -> Path:
    success = _load_run(SUCCESS_DIR)
    failure = _load_run(FAIL_DIR)

    t_ok = np.asarray(success["time"], dtype=float)
    z_ok = np.asarray(success["base_pos"], dtype=float)[:, 2]
    roll_ok = np.asarray(success["base_ori_euler_xyz"], dtype=float)[:, 0]
    pitch_ok = np.asarray(success["base_ori_euler_xyz"], dtype=float)[:, 1]

    t_bad = np.asarray(failure["time"], dtype=float)
    z_bad = np.asarray(failure["base_pos"], dtype=float)[:, 2]
    roll_bad = np.asarray(failure["base_ori_euler_xyz"], dtype=float)[:, 0]
    pitch_bad = np.asarray(failure["base_ori_euler_xyz"], dtype=float)[:, 1]
    planned_bad = np.asarray(failure["planned_contact"], dtype=float)
    current_bad = np.asarray(failure["current_contact"], dtype=float)
    actual_bad = np.asarray(failure["foot_contact"], dtype=float)
    rr_margin = np.asarray(failure["support_margin"], dtype=float)[:, LEG_IDX["RR"]]
    front_confirm = np.asarray(failure["front_touchdown_confirm_active"], dtype=float)
    rr_confirm = np.zeros_like(t_bad)
    if "rear_touchdown_confirm_active" in failure:
        rr_confirm = np.asarray(failure["rear_touchdown_confirm_active"], dtype=float)[:, LEG_IDX["RR"]]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.3], hspace=0.35, wspace=0.22)

    ax_ok_pose = fig.add_subplot(gs[0, 0])
    ax_ok_pitch = fig.add_subplot(gs[1, 0], sharex=ax_ok_pose)
    ax_text = fig.add_subplot(gs[2, 0])
    ax_bad_pose = fig.add_subplot(gs[0, 1])
    ax_bad_pitch = fig.add_subplot(gs[1, 1], sharex=ax_bad_pose)
    ax_bad_contact = fig.add_subplot(gs[2, 1], sharex=ax_bad_pose)

    ax_ok_pose.plot(t_ok, z_ok, label="base z", color="tab:blue")
    ax_ok_pose.axhline(float(success["summary"]["ref_base_height"]), linestyle="--", color="tab:blue", alpha=0.4, label="ref z")
    ax_ok_pose.set_title("Stock baseline (sampling+trot): stable reference run")
    ax_ok_pose.set_ylabel("height [m]")
    ax_ok_pose.grid(alpha=0.3)
    ax_ok_pose.legend(loc="upper right")

    ax_ok_pitch.plot(t_ok, np.rad2deg(roll_ok), label="roll", color="tab:orange")
    ax_ok_pitch.plot(t_ok, np.rad2deg(pitch_ok), label="pitch", color="tab:green")
    ax_ok_pitch.set_ylabel("angle [deg]")
    ax_ok_pitch.set_xlabel("time [s]")
    ax_ok_pitch.grid(alpha=0.3)
    ax_ok_pitch.legend(loc="upper right")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.98,
        "How to read this comparison",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ax_text.text(
        0.0,
        0.84,
        "- Left: stock sampling controller keeps height and small roll/pitch.\n"
        "- Right: current linear_osqp branch achieves real swing, but late RR swing fails.\n"
        "- In the bottom-right panel, RR planned/current/actual contact separate near the end.\n"
        "- RR actual contact does not recover quickly enough before hip/trunk contact.\n"
        "- This is why the issue is now interpreted as a touchdown/recontact realization gap,\n"
        "  not simply an MPC solver failure.",
        fontsize=11,
        va="top",
        linespacing=1.5,
    )
    ax_text.text(
        0.0,
        0.40,
        "Stock baseline summary\n" + _legend_text(success["summary"]),
        fontsize=10,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )
    ax_text.text(
        0.52,
        0.40,
        "Current linear_osqp summary\n" + _legend_text(failure["summary"]),
        fontsize=10,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )

    ax_bad_pose.plot(t_bad, z_bad, label="base z", color="tab:red")
    ax_bad_pose.axhline(float(failure["summary"]["ref_base_height"]), linestyle="--", color="tab:red", alpha=0.4, label="ref z")
    ax_bad_pose.set_title("Current linear_osqp branch: gait-like but late rear collapse")
    ax_bad_pose.set_ylabel("height [m]")
    ax_bad_pose.grid(alpha=0.3)
    ax_bad_pose.legend(loc="upper right")

    ax_bad_pitch.plot(t_bad, np.rad2deg(roll_bad), label="roll", color="tab:orange")
    ax_bad_pitch.plot(t_bad, np.rad2deg(pitch_bad), label="pitch", color="tab:green")
    ax_bad_pitch.set_ylabel("angle [deg]")
    ax_bad_pitch.set_xlabel("time [s]")
    ax_bad_pitch.grid(alpha=0.3)
    ax_bad_pitch.legend(loc="upper left")

    rr = LEG_IDX["RR"]
    ax_bad_contact.plot(t_bad, _contact_to_band(planned_bad[:, rr], 2.2), label="RR planned contact", color="black", linewidth=1.8)
    ax_bad_contact.plot(t_bad, _contact_to_band(current_bad[:, rr], 1.1), label="RR controller contact", color="tab:purple", linewidth=1.8)
    ax_bad_contact.plot(t_bad, _contact_to_band(actual_bad[:, rr], 0.0), label="RR actual contact", color="tab:red", linewidth=1.8)
    ax_bad_contact.plot(t_bad, 3.4 + 25.0 * rr_margin, label="RR support margin x25 + 3.4", color="tab:blue", linewidth=1.2)
    ax_bad_contact.fill_between(t_bad, 4.5, 4.5 + 0.6 * np.clip(front_confirm[:, LEG_IDX["FR"]], 0, 1), color="tab:cyan", alpha=0.35, label="FR confirm active")
    ax_bad_contact.fill_between(t_bad, 5.4, 5.4 + 0.6 * np.clip(rr_confirm, 0, 1), color="tab:green", alpha=0.35, label="RR confirm active")
    ax_bad_contact.set_ylim(-0.2, 6.4)
    ax_bad_contact.set_yticks([0.4, 1.5, 2.6, 3.8, 4.8, 5.7])
    ax_bad_contact.set_yticklabels(
        ["RR actual", "RR current", "RR planned", "margin", "FR confirm", "RR confirm"]
    )
    ax_bad_contact.set_xlabel("time [s]")
    ax_bad_contact.set_title("Late RR swing / touchdown-recontact timeline")
    ax_bad_contact.grid(alpha=0.3)
    ax_bad_contact.legend(loc="upper left", fontsize=8, ncol=2)

    term_time = failure["summary"]["meta"].get("termination_time")
    if term_time is not None:
        for ax in (ax_bad_pose, ax_bad_pitch, ax_bad_contact):
            ax.axvline(float(term_time), color="tab:red", linestyle="--", alpha=0.65)

    out_path = OUT_DIR / "linear_osqp_progress_explainer.png"
    fig.suptitle("Quadruped-PyMPC stock baseline vs current linear_osqp backend", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = make_figure()
    print(out)
