from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "report_progress_explainer" / "crawl_same_scenario"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOCK_DIR = ROOT / "outputs" / "same_scenario_compare_mp4" / "sampling_crawl" / "episode_000"
LINEAR_DIR = ROOT / "outputs" / "same_scenario_compare_mp4" / "linear_crawl" / "episode_000"

LEG_NAMES = ["FL", "FR", "RL", "RR"]


def _load_episode(ep_dir: Path) -> dict:
    data = np.load(ep_dir / "run_log.npz", allow_pickle=True)
    summary = json.loads((ep_dir / "summary.json").read_text(encoding="utf-8"))
    out = {key: data[key] for key in data.files}
    out["summary"] = summary
    return out


def _contact_to_band(values: np.ndarray, offset: float, height: float = 0.8) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    return offset + height * vals


def _invalid_contact_text(summary: dict) -> str:
    invalid = summary.get("meta", {}).get("invalid_contact_keys", [])
    if not invalid:
        return "none"
    return invalid[0].replace("world:0_", "").replace(":", " ")


def make_compare_table() -> Path:
    stock = json.loads((STOCK_DIR / "summary.json").read_text(encoding="utf-8"))
    linear = json.loads((LINEAR_DIR / "summary.json").read_text(encoding="utf-8"))

    columns = [
        "controller",
        "duration [s]",
        "terminated",
        "mean base z [m]",
        "mean |roll| [rad]",
        "front swing",
        "termination",
    ]
    rows = [
        [
            "sampling",
            f"{stock['duration_s']:.3f}",
            str(stock["terminated_any"]),
            f"{stock['mean_base_z']:.3f}",
            f"{stock['mean_abs_roll']:.3f}",
            f"{stock['front_actual_swing_realization_mean']:.3f}",
            _invalid_contact_text(stock),
        ],
        [
            "linear_osqp",
            f"{linear['duration_s']:.3f}",
            str(linear["terminated_any"]),
            f"{linear['mean_base_z']:.3f}",
            f"{linear['mean_abs_roll']:.3f}",
            f"{linear['front_actual_swing_realization_mean']:.3f}",
            _invalid_contact_text(linear),
        ],
    ]

    fig, ax = plt.subplots(figsize=(11.8, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
        elif row == 1:
            cell.set_facecolor("#f4f8ff")
        elif row == 2:
            cell.set_facecolor("#fff5ef")

    fig.suptitle(
        "Same-scenario crawl comparison\n"
        "robot=aliengo | scene=flat | gait=crawl | command=vx=0.12, yaw=0.0",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out_path = OUT_DIR / "crawl_same_scenario_compare_table.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_stock_image() -> Path:
    stock = _load_episode(STOCK_DIR)
    t = np.asarray(stock["time"], dtype=float)
    z = np.asarray(stock["base_pos"], dtype=float)[:, 2]
    roll = np.asarray(stock["base_ori_euler_xyz"], dtype=float)[:, 0]
    pitch = np.asarray(stock["base_ori_euler_xyz"], dtype=float)[:, 1]
    actual = np.asarray(stock["foot_contact"], dtype=float)

    fig = plt.figure(figsize=(12.2, 8.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(t, z, color="tab:blue", linewidth=2.0, label="base z")
    ax1.axhline(float(stock["summary"]["ref_base_height"]), color="tab:blue", linestyle="--", alpha=0.45, label="ref z")
    ax1.set_title("Stock sampling-based MPC | aliengo | flat | crawl | vx=0.12 | yaw=0.0")
    ax1.set_ylabel("height [m]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    ax2.plot(t, np.rad2deg(roll), color="tab:orange", linewidth=1.8, label="roll")
    ax2.plot(t, np.rad2deg(pitch), color="tab:green", linewidth=1.8, label="pitch")
    ax2.set_ylabel("angle [deg]")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    offsets = [0.0, 1.1, 2.2, 3.3]
    for idx, leg_name in enumerate(LEG_NAMES):
        ax3.plot(t, _contact_to_band(actual[:, idx], offsets[idx]), linewidth=1.8, label=f"{leg_name} actual")
    ax3.set_yticks([0.4, 1.5, 2.6, 3.7])
    ax3.set_yticklabels(LEG_NAMES)
    ax3.set_xlabel("time [s]")
    ax3.set_title("Actual foot-contact timeline")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=9)

    fig.text(
        0.02,
        0.02,
        f"duration={stock['summary']['duration_s']:.3f}s | mean_z={stock['summary']['mean_base_z']:.3f} | "
        f"mean|roll|={stock['summary']['mean_abs_roll']:.3f} | front swing={stock['summary']['front_actual_swing_realization_mean']:.3f} | "
        f"termination={_invalid_contact_text(stock['summary'])}",
        fontsize=10,
        family="monospace",
    )
    fig.subplots_adjust(right=0.79, bottom=0.08, top=0.94)

    out_path = OUT_DIR / "stock_sampling_crawl_explainer.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_linear_image() -> Path:
    linear = _load_episode(LINEAR_DIR)
    t = np.asarray(linear["time"], dtype=float)
    z = np.asarray(linear["base_pos"], dtype=float)[:, 2]
    roll = np.asarray(linear["base_ori_euler_xyz"], dtype=float)[:, 0]
    pitch = np.asarray(linear["base_ori_euler_xyz"], dtype=float)[:, 1]
    actual = np.asarray(linear["foot_contact"], dtype=float)

    fig = plt.figure(figsize=(12.2, 8.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(t, z, color="tab:red", linewidth=2.0, label="base z")
    ax1.axhline(float(linear["summary"]["ref_base_height"]), color="tab:red", linestyle="--", alpha=0.45, label="ref z")
    ax1.set_title("Custom linear_osqp MPC | aliengo | flat | crawl | vx=0.12 | yaw=0.0")
    ax1.set_ylabel("height [m]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    ax2.plot(t, np.rad2deg(roll), color="tab:orange", linewidth=1.8, label="roll")
    ax2.plot(t, np.rad2deg(pitch), color="tab:green", linewidth=1.8, label="pitch")
    ax2.set_ylabel("angle [deg]")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    offsets = [0.0, 1.1, 2.2, 3.3]
    for idx, leg_name in enumerate(LEG_NAMES):
        ax3.plot(t, _contact_to_band(actual[:, idx], offsets[idx]), linewidth=1.8, label=f"{leg_name} actual")
    ax3.set_yticks([0.4, 1.5, 2.6, 3.7])
    ax3.set_yticklabels(LEG_NAMES)
    ax3.set_xlabel("time [s]")
    ax3.set_title("Actual foot-contact timeline")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=9)

    term_time = linear["summary"]["meta"].get("termination_time")
    if term_time is not None:
        for ax in (ax1, ax2, ax3):
            ax.axvline(float(term_time), color="tab:red", linestyle="--", alpha=0.65)

    fig.text(
        0.02,
        0.02,
        f"duration={linear['summary']['duration_s']:.3f}s | mean_z={linear['summary']['mean_base_z']:.3f} | "
        f"mean|roll|={linear['summary']['mean_abs_roll']:.3f} | front swing={linear['summary']['front_actual_swing_realization_mean']:.3f} | "
        f"termination={_invalid_contact_text(linear['summary'])}",
        fontsize=10,
        family="monospace",
    )
    fig.subplots_adjust(right=0.79, bottom=0.08, top=0.94)

    out_path = OUT_DIR / "linear_osqp_crawl_explainer.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _annotate(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 100), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0.0, out)
    cv2.putText(out, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    y = 54
    for line in lines[:2]:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        y += 22
    return out


def _write_gif(video_path: Path, title: str, lines: list[str], out_path: Path, max_frames: int = 80, frame_step: int = 3, target_width: int = 560) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")

    frames: list[Image.Image] = []
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            h, w = frame.shape[:2]
            target_h = int(round(h * (target_width / float(w))))
            frame = cv2.resize(frame, (target_width, target_h), interpolation=cv2.INTER_AREA)
            frame = _annotate(frame, title, lines)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            kept += 1
            if kept >= max_frames:
                break
        idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=False,
    )
    return out_path


def make_stock_gif() -> Path:
    summary = json.loads((STOCK_DIR / "summary.json").read_text(encoding="utf-8"))
    return _write_gif(
        STOCK_DIR / "mujoco_scene.mp4",
        "Stock sampling MPC | crawl",
        [
            "aliengo | flat | vx=0.12 | yaw=0.0",
            f"duration={summary['duration_s']:.3f}s | {_invalid_contact_text(summary)}",
        ],
        OUT_DIR / "stock_sampling_crawl.gif",
    )


def make_linear_gif() -> Path:
    summary = json.loads((LINEAR_DIR / "summary.json").read_text(encoding="utf-8"))
    return _write_gif(
        LINEAR_DIR / "mujoco_scene.mp4",
        "Custom linear_osqp | crawl",
        [
            "aliengo | flat | vx=0.12 | yaw=0.0",
            f"duration={summary['duration_s']:.3f}s | {_invalid_contact_text(summary)}",
        ],
        OUT_DIR / "linear_osqp_crawl.gif",
    )


def main() -> None:
    outputs = [
        make_compare_table(),
        make_stock_image(),
        make_linear_image(),
        make_stock_gif(),
        make_linear_gif(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
