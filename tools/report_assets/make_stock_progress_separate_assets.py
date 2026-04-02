from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "report_progress_explainer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOCK_DIR = ROOT / "outputs" / "stock_stack_runs" / "stock_sampling_compare" / "episode_000"
STOCK_MP4_DIR = ROOT / "outputs" / "stock_stack_runs" / "stock_sampling_compare_mp4" / "episode_000"
LINEAR_DIR = ROOT / "outputs" / "stock_stack_runs" / "stock_front_touchdown_confirm_probe_j_rearfast" / "episode_000"

LEG_NAMES = ["FL", "FR", "RL", "RR"]
LEG_IDX = {name: i for i, name in enumerate(LEG_NAMES)}


def _load_run(ep_dir: Path) -> dict:
    data = np.load(ep_dir / "run_log.npz", allow_pickle=True)
    summary = json.loads((ep_dir / "summary.json").read_text(encoding="utf-8"))
    out = {k: data[k] for k in data.files}
    out["summary"] = summary
    return out


def _legend_lines(summary: dict) -> list[str]:
    meta = summary.get("meta", {})
    invalid = meta.get("invalid_contact_keys") or ["none"]
    return [
        f"type={meta.get('controller_type', 'unknown')}  gait={meta.get('gait', 'unknown')}",
        f"duration={summary.get('duration_s', 0.0):.3f}s  mean_vx={summary.get('mean_vx', 0.0):.3f}",
        f"mean_base_z={summary.get('mean_base_z', 0.0):.3f}  front swing={summary.get('front_actual_swing_realization_mean', 0.0):.3f}",
        f"invalid={', '.join(invalid)}",
    ]


def _contact_to_band(values: np.ndarray, offset: float, height: float = 0.8) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    return offset + height * vals


def make_stock_image() -> Path:
    stock = _load_run(STOCK_DIR)
    t = np.asarray(stock["time"], dtype=float)
    z = np.asarray(stock["base_pos"], dtype=float)[:, 2]
    actual = np.asarray(stock["foot_contact"], dtype=float)
    meta = stock["summary"]["meta"]

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.plot(t, z, color="tab:blue", linewidth=2, label="base z")
    ax1.axhline(float(stock["summary"]["ref_base_height"]), linestyle="--", color="tab:blue", alpha=0.4, label="ref z")
    ax1.set_title(
        f"Stock baseline | controller={meta['controller_type']} | gait={meta['gait']} | "
        f"robot={meta['robot']} | scene={meta['scene']}"
    )
    ax1.set_ylabel("height [m]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    offsets = [0.0, 1.1, 2.2, 3.3]
    for leg_i, leg_name in enumerate(LEG_NAMES):
        ax2.plot(t, _contact_to_band(actual[:, leg_i], offsets[leg_i]), linewidth=1.8, label=f"{leg_name} actual")
    ax2.set_yticks([0.4, 1.5, 2.6, 3.7])
    ax2.set_yticklabels(["FL", "FR", "RL", "RR"])
    ax2.set_xlabel("time [s]")
    ax2.set_title("Actual foot-contact timeline")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", ncol=2, fontsize=9)

    fig.text(
        0.02,
        0.02,
        f"Summary: duration={stock['summary']['duration_s']:.3f}s | mean_vx={stock['summary']['mean_vx']:.3f} | "
        f"front swing={stock['summary']['front_actual_swing_realization_mean']:.3f} | invalid contact=none",
        fontsize=10,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = OUT_DIR / "stock_sampling_baseline_simple.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_linear_image() -> Path:
    linear = _load_run(LINEAR_DIR)
    t = np.asarray(linear["time"], dtype=float)
    z = np.asarray(linear["base_pos"], dtype=float)[:, 2]
    roll = np.asarray(linear["base_ori_euler_xyz"], dtype=float)[:, 0]
    pitch = np.asarray(linear["base_ori_euler_xyz"], dtype=float)[:, 1]
    planned = np.asarray(linear["planned_contact"], dtype=float)
    current = np.asarray(linear["current_contact"], dtype=float)
    actual = np.asarray(linear["foot_contact"], dtype=float)
    meta = linear["summary"]["meta"]

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(t, z, color="tab:red", linewidth=2, label="base z")
    ax1.axhline(float(linear["summary"]["ref_base_height"]), linestyle="--", color="tab:red", alpha=0.4, label="ref z")
    ax1.set_title(
        f"Current branch | controller={meta['controller_type']} | gait={meta['gait']} | "
        f"robot={meta['robot']} | scene={meta['scene']}"
    )
    ax1.set_ylabel("height [m]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(t, np.rad2deg(roll), color="tab:orange", linewidth=1.8, label="roll")
    ax2.plot(t, np.rad2deg(pitch), color="tab:green", linewidth=1.8, label="pitch")
    ax2.set_ylabel("angle [deg]")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    rr = LEG_IDX["RR"]
    ax3.plot(t, _contact_to_band(planned[:, rr], 2.2), color="black", linewidth=1.8, label="RR planned")
    ax3.plot(t, _contact_to_band(current[:, rr], 1.1), color="tab:purple", linewidth=1.8, label="RR current")
    ax3.plot(t, _contact_to_band(actual[:, rr], 0.0), color="tab:red", linewidth=1.8, label="RR actual")
    ax3.set_ylim(-0.2, 3.4)
    ax3.set_yticks([0.4, 1.5, 2.6])
    ax3.set_yticklabels(["RR actual", "RR current", "RR planned"])
    ax3.set_xlabel("time [s]")
    ax3.set_title("Rear-right contact timeline")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper left", ncol=3, fontsize=9)

    term_time = linear["summary"]["meta"].get("termination_time")
    if term_time is not None:
        for ax in (ax1, ax2, ax3):
            ax.axvline(float(term_time), color="tab:red", linestyle="--", alpha=0.65)

    fig.text(
        0.02,
        0.02,
        f"Summary: duration={linear['summary']['duration_s']:.3f}s | mean_vx={linear['summary']['mean_vx']:.3f} | "
        f"front swing={linear['summary']['front_actual_swing_realization_mean']:.3f} | "
        f"termination=rear-right hip contact",
        fontsize=10,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = OUT_DIR / "linear_osqp_rear_recontact_simple.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _annotate(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 96), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0.0, out)
    cv2.putText(out, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    y = 52
    for line in lines[:2]:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        y += 20
    return out


def _write_gif_from_video(video_path: Path, title: str, lines: list[str], out_path: Path, max_frames: int = 80, frame_step: int = 3, target_width: int = 560) -> Path:
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
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
    summary = json.loads((STOCK_MP4_DIR / "summary.json").read_text(encoding="utf-8"))
    lines = [
        f"controller=sampling | gait=trot | scene=flat",
        f"duration={summary['duration_s']:.3f}s | stable stock baseline",
    ]
    return _write_gif_from_video(
        STOCK_MP4_DIR / "mujoco_scene.mp4",
        "Stock baseline",
        lines,
        OUT_DIR / "stock_sampling_baseline_simple.gif",
    )


def make_linear_gif() -> Path:
    summary = json.loads((LINEAR_DIR / "summary.json").read_text(encoding="utf-8"))
    lines = [
        f"controller=linear_osqp | gait=crawl | scene=flat",
        f"duration={summary['duration_s']:.3f}s | rear-right hip contact",
    ]
    return _write_gif_from_video(
        LINEAR_DIR / "mujoco_scene.mp4",
        "Current linear_osqp branch",
        lines,
        OUT_DIR / "linear_osqp_rear_recontact_simple.gif",
    )


def main() -> None:
    outputs = [
        make_stock_image(),
        make_linear_image(),
        make_stock_gif(),
        make_linear_gif(),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
