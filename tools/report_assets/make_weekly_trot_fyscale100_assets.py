from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
WEEKLY_DIR = ROOT / "outputs" / "report_progress_explainer" / "weekly_progress_20260410"
CLIPS_DIR = WEEKLY_DIR / "clips"

RUNS = {
    "turn_old": ROOT / "outputs" / "curated_runs" / "current" / "trot_turn_10s_g025_pitchoff003" / "episode_000" / "summary.json",
    "turn_new": WEEKLY_DIR / "custom_runs" / "trot_turn_10s_fyscale100" / "episode_000" / "summary.json",
    "turn_stock": ROOT / "outputs" / "curated_runs" / "stock_baselines" / "stock_trot_turn_10s_weeklyref" / "episode_000" / "summary.json",
    "disturb_old": ROOT / "outputs" / "curated_runs" / "current" / "trot_disturb_4s_g025_pitchoff003" / "episode_000" / "summary.json",
    "disturb_new": WEEKLY_DIR / "custom_runs" / "trot_disturb_4s_fyscale100" / "episode_000" / "summary.json",
    "disturb_stock": ROOT / "outputs" / "curated_runs" / "stock_baselines" / "stock_sampling_trot_disturb_4s_x48_recheck" / "episode_000" / "summary.json",
    "straight_new": WEEKLY_DIR / "custom_runs" / "trot_straight_4s_fyscale100" / "episode_000" / "summary.json",
    "straight_stock": ROOT / "outputs" / "curated_runs" / "stock_baselines" / "stock_sampling_trot_4s_s012_isolated_recheck" / "episode_000" / "summary.json",
    "crawl_old": ROOT / "outputs" / "curated_runs" / "current" / "crawl_current_default_20s" / "episode_000" / "summary.json",
    "crawl_new": ROOT / "outputs" / "archive" / "raw_runs" / "20260410_fyscale100_recheck" / "crawl_20s" / "episode_000" / "summary.json",
    "crawl_force_base": ROOT / "outputs" / "archive" / "raw_runs" / "20260410_crawl_force_relax_recheck" / "baseline_fy015_grf035" / "episode_000" / "summary.json",
    "crawl_force_fy100_grf035": ROOT / "outputs" / "archive" / "raw_runs" / "20260410_crawl_force_relax_recheck" / "fy100_grf035" / "episode_000" / "summary.json",
    "crawl_force_fy015_grf100": ROOT / "outputs" / "archive" / "raw_runs" / "20260410_crawl_force_relax_recheck" / "fy015_grf100" / "episode_000" / "summary.json",
    "crawl_force_fy100_grf100": ROOT / "outputs" / "archive" / "raw_runs" / "20260410_crawl_force_relax_recheck" / "fy100_grf100" / "episode_000" / "summary.json",
}

VIDEOS = {
    "straight": WEEKLY_DIR / "custom_runs" / "trot_straight_4s_fyscale100" / "episode_000" / "mujoco_scene.mp4",
    "turn": WEEKLY_DIR / "custom_runs" / "trot_turn_10s_fyscale100" / "episode_000" / "mujoco_scene.mp4",
    "disturb": WEEKLY_DIR / "custom_runs" / "trot_disturb_4s_fyscale100" / "episode_000" / "mujoco_scene.mp4",
}

GIFS = {
    "straight": CLIPS_DIR / "trot_straight_custom_fyscale100.gif",
    "turn": CLIPS_DIR / "trot_turn_custom_fyscale100.gif",
    "disturb": CLIPS_DIR / "trot_disturb_custom_fyscale100.gif",
}

GRAPH_PATH = WEEKLY_DIR / "weekly_trot_fyscale100_recheck.png"
CRAWL_GRAPH_PATH = WEEKLY_DIR / "weekly_crawl_force_relax_recheck.png"

COLORS = {
    "old": "#7F7F7F",
    "new": "#E45756",
    "stock": "#4C78A8",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _put_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def make_graph() -> Path:
    data = {name: _load(path) for name, path in RUNS.items()}
    rows = [
        ("Turn 10 s\nmean |roll|", data["turn_old"]["mean_abs_roll"], data["turn_new"]["mean_abs_roll"], data["turn_stock"]["mean_abs_roll"]),
        ("Turn 10 s\nmean |pitch|", data["turn_old"]["mean_abs_pitch"], data["turn_new"]["mean_abs_pitch"], data["turn_stock"]["mean_abs_pitch"]),
        ("Disturb 4 s\nmean |roll|", data["disturb_old"]["mean_abs_roll"], data["disturb_new"]["mean_abs_roll"], data["disturb_stock"]["mean_abs_roll"]),
        ("Disturb 4 s\nmean |pitch|", data["disturb_old"]["mean_abs_pitch"], data["disturb_new"]["mean_abs_pitch"], data["disturb_stock"]["mean_abs_pitch"]),
    ]
    labels = [row[0] for row in rows]
    old_vals = [row[1] for row in rows]
    new_vals = [row[2] for row in rows]
    stock_vals = [row[3] for row in rows]

    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    x = np.arange(len(labels))
    width = 0.24
    b1 = ax.bar(x - width, old_vals, width, label="previous current", color=COLORS["old"])
    b2 = ax.bar(x, new_vals, width, label="fy_scale=1.0 recheck", color=COLORS["new"])
    b3 = ax.bar(x + width, stock_vals, width, label="stock reference", color=COLORS["stock"])
    _put_labels(ax, b1)
    _put_labels(ax, b2)
    _put_labels(ax, b3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("metric value")
    ax.set_title("Additional recheck: fy_scale=1.0 in generic trot", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    note = (
        f"Straight 4 s sanity: custom |roll|={data['straight_new']['mean_abs_roll']:.3f}, "
        f"|pitch|={data['straight_new']['mean_abs_pitch']:.3f}; "
        f"stock |roll|={data['straight_stock']['mean_abs_roll']:.3f}, "
        f"|pitch|={data['straight_stock']['mean_abs_pitch']:.3f}. "
        f"Crawl unchanged: {data['crawl_old']['duration_s']:.3f}s -> {data['crawl_new']['duration_s']:.3f}s."
    )
    fig.text(0.02, 0.02, note, fontsize=9, family="monospace")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(GRAPH_PATH, dpi=220)
    plt.close(fig)
    return GRAPH_PATH


def make_crawl_force_relax_graph() -> Path:
    data = {name: _load(path) for name, path in RUNS.items()}
    labels = [
        "baseline\nfy=0.15\ngrf=0.35",
        "fy=1.0\ngrf=0.35",
        "fy=0.15\ngrf=1.0",
        "fy=1.0\ngrf=1.0",
    ]
    duration_vals = [
        data["crawl_force_base"]["duration_s"],
        data["crawl_force_fy100_grf035"]["duration_s"],
        data["crawl_force_fy015_grf100"]["duration_s"],
        data["crawl_force_fy100_grf100"]["duration_s"],
    ]
    roll_vals = [
        data["crawl_force_base"]["mean_abs_roll"],
        data["crawl_force_fy100_grf035"]["mean_abs_roll"],
        data["crawl_force_fy015_grf100"]["mean_abs_roll"],
        data["crawl_force_fy100_grf100"]["mean_abs_roll"],
    ]
    pitch_vals = [
        data["crawl_force_base"]["mean_abs_pitch"],
        data["crawl_force_fy100_grf035"]["mean_abs_pitch"],
        data["crawl_force_fy015_grf100"]["mean_abs_pitch"],
        data["crawl_force_fy100_grf100"]["mean_abs_pitch"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518"]

    for ax, values, title in [
        (axes[0], duration_vals, "duration (s)"),
        (axes[1], roll_vals, "mean |roll|"),
        (axes[2], pitch_vals, "mean |pitch|"),
    ]:
        bars = ax.bar(x, values, color=colors)
        _put_labels(ax, bars)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Crawl force-relax recheck: loosening fy/grf does not fix late seam", fontsize=14, fontweight="bold")
    note = (
        f"Current crawl default matches baseline at {data['crawl_old']['duration_s']:.3f}s. "
        f"Relaxed-force variants regress to {duration_vals[1]:.3f}s, {duration_vals[2]:.3f}s, and {duration_vals[3]:.3f}s."
    )
    fig.text(0.02, 0.02, note, fontsize=9, family="monospace")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(CRAWL_GRAPH_PATH, dpi=220)
    plt.close(fig)
    return CRAWL_GRAPH_PATH


def _video_to_gif(video_path: Path, out_path: Path, title: str, max_frames: int = 84, frame_step: int = 3, target_width: int = 560) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    frames: list[Image.Image] = []
    index = 0
    used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            scale = target_width / float(image.width)
            image = image.resize((target_width, int(image.height * scale)), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (image.width, image.height + 32), "white")
            canvas.paste(image, (0, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), title, fill="black")
            frames.append(canvas)
            used += 1
            if used >= max_frames:
                break
        index += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames extracted from {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        optimize=False,
    )
    return out_path


def make_gifs() -> list[Path]:
    return [
        _video_to_gif(VIDEOS["straight"], GIFS["straight"], "Custom trot straight 4 s (generic, fy_scale=1.0)"),
        _video_to_gif(VIDEOS["turn"], GIFS["turn"], "Custom trot turn 10 s (generic, fy_scale=1.0)"),
        _video_to_gif(VIDEOS["disturb"], GIFS["disturb"], "Custom trot disturb 4 s (generic, fy_scale=1.0)"),
    ]


def main() -> None:
    make_graph()
    make_crawl_force_relax_graph()
    make_gifs()
    print(GRAPH_PATH)
    print(CRAWL_GRAPH_PATH)
    for path in GIFS.values():
        print(path)


if __name__ == "__main__":
    main()
