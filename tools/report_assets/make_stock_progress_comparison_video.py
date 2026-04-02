from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "report_progress_explainer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_DIR = ROOT / "outputs" / "stock_stack_runs" / "stock_sampling_compare_mp4" / "episode_000"
RIGHT_DIR = ROOT / "outputs" / "stock_stack_runs" / "stock_front_touchdown_confirm_probe_j_rearfast" / "episode_000"


def _read_summary(ep_dir: Path) -> dict:
    return json.loads((ep_dir / "summary.json").read_text(encoding="utf-8"))


def _resize_keep_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    target_w = int(round(w * (target_h / float(h))))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _annotate(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 120), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0.0, out)
    cv2.putText(out, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    y = 62
    for line in lines:
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        y += 22
    return out


def _frame_count(cap: cv2.VideoCapture) -> int:
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return max(count, 0)


def make_video() -> Path:
    left_video = LEFT_DIR / "mujoco_scene.mp4"
    right_video = RIGHT_DIR / "mujoco_scene.mp4"
    left_summary = _read_summary(LEFT_DIR)
    right_summary = _read_summary(RIGHT_DIR)

    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError("Unable to open one of the input videos.")

    fps = cap_l.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    total_l = _frame_count(cap_l)
    total_r = _frame_count(cap_r)
    total = max(total_l, total_r)

    ok_l, frame_l = cap_l.read()
    ok_r, frame_r = cap_r.read()
    if not ok_l or not ok_r:
        raise RuntimeError("Unable to read the first frame from one of the videos.")

    target_h = min(frame_l.shape[0], frame_r.shape[0])
    frame_l = _resize_keep_height(frame_l, target_h)
    frame_r = _resize_keep_height(frame_r, target_h)
    gap = 20
    header_h = 70
    canvas_h = target_h + header_h
    canvas_w = frame_l.shape[1] + gap + frame_r.shape[1]

    out_path = OUT_DIR / "stock_vs_linear_osqp_comparison.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for {out_path}")

    lines_l = [
        f"type={left_summary['meta']['controller_type']}  gait={left_summary['meta']['gait']}",
        f"duration={left_summary['duration_s']:.3f}s  mean_vx={left_summary['mean_vx']:.3f}",
        f"front swing={left_summary['front_actual_swing_realization_mean']:.3f}  terminated={left_summary['terminated_any']}",
    ]
    invalid = ", ".join(right_summary["meta"].get("invalid_contact_keys", [])) or "none"
    lines_r = [
        f"type={right_summary['meta']['controller_type']}  gait={right_summary['meta']['gait']}",
        f"duration={right_summary['duration_s']:.3f}s  mean_vx={right_summary['mean_vx']:.3f}",
        f"front swing={right_summary['front_actual_swing_realization_mean']:.3f}  fail={invalid}",
    ]

    last_l = frame_l.copy()
    last_r = frame_r.copy()
    idx_l = 1
    idx_r = 1

    for _ in range(total):
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        left_anno = _annotate(last_l, "Stock baseline: sampling + trot", lines_l)
        right_anno = _annotate(last_r, "Current backend: linear_osqp + crawl", lines_r)

        canvas[header_h:, : left_anno.shape[1]] = left_anno
        canvas[header_h:, left_anno.shape[1] + gap : left_anno.shape[1] + gap + right_anno.shape[1]] = right_anno
        cv2.putText(
            canvas,
            "Comparison target: stock stack is stable, current linear_osqp still collapses in late touchdown/recontact.",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)

        if idx_l < total_l:
            ok_l, next_l = cap_l.read()
            if ok_l:
                last_l = _resize_keep_height(next_l, target_h)
            idx_l += 1
        if idx_r < total_r:
            ok_r, next_r = cap_r.read()
            if ok_r:
                last_r = _resize_keep_height(next_r, target_h)
            idx_r += 1

    writer.release()
    cap_l.release()
    cap_r.release()
    return out_path


if __name__ == "__main__":
    out = make_video()
    print(out)
