from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    from simulation.autotune_linear_osqp import BASE_PARAMS, _candidate_score, _run_candidate, _slugify_params, _write_json
else:
    from .autotune_linear_osqp import BASE_PARAMS, _candidate_score, _run_candidate, _slugify_params, _write_json


PHASE_MINUTES = {
    "phase1": 20.0,
    "phase2": 40.0,
    "stage_a": 80.0,
    "stage_b": 80.0,
    "stage_c": 80.0,
    "phase4": 50.0,
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(base)
    if override:
        out.update(override)
    return out


def _params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _load_autotune_seed(repo_root: Path) -> dict[str, Any]:
    status_path = repo_root / "outputs" / "autotune_session" / "status.json"
    params = dict(BASE_PARAMS)
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            best_params = payload.get("best_params")
            if isinstance(best_params, dict):
                params.update(best_params)
        except Exception:
            pass
    params.setdefault("speed", 0.12)
    params.setdefault("lateral_speed", 0.0)
    params.setdefault("yaw_rate", 0.0)
    params.setdefault("seed", 0)
    return params


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _height_ratio(summary: dict[str, Any]) -> float:
    ref_height = float(summary.get("ref_base_height") or 0.0)
    mean_base_z = float(summary.get("mean_base_z") or 0.0)
    if ref_height <= 1e-9:
        return 0.0
    return mean_base_z / ref_height


def _promotion_ready_phase2(summary: dict[str, Any]) -> bool:
    return bool(
        float(summary.get("duration_s") or 0.0) >= 0.9
        and _height_ratio(summary) >= 0.75
        and float(summary.get("mean_abs_pitch") or np.inf) <= 0.15
        and abs(float(summary.get("mean_vy") or 0.0)) <= 0.03
        and float(summary.get("steps_any_current_swing") or 0.0) >= 100.0
    )


def _promotion_ready_stage_a(summary: dict[str, Any]) -> bool:
    return _promotion_ready_phase2(summary)


def _promotion_ready_stage_b(summary: dict[str, Any]) -> bool:
    return bool(
        _promotion_ready_phase2(summary)
        and float(summary.get("duration_s") or 0.0) >= 1.5
        and float(summary.get("legs_with_current_swing") or 0.0) >= 2.0
    )


def _promotion_ready_stage_c(summary: dict[str, Any]) -> bool:
    return bool(
        _promotion_ready_stage_b(summary)
        and float(summary.get("duration_s") or 0.0) >= 3.0
    )


def _overnight_success(summary: dict[str, Any]) -> bool:
    quality = summary.get("quality_gate", {})
    ref_height = float(summary.get("ref_base_height") or 0.0)
    mean_base_z = float(summary.get("mean_base_z") or 0.0)
    height_ratio = (mean_base_z / ref_height) if ref_height > 1e-9 else 0.0
    return bool(
        quality.get("passes", False)
        and float(summary.get("legs_with_current_swing") or 0.0) >= 4.0
        and abs(float(summary.get("mean_vy") or 0.0)) <= 0.03
        and float(summary.get("mean_vx") or 0.0) >= 0.02
        and height_ratio >= 0.75
    )


def _phase2_candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    combos = list(product([2.0, 4.0], [6.0, 8.0], [0.0, 0.1]))
    for idx, (shift_x, shift_y, margin_boost) in enumerate(combos):
        candidates.append(
            _deep_merge(
                base,
                {
                    "support_floor_ratio": 0.22 if idx % 2 == 0 else 0.24,
                    "grf_max_scale": 0.46 if idx < 4 else 0.50,
                    "startup_full_stance_steps": 15 if idx % 2 == 0 else 20,
                    "preswing_shift_x_gain": shift_x,
                    "preswing_shift_y_gain": shift_y,
                    "support_margin_vertical_boost": margin_boost,
                },
            )
        )
    return candidates


def _nearest_choices(value: float, choices: list[float], width: int = 3) -> list[float]:
    ordered = sorted(choices, key=lambda item: (abs(float(item) - float(value)), float(item)))
    picked = ordered[: max(1, width)]
    picked.sort()
    return picked


def _mutate_from_space(
    rng: random.Random,
    base: dict[str, Any],
    search_space: dict[str, list[Any]],
    max_changes: int,
    local: bool,
) -> dict[str, Any]:
    candidate = dict(base)
    keys = list(search_space.keys())
    n_changes = rng.randint(1, max(1, min(max_changes, len(keys))))
    for key in rng.sample(keys, n_changes):
        options = list(search_space[key])
        if local and isinstance(base.get(key), (int, float)):
            options = _nearest_choices(float(base[key]), [float(x) for x in options], width=min(3, len(options)))
        candidate[key] = rng.choice(options)
    return candidate


def _top_unique_param_sets(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    ranked = sorted(records, key=lambda item: (float(item["score"]), float(item["duration_s"])), reverse=True)
    selected: list[dict[str, Any]] = []
    for record in ranked:
        key = json.dumps(record["params"], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


class OvernightRunner:
    def __init__(self, repo_root: Path, base_dir: Path, timeout_s: int, seed: int) -> None:
        self.repo_root = repo_root
        self.base_dir = base_dir
        self.timeout_s = timeout_s
        self.rng = random.Random(seed)
        self.history_path = base_dir / "history.jsonl"
        self.status_path = base_dir / "status.json"
        self.scratch_dir = base_dir / "scratch"
        self.promoted_dir = base_dir / "promoted_runs"
        self.best_dir = base_dir / "best_run"
        self.milestones_dir = base_dir / "milestones"
        self.finalists_dir = base_dir / "finalists"
        self.snapshots_dir = base_dir / "snapshots"
        self.run_index = 0
        self.completed_runs = 0
        self.best_record: dict[str, Any] | None = None
        self.search_anchor: dict[str, Any] | None = None
        self.promoted_records: list[dict[str, Any]] = []
        self.media_best_duration = 0.0

        for path in (self.base_dir, self.scratch_dir, self.promoted_dir, self.milestones_dir, self.finalists_dir, self.snapshots_dir):
            path.mkdir(parents=True, exist_ok=True)

    def snapshot_baselines(self) -> None:
        _copy_tree_if_exists(
            self.repo_root / "outputs" / "autotune_session" / "best_run",
            self.snapshots_dir / "autotune_best",
        )
        _copy_tree_if_exists(
            self.repo_root / "outputs" / "curated_runs" / "09_linear_current_best_mp4",
            self.snapshots_dir / "current_mp4_baseline",
        )

    def update_status(self, phase: str, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "updated_at": _now(),
            "phase": phase,
            "completed_runs": self.completed_runs,
            "best_record": self.best_record,
            "search_anchor": self.search_anchor,
            "promoted_count": len(self.promoted_records),
            "best_run_path": str(self.best_dir),
            "promoted_dir": str(self.promoted_dir),
            "milestones_dir": str(self.milestones_dir),
            "finalists_dir": str(self.finalists_dir),
            "finished": False,
        }
        if extra:
            payload.update(extra)
        _write_json(self.status_path, payload)

    def run_candidate(self, phase: str, params: dict[str, Any], save_media: bool) -> tuple[dict[str, Any], Path]:
        self.run_index += 1
        params = _deep_merge(BASE_PARAMS, params)
        slug = f"{phase}_{self.run_index:04d}_{_slugify_params(params)}"
        artifact_dir = self.scratch_dir / slug
        started_at = _now()
        summary = _run_candidate(
            repo_root=self.repo_root,
            artifact_dir=artifact_dir,
            params=params,
            save_media=save_media,
            timeout_s=self.timeout_s,
        )
        score = float(_candidate_score(summary))
        duration_s = float(summary.get("duration_s") or 0.0)
        record = {
            "run_index": self.run_index,
            "phase": phase,
            "started_at": started_at,
            "completed_at": _now(),
            "params": params,
            "score": score,
            "duration_s": duration_s,
            "summary": summary,
            "artifact_dir": str(artifact_dir),
        }
        self.completed_runs += 1
        _append_history(self.history_path, record)

        anchor_score = float(self.search_anchor["score"]) if self.search_anchor is not None else float("-inf")
        anchor_duration = float(self.search_anchor["duration_s"]) if self.search_anchor is not None else 0.0
        if self.search_anchor is None or score > anchor_score or duration_s > anchor_duration:
            self.search_anchor = {
                "phase": phase,
                "params": params,
                "score": score,
                "duration_s": duration_s,
                "summary": summary,
            }
        return record, artifact_dir

    def promote(self, record: dict[str, Any], artifact_dir: Path, phase: str, reason: str, rerender_media: bool) -> dict[str, Any]:
        params = dict(record["params"])
        slug = f"{phase}_{record['run_index']:04d}_{_slugify_params(params)}"
        dest = self.promoted_dir / slug
        if rerender_media:
            summary = _run_candidate(
                repo_root=self.repo_root,
                artifact_dir=dest,
                params=params,
                save_media=True,
                timeout_s=self.timeout_s,
            )
            record["summary"] = summary
            record["score"] = float(_candidate_score(summary))
            record["duration_s"] = float(summary.get("duration_s") or 0.0)
            self.media_best_duration = max(self.media_best_duration, record["duration_s"])
        else:
            _copy_tree_if_exists(artifact_dir, dest)
        record["artifact_dir"] = str(dest)
        record["promotion_reason"] = reason
        (dest / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.promoted_records.append(record)

        best_score = float(self.best_record["score"]) if self.best_record is not None else float("-inf")
        if self.best_record is None or float(record["score"]) > best_score:
            self.best_record = {
                "phase": phase,
                "params": record["params"],
                "score": float(record["score"]),
                "duration_s": float(record["duration_s"]),
                "summary": record["summary"],
                "artifact_dir": record["artifact_dir"],
                "promotion_reason": reason,
            }
            _copy_tree_if_exists(dest, self.best_dir)
        return record

    def cleanup_artifact(self, artifact_dir: Path) -> None:
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="6-hour sequential overnight search for the linear OSQP MuJoCo recovery loop.")
    parser.add_argument("--minutes", type=float, default=360.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--base-dir", type=str, default="outputs/overnight_linear_session")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=1500)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = (repo_root / args.base_dir).resolve()
    runner = OvernightRunner(repo_root=repo_root, base_dir=base_dir, timeout_s=args.candidate_timeout_seconds, seed=args.seed)
    runner.snapshot_baselines()

    base_params = _load_autotune_seed(repo_root)
    runner.update_status("phase0", {"baseline_params": base_params, "phase_started_at": _now()})

    scale = max(float(args.minutes), 1.0) / 360.0
    phase1_budget = PHASE_MINUTES["phase1"] * scale * 60.0
    phase2_budget = PHASE_MINUTES["phase2"] * scale * 60.0
    stage_a_budget = PHASE_MINUTES["stage_a"] * scale * 60.0
    stage_b_budget = PHASE_MINUTES["stage_b"] * scale * 60.0
    stage_c_budget = PHASE_MINUTES["stage_c"] * scale * 60.0
    phase4_budget = PHASE_MINUTES["phase4"] * scale * 60.0

    start_time = time.time()

    # Phase 1: smoke trio on pure-forward command semantics.
    phase1_candidates = [
        dict(base_params),
        _deep_merge(
            base_params,
            {
                "startup_full_stance_time_s": 0.24
                if abs(float(base_params.get("startup_full_stance_time_s", 0.0)) - 0.20) < 1e-9
                else 0.20
            },
        ),
        _deep_merge(base_params, {"gait_step_freq": 0.50, "gait_duty_factor": 0.85}),
    ]
    dedup_phase1_candidates: list[dict[str, Any]] = []
    seen_phase1: set[str] = set()
    for candidate in phase1_candidates:
        key = _params_key(candidate)
        if key in seen_phase1:
            continue
        seen_phase1.add(key)
        dedup_phase1_candidates.append(candidate)
    phase1_candidates = dedup_phase1_candidates
    phase1_records: list[dict[str, Any]] = []
    phase1_end = start_time + phase1_budget
    for candidate in phase1_candidates:
        if time.time() >= phase1_end:
            break
        record, artifact_dir = runner.run_candidate("phase1", candidate, save_media=False)
        phase1_records.append(record)
        runner.cleanup_artifact(artifact_dir)
        runner.update_status("phase1", {"last_record": record})

    phase1_records.sort(key=lambda item: (float(item["duration_s"]), float(item["score"])), reverse=True)
    seed_base = phase1_records[0]["params"] if phase1_records else dict(base_params)
    seed_duration = float(phase1_records[0]["duration_s"]) if phase1_records else 0.0
    if phase1_records:
        _run_candidate(
            repo_root=repo_root,
            artifact_dir=runner.best_dir,
            params=seed_base,
            save_media=False,
            timeout_s=args.candidate_timeout_seconds,
        )
        runner.best_record = {
            "phase": "phase1",
            "params": seed_base,
            "score": float(phase1_records[0]["score"]),
            "duration_s": seed_duration,
            "summary": phase1_records[0]["summary"],
            "artifact_dir": str(runner.best_dir),
            "promotion_reason": "phase1_seed_base",
        }
        runner.update_status("phase1", {"seed_base": seed_base, "seed_duration_s": seed_duration})

    # Phase 2: structured 8-candidate pre-swing probe.
    phase2_records: list[dict[str, Any]] = []
    phase2_end = phase1_end + phase2_budget
    for candidate in _phase2_candidates(seed_base):
        if time.time() >= phase2_end:
            break
        record, artifact_dir = runner.run_candidate("phase2", candidate, save_media=False)
        phase2_records.append(record)
        promote = _promotion_ready_phase2(record["summary"])
        rerender_media = promote and float(record["duration_s"]) > max(seed_duration, runner.media_best_duration) * 1.15
        if promote:
            runner.promote(record, artifact_dir, "phase2", "phase2_gate", rerender_media=rerender_media)
        runner.cleanup_artifact(artifact_dir)
        runner.update_status("phase2", {"last_record": record})

    phase2_records.sort(key=lambda item: (float(item["duration_s"]), float(item["score"])), reverse=True)
    search_base = phase2_records[0]["params"] if phase2_records else seed_base
    if runner.best_record is None and phase2_records:
        top = phase2_records[0]
        _run_candidate(
            repo_root=repo_root,
            artifact_dir=runner.best_dir,
            params=top["params"],
            save_media=False,
            timeout_s=args.candidate_timeout_seconds,
        )
        runner.best_record = {
            "phase": "phase2",
            "params": top["params"],
            "score": float(top["score"]),
            "duration_s": float(top["duration_s"]),
            "summary": top["summary"],
            "artifact_dir": str(runner.best_dir),
            "promotion_reason": "fallback_best",
        }

    # Phase 3: staged search.
    stage_a_space = {
        "gait_step_freq": [0.45, 0.50, 0.55],
        "gait_duty_factor": [0.80, 0.85],
        "contact_latch_budget_steps": [6, 8, 10],
        "startup_full_stance_steps": [15, 20, 25],
        "support_centroid_x_gain": [4.0, 6.0, 8.0],
        "support_centroid_y_gain": [10.0, 12.0, 14.0],
        "preswing_shift_x_gain": [2.0, 4.0, 6.0],
        "preswing_shift_y_gain": [6.0, 8.0, 10.0],
        "grf_max_scale": [0.46, 0.50],
        "support_floor_ratio": [0.22, 0.24],
        "z_pos_gain": [28.0, 32.0],
        "roll_angle_gain": [24.0, 30.0, 36.0],
    }
    stage_b_space = dict(stage_a_space)
    stage_b_space.update({
        "preswing_phase_end": [0.25, 0.35, 0.45],
        "support_margin_ref": [0.02, 0.025, 0.03],
        "support_margin_vertical_boost": [0.0, 0.05, 0.1],
        "reduced_support_vertical_boost": [0.0, 0.05, 0.1],
    })
    stage_c_space = {
        "stance_target_blend": [0.0, 0.02, 0.04, 0.08],
        "latched_force_scale": [0.92, 0.95, 0.98],
        "roll_angle_gain": [24.0, 30.0, 36.0],
        "pitch_angle_gain": [24.0, 28.0, 32.0],
        "pitch_rate_gain": [6.0, 8.0, 10.0],
        "startup_full_stance_steps": [15, 20, 25],
        "support_margin_ref": [0.02, 0.025, 0.03],
        "support_margin_vertical_boost": [0.0, 0.05, 0.1],
    }

    search_anchor = dict(search_base)
    media_threshold_duration = max(seed_duration, runner.media_best_duration, 1e-6)

    def run_stage(
        stage_name: str,
        deadline: float,
        search_space: dict[str, list[Any]],
        gate_fn,
        local: bool,
        max_changes: int,
    ) -> None:
        nonlocal search_anchor, media_threshold_duration
        while time.time() < deadline:
            parent = runner.best_record["params"] if runner.best_record is not None else search_anchor
            candidate = _mutate_from_space(runner.rng, parent, search_space, max_changes=max_changes, local=local)
            record, artifact_dir = runner.run_candidate(stage_name, candidate, save_media=False)
            gate_pass = gate_fn(record["summary"])
            rerender_media = gate_pass and float(record["duration_s"]) > media_threshold_duration * 1.15
            if gate_pass:
                runner.promote(record, artifact_dir, stage_name, f"{stage_name}_gate", rerender_media=rerender_media)
                if rerender_media:
                    media_threshold_duration = max(media_threshold_duration, float(record["duration_s"]))
            if runner.search_anchor is not None:
                search_anchor = dict(runner.search_anchor["params"])
            runner.cleanup_artifact(artifact_dir)
            runner.update_status(stage_name, {"last_record": record})

    stage_a_end = phase2_end + stage_a_budget
    run_stage("stage_a", stage_a_end, stage_a_space, _promotion_ready_stage_a, local=False, max_changes=4)
    stage_b_end = stage_a_end + stage_b_budget
    run_stage("stage_b", stage_b_end, stage_b_space, _promotion_ready_stage_b, local=True, max_changes=3)
    stage_c_end = stage_b_end + stage_c_budget
    run_stage("stage_c", stage_c_end, stage_c_space, _promotion_ready_stage_c, local=True, max_changes=2)

    # Phase 4: final top-3 seed validation with media.
    phase4_deadline = stage_c_end + phase4_budget
    candidate_pool = _top_unique_param_sets(runner.promoted_records, limit=3)
    if not candidate_pool:
        fallback = runner.best_record if runner.best_record is not None else {
            "params": search_anchor,
            "score": float("-inf"),
            "duration_s": 0.0,
            "summary": {},
        }
        candidate_pool = [fallback]

    final_results: list[dict[str, Any]] = []
    for idx, record in enumerate(candidate_pool[:3], start=1):
        if time.time() >= phase4_deadline:
            break
        candidate_params = dict(record["params"])
        seed_runs: list[dict[str, Any]] = []
        for seed in (0, 1, 2):
            if time.time() >= phase4_deadline:
                break
            seed_params = dict(candidate_params)
            seed_params["seed"] = seed
            artifact_dir = runner.finalists_dir / f"candidate_{idx:02d}" / f"seed_{seed}"
            summary = _run_candidate(
                repo_root=repo_root,
                artifact_dir=artifact_dir,
                params=seed_params,
                save_media=True,
                timeout_s=args.candidate_timeout_seconds,
            )
            seed_record = {
                "candidate_index": idx,
                "seed": seed,
                "params": seed_params,
                "summary": summary,
                "duration_s": float(summary.get("duration_s") or 0.0),
                "mean_vx": float(summary.get("mean_vx") or 0.0),
                "overnight_success": _overnight_success(summary),
                "artifact_dir": str(artifact_dir),
            }
            seed_runs.append(seed_record)
            _append_history(runner.history_path, {"phase": "phase4", **seed_record})
            runner.completed_runs += 1
            runner.update_status("phase4", {"last_record": seed_record})
        if seed_runs:
            durations = [item["duration_s"] for item in seed_runs]
            mean_vx_values = [item["mean_vx"] for item in seed_runs]
            final_results.append({
                "candidate_index": idx,
                "params": candidate_params,
                "seed_runs": seed_runs,
                "success_count": int(sum(1 for item in seed_runs if item["overnight_success"])),
                "median_duration_s": float(np.median(durations)),
                "median_mean_vx": float(np.median(mean_vx_values)),
            })

    final_results.sort(
        key=lambda item: (item["success_count"], item["median_duration_s"], item["median_mean_vx"]),
        reverse=True,
    )
    winner = final_results[0] if final_results else None
    winner_success = bool(winner is not None and int(winner["success_count"]) >= 2)
    if winner is not None:
        winning_seed = winner["seed_runs"][0]
        _copy_tree_if_exists(Path(winning_seed["artifact_dir"]), runner.best_dir)

    runner.update_status(
        "finished",
        {
            "finished": True,
            "winner": winner,
            "winner_success": winner_success,
            "final_results": final_results,
            "phase_finished_at": _now(),
        },
    )


if __name__ == "__main__":
    main()
