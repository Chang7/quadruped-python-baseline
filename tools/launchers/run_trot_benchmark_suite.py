from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_ROOT = ROOT / "outputs" / "archive" / "raw_runs" / "20260408_trot_benchmark_suite"
DEFAULT_REPORT_ROOT = ROOT / "outputs" / "report_progress_explainer" / "trot_benchmark_suite_20260408"


@dataclass(frozen=True)
class TrotBenchmarkCase:
    name: str
    label: str
    controller: str
    seconds: int
    speed: float
    yaw_rate: float = 0.0
    dynamic_trot_profile: str | None = None
    disturbance_pulses: tuple[str, ...] = field(default_factory=tuple)

    def command(self, artifact_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "simulation.run_linear_osqp",
            "--controller",
            self.controller,
            "--gait",
            "trot",
            "--seconds",
            str(self.seconds),
            "--speed",
            str(self.speed),
            "--yaw-rate",
            str(self.yaw_rate),
            "--artifact-dir",
            str(artifact_dir.relative_to(ROOT)),
        ]
        if self.dynamic_trot_profile is not None:
            cmd.extend(["--dynamic-trot-profile", self.dynamic_trot_profile])
        for pulse in self.disturbance_pulses:
            cmd.extend(["--disturbance-pulse", pulse])
        return cmd


CASES = [
    TrotBenchmarkCase(
        name="stock_trot_straight_4s",
        label="trot straight 4 s",
        controller="sampling",
        seconds=4,
        speed=0.12,
    ),
    TrotBenchmarkCase(
        name="linear_trot_straight_4s",
        label="trot straight 4 s",
        controller="linear_osqp",
        seconds=4,
        speed=0.12,
        dynamic_trot_profile="generic",
    ),
    TrotBenchmarkCase(
        name="stock_trot_turn_4s_y04",
        label="trot turn 4 s",
        controller="sampling",
        seconds=4,
        speed=0.12,
        yaw_rate=0.4,
    ),
    TrotBenchmarkCase(
        name="linear_trot_turn_4s_y04",
        label="trot turn 4 s",
        controller="linear_osqp",
        seconds=4,
        speed=0.12,
        yaw_rate=0.4,
        dynamic_trot_profile="generic",
    ),
    TrotBenchmarkCase(
        name="stock_trot_disturb_4s_x48",
        label="trot disturb 4 s",
        controller="sampling",
        seconds=4,
        speed=0.12,
        disturbance_pulses=("x:0.5:0.25:4.0", "x:2.3:0.25:8.0"),
    ),
    TrotBenchmarkCase(
        name="linear_trot_disturb_4s_x48",
        label="trot disturb 4 s",
        controller="linear_osqp",
        seconds=4,
        speed=0.12,
        dynamic_trot_profile="generic",
        disturbance_pulses=("x:0.5:0.25:4.0", "x:2.3:0.25:8.0"),
    ),
    TrotBenchmarkCase(
        name="stock_trot_straight_20s",
        label="trot straight 20 s",
        controller="sampling",
        seconds=20,
        speed=0.12,
    ),
    TrotBenchmarkCase(
        name="linear_trot_straight_20s",
        label="trot straight 20 s",
        controller="linear_osqp",
        seconds=20,
        speed=0.12,
        dynamic_trot_profile="generic",
    ),
]


def _summary_payload(case: TrotBenchmarkCase, summary_path: Path, output_dir: Path, cmd: list[str]) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "name": case.name,
        "label": case.label,
        "controller": case.controller,
        "seconds": case.seconds,
        "speed": case.speed,
        "yaw_rate": case.yaw_rate,
        "dynamic_trot_profile": case.dynamic_trot_profile,
        "disturbance_pulses": list(case.disturbance_pulses),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "command": cmd,
        "terminated_any": bool(summary["terminated_any"]),
        "invalid_contact": summary.get("meta", {}).get("invalid_contact_keys", []),
        "metrics": {
            "duration_s": float(summary["duration_s"]),
            "mean_vx": float(summary["mean_vx"]),
            "mean_base_z": float(summary["mean_base_z"]),
            "mean_abs_roll": float(summary["mean_abs_roll"]),
            "mean_abs_pitch": float(summary["mean_abs_pitch"]),
        },
    }


def _run_case(case: TrotBenchmarkCase, suite_root: Path, skip_existing: bool) -> dict[str, object]:
    output_dir = suite_root / case.name
    summary_path = output_dir / "episode_000" / "summary.json"
    cmd = case.command(output_dir)
    if not (skip_existing and summary_path.exists()):
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    return _summary_payload(case, summary_path, output_dir, cmd)


def _write_manifest(suite_root: Path, cases: list[dict[str, object]]) -> Path:
    manifest = {
        "suite_root": str(suite_root),
        "cases": cases,
    }
    manifest_path = suite_root / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_summary_json(suite_root: Path, cases: list[dict[str, object]]) -> Path:
    by_name = {case["name"]: case for case in cases}
    summary = {
        "short_horizon": {
            "straight_4s": {
                "stock": by_name["stock_trot_straight_4s"]["metrics"],
                "linear": by_name["linear_trot_straight_4s"]["metrics"],
            },
            "turn_4s_y04": {
                "stock": by_name["stock_trot_turn_4s_y04"]["metrics"],
                "linear": by_name["linear_trot_turn_4s_y04"]["metrics"],
            },
            "disturb_4s_x48": {
                "stock": by_name["stock_trot_disturb_4s_x48"]["metrics"],
                "linear": by_name["linear_trot_disturb_4s_x48"]["metrics"],
            },
        },
        "long_straight": {
            "stock": by_name["stock_trot_straight_20s"]["metrics"],
            "linear": by_name["linear_trot_straight_20s"]["metrics"],
        },
    }
    out_path = suite_root / "suite_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def _run_dashboard(manifest_path: Path, report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "report_assets" / "make_trot_benchmark_dashboard.py"),
        "--manifest",
        str(manifest_path),
        "--out-dir",
        str(report_root),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed trot benchmark suite and generate a compact dashboard.")
    parser.add_argument("--suite-root", type=str, default=str(DEFAULT_SUITE_ROOT))
    parser.add_argument("--report-root", type=str, default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing suite case outputs when summary.json already exists.")
    args = parser.parse_args()

    suite_root = Path(args.suite_root).resolve()
    report_root = Path(args.report_root).resolve()
    suite_root.mkdir(parents=True, exist_ok=True)

    case_payloads = []
    for case in CASES:
        print(f"[suite] {case.name}")
        case_payloads.append(_run_case(case, suite_root, skip_existing=args.skip_existing))

    manifest_path = _write_manifest(suite_root, case_payloads)
    summary_path = _write_summary_json(suite_root, case_payloads)
    _run_dashboard(manifest_path, report_root)

    print(manifest_path)
    print(summary_path)
    print(report_root)


if __name__ == "__main__":
    main()
