from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STOCK_COLOR = "#4C78A8"
LINEAR_COLOR = "#E45756"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cases_by_name(manifest: dict) -> dict[str, dict]:
    return {case["name"]: case for case in manifest["cases"]}


def _invalid_contact_text(case: dict) -> str:
    invalid = case.get("invalid_contact", [])
    if not invalid:
        return "none"
    return str(invalid[0]).replace("world:0_", "").replace(":", " ")


def _bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _make_short_horizon(cases: dict[str, dict], out_dir: Path) -> Path:
    scenario_defs = [
        ("straight_4s", "Straight"),
        ("turn_4s_y04", "Turn"),
        ("disturb_4s_x48", "Disturb"),
    ]
    metrics = [
        ("mean_vx", "mean vx [m/s]"),
        ("mean_base_z", "mean base z [m]"),
        ("mean_abs_pitch", "mean |pitch| [rad]"),
        ("mean_abs_roll", "mean |roll| [rad]"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    axes = axes.ravel()
    x = np.arange(len(scenario_defs))
    width = 0.34

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        stock_vals = [cases[f"stock_trot_{scenario_key}"]["metrics"][metric_key] for scenario_key, _ in scenario_defs]
        linear_vals = [cases[f"linear_trot_{scenario_key}"]["metrics"][metric_key] for scenario_key, _ in scenario_defs]
        stock_bars = ax.bar(x - width / 2, stock_vals, width, color=STOCK_COLOR, label="stock sampling")
        linear_bars = ax.bar(x + width / 2, linear_vals, width, color=LINEAR_COLOR, label="linear_osqp")
        _bar_labels(ax, stock_bars)
        _bar_labels(ax, linear_bars)
        ax.set_xticks(x, [label for _, label in scenario_defs])
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper left")
    status_line = " | ".join(
        f"{label}: stock={'ok' if not cases[f'stock_trot_{scenario_key}']['terminated_any'] else 'terminated'} "
        f"| linear={'ok' if not cases[f'linear_trot_{scenario_key}']['terminated_any'] else 'terminated'}"
        for scenario_key, label in scenario_defs
    )

    fig.suptitle(
        "Trot benchmark suite: short-horizon comparison\n"
        "matched 4 s checks for straight / turn / disturbance",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.02, 0.02, status_line, fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out_path = out_dir / "trot_benchmark_short_horizon.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _make_long_straight(cases: dict[str, dict], out_dir: Path) -> Path:
    stock = cases["stock_trot_straight_20s"]
    linear = cases["linear_trot_straight_20s"]
    labels = ["duration [s]", "mean vx", "mean z", "mean |roll|", "mean |pitch|"]
    stock_vals = [
        stock["metrics"]["duration_s"],
        stock["metrics"]["mean_vx"],
        stock["metrics"]["mean_base_z"],
        stock["metrics"]["mean_abs_roll"],
        stock["metrics"]["mean_abs_pitch"],
    ]
    linear_vals = [
        linear["metrics"]["duration_s"],
        linear["metrics"]["mean_vx"],
        linear["metrics"]["mean_base_z"],
        linear["metrics"]["mean_abs_roll"],
        linear["metrics"]["mean_abs_pitch"],
    ]

    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.65], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    x = np.arange(len(labels))
    width = 0.34
    stock_bars = ax.bar(x - width / 2, stock_vals, width, color=STOCK_COLOR, label="stock sampling")
    linear_bars = ax.bar(x + width / 2, linear_vals, width, color=LINEAR_COLOR, label="linear_osqp")
    _bar_labels(ax, stock_bars)
    _bar_labels(ax, linear_bars)
    ax.set_xticks(x, labels)
    ax.set_title("20 s straight-line trot comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    rows = [
        ["stock sampling", str(stock["terminated_any"]), _invalid_contact_text(stock)],
        ["linear_osqp", str(linear["terminated_any"]), _invalid_contact_text(linear)],
    ]
    table = ax2.table(
        cellText=rows,
        colLabels=["controller", "terminated", "invalid contact"],
        loc="center",
        cellLoc="center",
    )
    ax2.axis("off")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8EEF7")
        elif row == 1:
            cell.set_facecolor("#F4F8FF")
        elif row == 2:
            cell.set_facecolor("#FFF5EF")

    fig.suptitle(
        "Trot benchmark suite: long straight-line comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_path = out_dir / "trot_benchmark_long_straight.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _make_summary_table(cases: dict[str, dict], out_dir: Path) -> Path:
    ordered = [
        "stock_trot_straight_4s",
        "linear_trot_straight_4s",
        "stock_trot_turn_4s_y04",
        "linear_trot_turn_4s_y04",
        "stock_trot_disturb_4s_x48",
        "linear_trot_disturb_4s_x48",
        "stock_trot_straight_20s",
        "linear_trot_straight_20s",
    ]
    rows = []
    for name in ordered:
        case = cases[name]
        rows.append(
            [
                case["controller"],
                case["label"],
                f"{case['metrics']['duration_s']:.3f}",
                str(case["terminated_any"]),
                f"{case['metrics']['mean_vx']:.3f}",
                f"{case['metrics']['mean_base_z']:.3f}",
                f"{case['metrics']['mean_abs_roll']:.3f}",
                f"{case['metrics']['mean_abs_pitch']:.3f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "controller",
            "scenario",
            "duration [s]",
            "terminated",
            "mean vx",
            "mean z",
            "mean |roll|",
            "mean |pitch|",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8EEF7")
        elif row % 2 == 1:
            cell.set_facecolor("#F7FAFD")
        else:
            cell.set_facecolor("#FFF8F4")

    fig.suptitle(
        "Trot benchmark suite: summary table",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / "trot_benchmark_summary_table.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _write_readme(manifest: dict, out_dir: Path, generated: list[Path]) -> Path:
    lines = [
        "# Trot benchmark suite",
        "",
        f"Source suite root: `{manifest['suite_root']}`",
        "",
        "Generated files:",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "Coverage:",
            "- short-horizon matched 4 s straight / turn / disturbance checks",
            "- long-horizon 20 s straight-line check",
            "",
            "Interpretation:",
            "- use this bundle as the main trot benchmark index",
            "- keep crawl separate as a contact-transition diagnostic",
            "",
        ]
    )
    out_path = out_dir / "README.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact dashboard from a trot benchmark suite manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to suite_manifest.json")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for generated PNG assets")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    cases = _cases_by_name(manifest)
    summary_path = _make_summary_table(cases, out_dir)
    short_path = _make_short_horizon(cases, out_dir)
    long_path = _make_long_straight(cases, out_dir)
    readme_path = _write_readme(manifest, out_dir, [summary_path, short_path, long_path])

    print(summary_path)
    print(short_path)
    print(long_path)
    print(readme_path)


if __name__ == "__main__":
    main()
