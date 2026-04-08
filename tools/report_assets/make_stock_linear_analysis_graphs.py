from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "report_progress_explainer" / "stock_vs_linear_analysis_20260408"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "stock_trot_straight": ROOT / "outputs" / "curated_runs" / "stock_sampling_trot_4s_s012_isolated_recheck" / "episode_000" / "summary.json",
    "stock_trot_turn": ROOT / "outputs" / "curated_runs" / "stock_sampling_trot_turn_4s_y04_recheck" / "episode_000" / "summary.json",
    "stock_trot_disturb": ROOT / "outputs" / "curated_runs" / "stock_sampling_trot_disturb_4s_x48_recheck" / "episode_000" / "summary.json",
    "stock_crawl": ROOT / "outputs" / "curated_runs" / "stock_sampling_crawl_4s_s012_isolated_recheck" / "episode_000" / "summary.json",
    "linear_trot_straight": ROOT / "outputs" / "archive" / "raw_runs" / "20260408_stock_linear_graph_inputs" / "linear_trot_straight_4s" / "episode_000" / "summary.json",
    "linear_trot_turn": ROOT / "outputs" / "archive" / "raw_runs" / "20260408_stock_linear_graph_inputs" / "linear_trot_turn_4s" / "episode_000" / "summary.json",
    "linear_trot_disturb": ROOT / "outputs" / "archive" / "raw_runs" / "20260408_stock_linear_graph_inputs" / "linear_trot_disturb_4s" / "episode_000" / "summary.json",
    "linear_crawl": ROOT / "outputs" / "archive" / "raw_runs" / "20260408_stock_linear_graph_inputs" / "linear_crawl_4s" / "episode_000" / "summary.json",
}

STOCK_COLOR = "#4C78A8"
LINEAR_COLOR = "#E45756"
NEUTRAL_COLOR = "#7F7F7F"


def _load_runs() -> dict[str, dict]:
    return {key: json.loads(path.read_text(encoding="utf-8")) for key, path in RUNS.items()}


def _invalid_contact_text(summary: dict) -> str:
    invalid = summary.get("meta", {}).get("invalid_contact_keys", [])
    if not invalid:
        return "none"
    return invalid[0].replace("world:0_", "").replace(":", " ")


def _scenario_payload(data: dict[str, dict], scenario: str) -> tuple[dict, dict]:
    return data[f"stock_{scenario}"], data[f"linear_{scenario}"]


def _add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
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


def make_trot_overview(data: dict[str, dict]) -> Path:
    scenarios = [
        ("trot_straight", "Straight"),
        ("trot_turn", "Turn"),
        ("trot_disturb", "Disturb"),
    ]
    metrics = [
        ("mean_vx", "mean vx [m/s]"),
        ("mean_base_z", "mean base z [m]"),
        ("mean_abs_pitch", "mean |pitch| [rad]"),
        ("mean_abs_roll", "mean |roll| [rad]"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    axes = axes.ravel()
    x = np.arange(len(scenarios))
    width = 0.34

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        stock_vals = []
        linear_vals = []
        for scenario_key, _ in scenarios:
            stock, linear = _scenario_payload(data, scenario_key)
            stock_vals.append(stock[metric_key])
            linear_vals.append(linear[metric_key])

        bars_stock = ax.bar(x - width / 2, stock_vals, width, label="stock sampling", color=STOCK_COLOR)
        bars_linear = ax.bar(x + width / 2, linear_vals, width, label="linear_osqp", color=LINEAR_COLOR)
        _add_bar_labels(ax, bars_stock)
        _add_bar_labels(ax, bars_linear)
        ax.set_xticks(x, [label for _, label in scenarios])
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Stock vs linear(custom): 4 s trot comparison\n"
        "robot=aliengo | scene=flat | gait=trot | straight/turn/disturbance diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    notes = []
    for scenario_key, label in scenarios:
        stock, linear = _scenario_payload(data, scenario_key)
        notes.append(
            f"{label}: stock={'ok' if not stock['terminated_any'] else 'terminated'} "
            f"| linear={'ok' if not linear['terminated_any'] else 'terminated'}"
        )
    fig.text(0.02, 0.02, " | ".join(notes), fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    out_path = OUT_DIR / "trot_stock_vs_linear_overview.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_crawl_diagnostic(data: dict[str, dict]) -> Path:
    stock = data["stock_crawl"]
    linear = data["linear_crawl"]
    labels = [
        "duration [s]",
        "mean base z [m]",
        "mean |roll| [rad]",
        "mean |pitch| [rad]",
        "actual swing total",
        "rear TD support",
    ]
    stock_vals = [
        stock["duration_s"],
        stock["mean_base_z"],
        stock["mean_abs_roll"],
        stock["mean_abs_pitch"],
        stock["actual_swing_realization_total"],
        stock["rear_touchdown_support_mean"],
    ]
    linear_vals = [
        linear["duration_s"],
        linear["mean_base_z"],
        linear["mean_abs_roll"],
        linear["mean_abs_pitch"],
        linear["actual_swing_realization_total"],
        linear["rear_touchdown_support_mean"],
    ]

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    x = np.arange(len(labels))
    width = 0.34
    bars_stock = ax.bar(x - width / 2, stock_vals, width, label="stock sampling", color=STOCK_COLOR)
    bars_linear = ax.bar(x + width / 2, linear_vals, width, label="linear_osqp", color=LINEAR_COLOR)
    _add_bar_labels(ax, bars_stock)
    _add_bar_labels(ax, bars_linear)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_title("4 s crawl diagnostic comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    rows = [
        ["stock sampling", f"{stock['duration_s']:.3f}", str(stock["terminated_any"]), _invalid_contact_text(stock)],
        ["linear_osqp", f"{linear['duration_s']:.3f}", str(linear["terminated_any"]), _invalid_contact_text(linear)],
    ]
    table = ax2.table(
        cellText=rows,
        colLabels=["controller", "duration [s]", "terminated", "invalid contact"],
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
        "Stock vs linear(custom): crawl diagnostic\n"
        "same 4 s horizon, but stock crawl is also not a clean stable baseline in this setting",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.02,
        0.02,
        "Interpretation: crawl remains a contact-transition diagnostic, not a primary success benchmark.",
        fontsize=10,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    out_path = OUT_DIR / "crawl_stock_vs_linear_diagnostic.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_summary_table(data: dict[str, dict]) -> Path:
    ordered = [
        ("stock_trot_straight", "stock", "trot straight"),
        ("linear_trot_straight", "linear", "trot straight"),
        ("stock_trot_turn", "stock", "trot turn"),
        ("linear_trot_turn", "linear", "trot turn"),
        ("stock_trot_disturb", "stock", "trot disturb"),
        ("linear_trot_disturb", "linear", "trot disturb"),
        ("stock_crawl", "stock", "crawl"),
        ("linear_crawl", "linear", "crawl"),
    ]
    rows = []
    for key, ctrl, scenario in ordered:
        summary = data[key]
        rows.append(
            [
                ctrl,
                scenario,
                f"{summary['duration_s']:.3f}",
                str(summary["terminated_any"]),
                f"{summary['mean_vx']:.3f}",
                f"{summary['mean_base_z']:.3f}",
                f"{summary['mean_abs_roll']:.3f}",
                f"{summary['mean_abs_pitch']:.3f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(14.5, 4.6))
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
        "Stock vs linear(custom): summary table for matched 4 s checks",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = OUT_DIR / "stock_vs_linear_summary_table.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_readme(trot_path: Path, crawl_path: Path, table_path: Path) -> Path:
    text = f"""# Stock vs linear(custom) analysis bundle

This folder contains matched-horizon (`4 s`) comparison figures between the stock sampling controller and the custom `linear_osqp` controller.

Files:
- `{table_path.name}`: compact summary table across trot/crawl checks
- `{trot_path.name}`: main trot comparison across straight / turn / disturbance
- `{crawl_path.name}`: crawl diagnostic comparison

Interpretation:
- `trot` is the main benchmark because stock `trot` is stable and emphasized in the upstream paper/repo.
- `crawl` is kept as a contact-transition diagnostic because stock crawl also terminates early in the current local setting.
"""
    out_path = OUT_DIR / "README.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> None:
    data = _load_runs()
    table_path = make_summary_table(data)
    trot_path = make_trot_overview(data)
    crawl_path = make_crawl_diagnostic(data)
    write_readme(trot_path, crawl_path, table_path)
    print(table_path)
    print(trot_path)
    print(crawl_path)


if __name__ == "__main__":
    main()
