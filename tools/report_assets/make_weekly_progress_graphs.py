from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "report_progress_explainer" / "weekly_progress_20260410"

COLORS = {
    "before": "#7F7F7F",
    "after": "#E45756",
    "stock": "#4C78A8",
    "failure": "#B279A2",
}


RUNS = {
    "trot_turn_before": ROOT / "outputs" / "curated_runs" / "predecessors" / "trot_current_turn_default_10s" / "episode_000" / "summary.json",
    "trot_turn_after": ROOT / "outputs" / "curated_runs" / "current" / "trot_turn_10s_g025_pitchoff003" / "episode_000" / "summary.json",
    "trot_turn_stock": ROOT / "outputs" / "curated_runs" / "stock_baselines" / "stock_trot_turn_10s_weeklyref" / "episode_000" / "summary.json",
    "trot_straight_before": ROOT / "outputs" / "curated_runs" / "predecessors" / "trot_current_straight_default_20s" / "episode_000" / "summary.json",
    "trot_straight_after": ROOT / "outputs" / "curated_runs" / "current" / "trot_straight_20s_g025_pitchoff003" / "episode_000" / "summary.json",
    "trot_straight_stock": ROOT / "outputs" / "curated_runs" / "stock_baselines" / "stock_trot_straight_20s_weeklyref" / "episode_000" / "summary.json",
    "trot_disturb_before": ROOT / "outputs" / "archive" / "raw_runs" / "trot_20260409" / "quality_sweeps" / "trot_disturb_4s_baseline_before" / "episode_000" / "summary.json",
    "trot_disturb_after": ROOT / "outputs" / "curated_runs" / "current" / "trot_disturb_4s_g025_pitchoff003" / "episode_000" / "summary.json",
    "trot_disturb_stock": ROOT / "outputs" / "curated_runs" / "stock_baselines" / "stock_sampling_trot_disturb_4s_x48_recheck" / "episode_000" / "summary.json",
    "turn_symroll_fail": ROOT / "outputs" / "archive" / "raw_runs" / "trot_20260409" / "quality_sweeps" / "trot_turn_10s_symroll_test" / "episode_000" / "summary.json",
    "disturb_symroll_fail": ROOT / "outputs" / "archive" / "raw_runs" / "trot_20260409" / "quality_sweeps" / "trot_disturb_4s_symroll_test" / "episode_000" / "summary.json",
    "crawl_current": ROOT / "outputs" / "curated_runs" / "current" / "crawl_current_default_20s" / "episode_000" / "summary.json",
    "crawl_weakleg_fail": ROOT / "outputs" / "archive" / "raw_runs" / "crawl_20260409" / "crawl_weakleg_share_ref040_test" / "episode_000" / "summary.json",
    "crawl_stock": ROOT / "outputs" / "curated_runs" / "stock_baselines" / "stock_sampling_crawl_4s_s012_isolated_recheck" / "episode_000" / "summary.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _loads() -> dict[str, dict]:
    return {name: _load(path) for name, path in RUNS.items()}


def _invalid(summary: dict) -> str:
    invalid = summary.get("meta", {}).get("invalid_contact_keys", [])
    if not invalid:
        return "none"
    return invalid[0].replace("world:0_", "").replace(":", " ")


def _bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_trot_improvement(data: dict[str, dict]) -> Path:
    rows = [
        ("Straight 20 s\nmean |pitch|", "mean_abs_pitch", data["trot_straight_before"], data["trot_straight_after"], data["trot_straight_stock"]),
        ("Turn 10 s\nmean |roll|", "mean_abs_roll", data["trot_turn_before"], data["trot_turn_after"], data["trot_turn_stock"]),
        ("Turn 10 s\nmean |pitch|", "mean_abs_pitch", data["trot_turn_before"], data["trot_turn_after"], data["trot_turn_stock"]),
        ("Disturb 4 s\nmean |roll|", "mean_abs_roll", data["trot_disturb_before"], data["trot_disturb_after"], data["trot_disturb_stock"]),
        ("Disturb 4 s\nmean |pitch|", "mean_abs_pitch", data["trot_disturb_before"], data["trot_disturb_after"], data["trot_disturb_stock"]),
    ]

    labels = [row[0] for row in rows]
    before = [row[2][row[1]] for row in rows]
    after = [row[3][row[1]] for row in rows]
    stock = [row[4][row[1]] for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6.8))
    x = np.arange(len(labels))
    width = 0.24
    b1 = ax.bar(x - width, before, width, label="this-week start", color=COLORS["before"])
    b2 = ax.bar(x, after, width, label="current tuned", color=COLORS["after"])
    b3 = ax.bar(x + width, stock, width, label="stock reference", color=COLORS["stock"])
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    _bar_labels(ax, b3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("metric value")
    ax.set_title("This week: trot quality improvements", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    notes = []
    for label, _, s_before, s_after, _ in rows:
        if "pitch" in label or "roll" in label:
            base = s_before["mean_abs_pitch"] if "pitch" in label else s_before["mean_abs_roll"]
            new = s_after["mean_abs_pitch"] if "pitch" in label else s_after["mean_abs_roll"]
            improvement = (base - new) / base * 100.0
            notes.append(f"{label.split()[0]}: {improvement:.1f}% better")
    fig.text(0.02, 0.02, " | ".join(notes), fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    out = OUT_DIR / "weekly_trot_improvement_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_failure_ablations(data: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    turn_labels = ["before", "Q-roll sym", "tuned", "stock"]
    turn_roll = [
        data["trot_turn_before"]["mean_abs_roll"],
        data["turn_symroll_fail"]["mean_abs_roll"],
        data["trot_turn_after"]["mean_abs_roll"],
        data["trot_turn_stock"]["mean_abs_roll"],
    ]
    turn_pitch = [
        data["trot_turn_before"]["mean_abs_pitch"],
        data["turn_symroll_fail"]["mean_abs_pitch"],
        data["trot_turn_after"]["mean_abs_pitch"],
        data["trot_turn_stock"]["mean_abs_pitch"],
    ]
    x = np.arange(len(turn_labels))
    width = 0.34
    bars1 = axes[0].bar(x - width / 2, turn_roll, width, label="mean |roll|", color=COLORS["failure"])
    bars2 = axes[0].bar(x + width / 2, turn_pitch, width, label="mean |pitch|", color=COLORS["after"])
    _bar_labels(axes[0], bars1)
    _bar_labels(axes[0], bars2)
    axes[0].set_xticks(x, turn_labels)
    axes[0].set_title("Turn ablation: roll-Q symmetry failed")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    disturb_labels = ["before", "Q-roll sym", "tuned", "stock"]
    disturb_roll = [
        data["trot_disturb_before"]["mean_abs_roll"],
        data["disturb_symroll_fail"]["mean_abs_roll"],
        data["trot_disturb_after"]["mean_abs_roll"],
        data["trot_disturb_stock"]["mean_abs_roll"],
    ]
    disturb_pitch = [
        data["trot_disturb_before"]["mean_abs_pitch"],
        data["disturb_symroll_fail"]["mean_abs_pitch"],
        data["trot_disturb_after"]["mean_abs_pitch"],
        data["trot_disturb_stock"]["mean_abs_pitch"],
    ]
    x2 = np.arange(len(disturb_labels))
    bars3 = axes[1].bar(x2 - width / 2, disturb_roll, width, label="mean |roll|", color=COLORS["failure"])
    bars4 = axes[1].bar(x2 + width / 2, disturb_pitch, width, label="mean |pitch|", color=COLORS["after"])
    _bar_labels(axes[1], bars3)
    _bar_labels(axes[1], bars4)
    axes[1].set_xticks(x2, disturb_labels)
    axes[1].set_title("Disturb ablation: Q symmetry vs tuned gains")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Failed hypothesis vs effective direction\n"
        "raising roll-side MPC cost did not solve the gap; tuned existing gains did",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "weekly_failure_ablation_trot.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_crawl_failure_story(data: dict[str, dict]) -> Path:
    labels = ["stock crawl 4 s", "current default 20 s", "weak-leg fail"]
    durations = [
        data["crawl_stock"]["duration_s"],
        data["crawl_current"]["duration_s"],
        data["crawl_weakleg_fail"]["duration_s"],
    ]
    rolls = [
        data["crawl_stock"]["mean_abs_roll"],
        data["crawl_current"]["mean_abs_roll"],
        data["crawl_weakleg_fail"]["mean_abs_roll"],
    ]
    heights = [
        data["crawl_stock"]["mean_base_z"],
        data["crawl_current"]["mean_base_z"],
        data["crawl_weakleg_fail"]["mean_base_z"],
    ]

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.8], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    x = np.arange(len(labels))
    width = 0.24
    b1 = ax.bar(x - width, durations, width, label="duration [s]", color=COLORS["stock"])
    b2 = ax.bar(x, rolls, width, label="mean |roll|", color=COLORS["failure"])
    b3 = ax.bar(x + width, heights, width, label="mean base z [m]", color=COLORS["after"])
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    _bar_labels(ax, b3)
    ax.set_xticks(x, labels)
    ax.set_title("Crawl diagnostic: broad weak-leg support caused regression")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    rows = [
        ["stock crawl", f"{data['crawl_stock']['duration_s']:.3f}", _invalid(data["crawl_stock"])],
        ["current default", f"{data['crawl_current']['duration_s']:.3f}", _invalid(data["crawl_current"])],
        ["weak_leg_share_ref=0.40", f"{data['crawl_weakleg_fail']['duration_s']:.3f}", _invalid(data["crawl_weakleg_fail"])],
    ]
    table = ax2.table(
        cellText=rows,
        colLabels=["case", "duration [s]", "first invalid contact"],
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
        elif row == 3:
            cell.set_facecolor("#FDECEC")

    fig.suptitle(
        "Crawl debugging story\n"
        "stock crawl is not a clean gold baseline here, and aggressive weak-leg rescue over-triggered",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "weekly_crawl_failure_story.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_stock_vs_custom(data: dict[str, dict]) -> Path:
    scenarios = [
        ("Straight 20 s", data["trot_straight_stock"], data["trot_straight_after"]),
        ("Turn 10 s", data["trot_turn_stock"], data["trot_turn_after"]),
        ("Disturb 4 s", data["trot_disturb_stock"], data["trot_disturb_after"]),
    ]
    metrics = [
        ("mean_vx", "mean vx"),
        ("mean_abs_roll", "mean |roll|"),
        ("mean_abs_pitch", "mean |pitch|"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    x = np.arange(len(scenarios))
    width = 0.34

    for ax, (metric, title) in zip(axes, metrics):
        stock_vals = [s[1][metric] for s in scenarios]
        custom_vals = [s[2][metric] for s in scenarios]
        b1 = ax.bar(x - width / 2, stock_vals, width, label="stock", color=COLORS["stock"])
        b2 = ax.bar(x + width / 2, custom_vals, width, label="custom current", color=COLORS["after"])
        _bar_labels(ax, b1)
        _bar_labels(ax, b2)
        ax.set_xticks(x, [s[0] for s in scenarios])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.suptitle(
        "Current stock vs custom reference comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "weekly_stock_vs_custom.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_readme(generated: list[Path]) -> Path:
    lines = [
        "# Weekly progress graphs (2026-04-10)",
        "",
        "This bundle is organized around this week's report story:",
        "- what improved in trot",
        "- which hypotheses failed",
        "- why crawl remains a diagnostic rather than a primary benchmark",
        "- how the current custom results compare against stock references",
        "",
        "Generated files:",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "- `weekly_progress_report_ko_20260410.docx`",
            "",
            "How to read them:",
            "- `weekly_trot_improvement_summary.png`",
            "  - week-over-week trot quality summary",
            "  - straight 20 s pitch: `0.052 -> 0.031`",
            "  - turn 10 s roll: `0.024 -> 0.018`",
            "  - turn 10 s pitch: `0.059 -> 0.043`",
            "  - disturb 4 s roll/pitch: `0.024 -> 0.020`, `0.053 -> 0.034`",
            "- `weekly_failure_ablation_trot.png`",
            "  - failed hypothesis: symmetric roll/pitch MPC weighting did not close the turn-roll gap",
            "  - successful direction: existing `dynamic_fy_roll_gain` and `pitch_ref_offset` tuning",
            "- `weekly_crawl_failure_story.png`",
            "  - current crawl default vs failed `weak_leg_share_ref=0.40` ablation",
            "  - also shows why crawl is currently a diagnostic scenario rather than a primary benchmark",
            "- `weekly_stock_vs_custom.png`",
            "  - same-horizon stock vs custom comparison for straight/turn/disturbance trot",
            "  - plus crawl diagnostic comparison",
            "",
            "Supporting clips:",
            "- `clips/trot_straight_current.gif`",
            "- `clips/trot_turn_current.gif`",
            "- `clips/trot_disturb_current.gif`",
            "- `clips/crawl_current.gif`",
            "",
            "Key source runs:",
            "- `outputs/curated_runs/predecessors/trot_current_turn_default_10s`",
            "- `outputs/curated_runs/current/trot_turn_10s_g025_pitchoff003`",
            "- `outputs/curated_runs/predecessors/trot_current_straight_default_20s`",
            "- `outputs/curated_runs/current/trot_straight_20s_g025_pitchoff003`",
            "- `outputs/archive/raw_runs/trot_20260409/quality_sweeps/trot_disturb_4s_baseline_before`",
            "- `outputs/curated_runs/current/trot_disturb_4s_g025_pitchoff003`",
            "- `outputs/curated_runs/stock_baselines/stock_trot_turn_10s_weeklyref`",
            "- `outputs/curated_runs/stock_baselines/stock_trot_straight_20s_weeklyref`",
            "- `outputs/curated_runs/stock_baselines/stock_sampling_trot_disturb_4s_x48_recheck`",
            "- `outputs/archive/raw_runs/trot_20260409/quality_sweeps/trot_turn_10s_symroll_test`",
            "- `outputs/archive/raw_runs/trot_20260409/quality_sweeps/trot_disturb_4s_symroll_test`",
            "- `outputs/curated_runs/current/crawl_current_default_20s`",
            "- `outputs/archive/raw_runs/crawl_20260409/crawl_weakleg_share_ref040_test`",
            "",
            "Report document:",
            "- `weekly_progress_report_ko_20260410.docx`",
            "",
        ]
    )
    out = OUT_DIR / "README.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _loads()
    generated = [
        make_trot_improvement(data),
        make_failure_ablations(data),
        make_crawl_failure_story(data),
        make_stock_vs_custom(data),
    ]
    readme = write_readme(generated)
    for path in generated + [readme]:
        print(path)


if __name__ == "__main__":
    main()
