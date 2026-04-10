from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "outputs" / "archive" / "raw_runs" / "20260410_trot_stock_vs_custom_same_scenarios"
OUT_DIR = ROOT / "outputs" / "report_progress_explainer" / "trot_stock_vs_custom_same_scenarios_20260410"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEG_NAMES = ["FL", "FR", "RL", "RR"]
STOCK_COLOR = "tab:blue"
CUSTOM_COLOR = "tab:red"
ROLL_COLOR = "tab:orange"
PITCH_COLOR = "tab:green"

SCENARIOS = [
    {
        "slug": "straight",
        "title": "Trot + Straight | stock vs custom linear_osqp",
        "subtitle": "aliengo | flat | vx=0.12 | yaw=0.0 | 20 s",
        "stock": RUN_ROOT / "stock_straight_20s" / "episode_000",
        "custom": RUN_ROOT / "custom_straight_20s" / "episode_000",
        "footnote": "Matched local benchmark. Custom run uses the current dynamic-trot default path.",
    },
    {
        "slug": "turn",
        "title": "Trot + Turn | stock vs custom linear_osqp",
        "subtitle": "aliengo | flat | vx=0.10 | yaw=0.3 | 10 s",
        "stock": RUN_ROOT / "stock_turn_10s" / "episode_000",
        "custom": RUN_ROOT / "custom_turn_10s" / "episode_000",
        "footnote": "Matched local benchmark. The main gap that remains is forward velocity tracking, not posture.",
    },
    {
        "slug": "disturbance",
        "title": "Trot + Disturbance | stock vs custom linear_osqp",
        "subtitle": "aliengo | flat | vx=0.12 | pulses at 0.5 s / 2.3 s | 4 s",
        "stock": RUN_ROOT / "stock_disturb_4s" / "episode_000",
        "custom": RUN_ROOT / "custom_disturb_4s" / "episode_000",
        "footnote": "Matched local benchmark. Disturbance pulses: x:0.5:0.25:4.0 and x:2.3:0.25:8.0.",
    },
]


def _load_episode(ep_dir: Path) -> dict:
    data = np.load(ep_dir / "run_log.npz", allow_pickle=True)
    summary = json.loads((ep_dir / "summary.json").read_text(encoding="utf-8"))
    out = {key: data[key] for key in data.files}
    out["summary"] = summary
    return out


def _contact_to_band(values: np.ndarray, offset: float, height: float = 0.8) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    return offset + height * vals


def _termination_text(summary: dict) -> str:
    if not summary.get("terminated_any", False):
        return "none"
    invalid = summary.get("meta", {}).get("invalid_contact_keys", [])
    if invalid:
        return invalid[0].replace("world:0_", "").replace(":", " ")
    return "terminated"


def _panel(
    ax_height,
    ax_angle,
    ax_contact,
    episode: dict,
    color: str,
    label: str,
    show_xlabel: bool,
) -> None:
    t = np.asarray(episode["time"], dtype=float)
    z = np.asarray(episode["base_pos"], dtype=float)[:, 2]
    euler = np.asarray(episode["base_ori_euler_xyz"], dtype=float)
    actual = np.asarray(episode["foot_contact"], dtype=float)
    roll = np.rad2deg(euler[:, 0])
    pitch = np.rad2deg(euler[:, 1])

    ax_height.plot(t, z, color=color, linewidth=2.0, label="base z")
    ax_height.axhline(
        float(episode["summary"]["ref_base_height"]),
        color=color,
        linestyle="--",
        alpha=0.45,
        label="ref z",
    )
    ax_height.set_title(label, fontsize=12)
    ax_height.grid(alpha=0.3)
    ax_height.legend(loc="upper left", fontsize=9)

    ax_angle.plot(t, roll, color=ROLL_COLOR, linewidth=1.8, label="roll")
    ax_angle.plot(t, pitch, color=PITCH_COLOR, linewidth=1.8, label="pitch")
    ax_angle.grid(alpha=0.3)
    ax_angle.legend(loc="upper left", fontsize=9)

    offsets = [0.0, 1.1, 2.2, 3.3]
    for idx, leg_name in enumerate(LEG_NAMES):
        ax_contact.plot(
            t,
            _contact_to_band(actual[:, idx], offsets[idx]),
            linewidth=1.8,
            label=f"{leg_name} actual",
        )
    ax_contact.set_yticks([0.4, 1.5, 2.6, 3.7])
    ax_contact.set_yticklabels(LEG_NAMES)
    ax_contact.grid(alpha=0.3)
    if show_xlabel:
        ax_contact.set_xlabel("time [s]")


def _footer_text(summary: dict) -> str:
    return (
        f"duration={summary['duration_s']:.3f}s | mean_z={summary['mean_base_z']:.3f} | "
        f"mean|roll|={summary['mean_abs_roll']:.3f} | mean|pitch|={summary['mean_abs_pitch']:.3f} | "
        f"mean_vx={summary['mean_vx']:.3f} | termination={_termination_text(summary)}"
    )


def make_scenario_panel(config: dict) -> Path:
    stock = _load_episode(config["stock"])
    custom = _load_episode(config["custom"])

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15.2, 8.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.15], "hspace": 0.35, "wspace": 0.2},
    )

    _panel(
        axes[0, 0],
        axes[1, 0],
        axes[2, 0],
        stock,
        STOCK_COLOR,
        "Stock sampling-based MPC",
        show_xlabel=True,
    )
    _panel(
        axes[0, 1],
        axes[1, 1],
        axes[2, 1],
        custom,
        CUSTOM_COLOR,
        "Custom linear_osqp MPC",
        show_xlabel=True,
    )

    axes[0, 0].set_ylabel("height [m]")
    axes[1, 0].set_ylabel("angle [deg]")
    axes[2, 0].set_title("Actual foot-contact timeline")
    axes[2, 1].set_title("Actual foot-contact timeline")

    fig.suptitle(f"{config['title']}\n{config['subtitle']}", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.02, 0.06, _footer_text(stock["summary"]), fontsize=9.5, family="monospace", color="#163e70")
    fig.text(0.52, 0.06, _footer_text(custom["summary"]), fontsize=9.5, family="monospace", color="#7a1d1d")
    fig.text(0.02, 0.02, config["footnote"], fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, hspace=0.35, wspace=0.2)

    out_path = OUT_DIR / f"trot_{config['slug']}_stock_vs_custom.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_summary_table() -> Path:
    rows = []
    for config in SCENARIOS:
        stock = json.loads((config["stock"] / "summary.json").read_text(encoding="utf-8"))
        custom = json.loads((config["custom"] / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            [
                config["slug"],
                f"{stock['duration_s']:.3f}",
                f"{custom['duration_s']:.3f}",
                f"{stock['mean_abs_roll']:.3f}",
                f"{custom['mean_abs_roll']:.3f}",
                f"{stock['mean_abs_pitch']:.3f}",
                f"{custom['mean_abs_pitch']:.3f}",
                f"{stock['mean_vx']:.3f}",
                f"{custom['mean_vx']:.3f}",
            ]
        )

    columns = [
        "scenario",
        "stock dur [s]",
        "custom dur [s]",
        "stock |roll|",
        "custom |roll|",
        "stock |pitch|",
        "custom |pitch|",
        "stock vx",
        "custom vx",
    ]

    fig, ax = plt.subplots(figsize=(12.8, 2.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
        elif row % 2 == 1:
            cell.set_facecolor("#f8fbff")

    fig.suptitle(
        "Matched local trot benchmarks | stock sampling vs custom linear_osqp",
        fontsize=12,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.01,
        0.02,
        "Scenarios align with the paper's straight / turning / disturbance trot families, "
        "but these are local matched benchmarks rather than exact paper-setting reproductions.",
        fontsize=8.8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))

    out_path = OUT_DIR / "trot_stock_vs_custom_summary_table.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_readme(outputs: list[Path]) -> Path:
    lines = [
        "# Trot Stock vs Custom Same-Scenario Panels",
        "",
        "Matched local comparison panels for the three trot benchmark families used in this repo.",
        "",
        "Generated files:",
    ]
    for output in outputs:
        lines.append(f"- `{output.name}`")
    lines.extend(
        [
            "",
            "Source runs:",
            "- `outputs/archive/raw_runs/20260410_trot_stock_vs_custom_same_scenarios/stock_straight_20s`",
            "- `outputs/archive/raw_runs/20260410_trot_stock_vs_custom_same_scenarios/custom_straight_20s`",
            "- `outputs/archive/raw_runs/20260410_trot_stock_vs_custom_same_scenarios/stock_turn_10s`",
            "- `outputs/archive/raw_runs/20260410_trot_stock_vs_custom_same_scenarios/custom_turn_10s`",
            "- `outputs/archive/raw_runs/20260410_trot_stock_vs_custom_same_scenarios/stock_disturb_4s`",
            "- `outputs/archive/raw_runs/20260410_trot_stock_vs_custom_same_scenarios/custom_disturb_4s`",
            "",
            "Notes:",
            "- These are matched local scenario comparisons, not a claim of exact paper-setting reproduction.",
            "- In the current local results, custom posture quality is better, while forward velocity tracking is still weaker in the turn case.",
        ]
    )
    out_path = OUT_DIR / "README.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    outputs = [make_summary_table()]
    for config in SCENARIOS:
        outputs.append(make_scenario_panel(config))
    outputs.append(write_readme(outputs))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
