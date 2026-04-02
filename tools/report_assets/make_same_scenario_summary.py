from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "same_scenario_compare"
OUT = ROOT / "outputs" / "report_progress_explainer"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "sampling_crawl": BASE / "sampling_crawl" / "episode_000" / "summary.json",
    "linear_crawl": BASE / "linear_crawl" / "episode_000" / "summary.json",
    "sampling_trot": BASE / "sampling_trot" / "episode_000" / "summary.json",
    "linear_trot": BASE / "linear_trot" / "episode_000" / "summary.json",
}


def load() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, path in RUNS.items():
        out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def cell(summary: dict) -> list[str]:
    meta = summary["meta"]
    invalid = ", ".join(meta.get("invalid_contact_keys", [])) or "none"
    fail = "no termination" if not summary["terminated_any"] else invalid.replace("world:0_", "").replace(":", " ")
    return [
        meta["controller_type"],
        meta["gait"],
        f"{summary['duration_s']:.3f}s",
        f"{summary['mean_vx']:.3f}",
        f"{summary['mean_base_z']:.3f}",
        f"{summary['mean_abs_roll']:.3f}",
        f"{summary['front_actual_swing_realization_mean']:.3f}",
        fail,
    ]


def make_table() -> Path:
    data = load()
    columns = [
        "controller",
        "gait",
        "duration",
        "mean_vx",
        "mean_z",
        "mean_roll",
        "front_swing",
        "termination",
    ]
    rows = [
        cell(data["sampling_crawl"]),
        cell(data["linear_crawl"]),
        cell(data["sampling_trot"]),
        cell(data["linear_trot"]),
    ]
    row_labels = [
        "same scenario A",
        "same scenario A",
        "same scenario B",
        "same scenario B",
    ]

    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=columns,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.7)

    for (r, c), cell_obj in tbl.get_celld().items():
        if r == 0:
            cell_obj.set_text_props(weight="bold")
            cell_obj.set_facecolor("#e8eef7")
        elif c == -1:
            cell_obj.set_text_props(weight="bold")
            if r in (1, 2):
                cell_obj.set_facecolor("#f7f7f7")
            else:
                cell_obj.set_facecolor("#eef7ee")
        else:
            if r in (1, 3):
                cell_obj.set_facecolor("#f4f8ff")
            else:
                cell_obj.set_facecolor("#fff6f1")

    fig.suptitle(
        "Same-scenario comparison on stock MuJoCo stack\n"
        "Scenario A: flat + aliengo + crawl + forward speed 0.12 | "
        "Scenario B: flat + aliengo + trot + forward speed 0.12",
        fontsize=12,
        fontweight="bold",
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = OUT / "same_scenario_compare_table.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(make_table())
