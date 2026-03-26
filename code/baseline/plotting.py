from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from baseline.config import MPCConfig, LEG_NAMES


def _ideal_normal_force(contact: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """Compute the ideal mg/n stance distribution used for qualitative comparison."""
    n_contact = contact.sum(axis=1, keepdims=True)
    safe_n = np.where(n_contact > 0, n_contact, 1)
    ideal = np.where(contact, cfg.mass * cfg.g / safe_n, 0.0)
    return ideal


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _reconstruct_reference_xy(t: np.ndarray, x: np.ndarray, x_ref0: np.ndarray) -> np.ndarray:
    """
    Reconstruct a meaningful 2D reference path by integrating the logged reference velocity.

    Why this is needed:
    x_ref0[:, 0:2] is not a useful XY path in the current baseline because x_ref[0] is reset to
    the current measured state at every MPC update. Integrating the reference velocity yields a
    path that can actually be compared against the measured COM motion.
    """
    xy_ref = np.zeros((t.size, 2), dtype=float)
    xy_ref[0] = x[0, 0:2]

    for k in range(1, t.size):
        dt = max(float(t[k] - t[k - 1]), 0.0)
        xy_ref[k] = xy_ref[k - 1] + dt * x_ref0[k - 1, 3:5]

    return xy_ref


def _plot_contact_compare(t: np.ndarray, contact_sched: np.ndarray, contact_actual: np.ndarray, outdir: Path) -> list[str]:
    saved: list[str] = []

    fig, axes = plt.subplots(4, 1, figsize=(6.2, 6.0), sharex=True)
    for i, ax in enumerate(axes):
        ax.step(t, contact_sched[:, i].astype(float), where="post", linewidth=1.8, label=f"{LEG_NAMES[i]} scheduled")
        ax.step(t, contact_actual[:, i].astype(float), where="post", linewidth=1.4, linestyle="--", label=f"{LEG_NAMES[i]} actual")
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0.0, 1.0])
        ax.set_ylabel("contact")
        ax.set_title(LEG_NAMES[i], loc="left", fontsize=10, pad=2)
        ax.grid(alpha=0.30, linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(h_pad=0.7)
    saved.append(_save(fig, outdir / "fig_contact_schedule_vs_actual.png"))

    mismatch = (contact_sched != contact_actual).astype(float)
    fig2, ax2 = plt.subplots(figsize=(5.8, 2.8))
    width = 0.18
    xs = np.arange(4)
    ax2.bar(xs, contact_sched.mean(axis=0), width=width, label="scheduled stance ratio")
    ax2.bar(xs + width, contact_actual.mean(axis=0), width=width, label="actual contact ratio")
    ax2.bar(xs + 2 * width, mismatch.mean(axis=0), width=width, label="mismatch ratio")
    ax2.set_xticks(xs + width)
    ax2.set_xticklabels(LEG_NAMES)
    ax2.set_ylabel("ratio")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.30, linewidth=0.6, axis="y")
    ax2.legend(frameon=True, fontsize=8)
    fig2.tight_layout()
    saved.append(_save(fig2, outdir / "fig_contact_mismatch_summary.png"))

    return saved


def plot_logs(log: dict, cfg: MPCConfig, output_dir: str = "outputs") -> list[str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(log["t"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    u = np.asarray(log["u"], dtype=float)
    contact = np.asarray(log["contact"], dtype=bool)
    x_ref0 = np.asarray(log.get("x_ref0", []), dtype=float)
    contact_actual = np.asarray(log.get("contact_actual", []), dtype=bool)

    if t.size == 0:
        return []

    saved: list[str] = []

    # Figure 1: forward velocity tracking
    fig1, ax1 = plt.subplots(figsize=(5.6, 3.0))
    ax1.plot(t, x[:, 3], linewidth=2.0, label=r"$v_x$")
    if x_ref0.size:
        ax1.plot(t, x_ref0[:, 3], "--", linewidth=2.0, label=r"$v_{x,ref}$")
    else:
        ax1.plot(t, np.full_like(t, cfg.desired_speed), "--", linewidth=2.0, label=r"$v_{x,ref}$")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel(r"$v_x$ [m/s]")
    ax1.legend(frameon=True)
    ax1.grid(alpha=0.35, linewidth=0.6)
    fig1.tight_layout()
    saved.append(_save(fig1, outdir / "fig_velocity_tracking.png"))

    # Figure 2: yaw / heading response
    fig2, ax2 = plt.subplots(figsize=(5.6, 3.0))
    ax2.plot(t, x[:, 8], linewidth=2.0, label=r"$\psi$")
    if x_ref0.size:
        ax2.plot(t, x_ref0[:, 8], "--", linewidth=2.0, label=r"$\psi_{ref}$")
    else:
        ax2.plot(t, np.full_like(t, cfg.desired_yaw), "--", linewidth=2.0, label=r"$\psi_{ref}$")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("yaw [rad]")
    ax2.legend(frameon=True)
    ax2.grid(alpha=0.35, linewidth=0.6)
    fig2.tight_layout()
    saved.append(_save(fig2, outdir / "fig_yaw_tracking.png"))

    # Figure 3: per-leg normal forces with ideal mg/n overlay
    fz = u[:, 2::3]
    ideal_fz = _ideal_normal_force(contact, cfg)
    fig3, axes = plt.subplots(4, 1, figsize=(6.0, 6.2), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, fz[:, i], linewidth=1.8, label=f"{LEG_NAMES[i]} MPC")
        ax.plot(t, ideal_fz[:, i], "--", linewidth=1.4, label=f"{LEG_NAMES[i]} ideal $mg/n$")
        ax.set_ylabel(r"$f_z$ [N]")
        ax.set_title(LEG_NAMES[i], loc="left", fontsize=10, pad=2)
        ax.grid(alpha=0.30, linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
    axes[-1].set_xlabel("time [s]")
    fig3.tight_layout(h_pad=0.7)
    saved.append(_save(fig3, outdir / "fig_leg_fz_subplots.png"))

    # Figure 4: XY path with reconstructed reference path
    fig4, ax4 = plt.subplots(figsize=(4.3, 3.8))
    ax4.plot(x[:, 0], x[:, 1], linewidth=2.0, label="actual path")
    if x_ref0.size:
        xy_ref = _reconstruct_reference_xy(t, x, x_ref0)
        ax4.plot(xy_ref[:, 0], xy_ref[:, 1], "--", linewidth=2.0, label="reference path")
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("y [m]")
    ax4.set_aspect("equal", adjustable="box")
    ax4.legend(frameon=True)
    ax4.grid(alpha=0.35, linewidth=0.6)
    fig4.tight_layout()
    saved.append(_save(fig4, outdir / "fig_xy_path.png"))

    if contact_actual.size and contact_actual.shape == contact.shape:
        saved.extend(_plot_contact_compare(t, contact, contact_actual, outdir))

    return saved
