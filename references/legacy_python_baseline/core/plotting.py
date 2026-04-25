from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from config import MPCConfig, LEG_NAMES, IDX_P, IDX_V, IDX_TH


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


def plot_logs(log: dict, cfg: MPCConfig, output_dir: str = "outputs") -> list[str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(log["t"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    u = np.asarray(log["u"], dtype=float)
    contact = np.asarray(log["contact"], dtype=bool)
    x_ref0 = np.asarray(log.get("x_ref0", []), dtype=float)

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

    # Figure 4: XY path for turning / sanity check
    fig4, ax4 = plt.subplots(figsize=(4.3, 3.8))
    ax4.plot(x[:, 0], x[:, 1], linewidth=2.0, label="actual path")
    if x_ref0.size:
        ax4.plot(x_ref0[:, 0], x_ref0[:, 1], "--", linewidth=2.0, label="reference path")
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("y [m]")
    ax4.set_aspect("equal", adjustable="box")
    ax4.legend(frameon=True)
    ax4.grid(alpha=0.35, linewidth=0.6)
    fig4.tight_layout()
    saved.append(_save(fig4, outdir / "fig_xy_path.png"))

    return saved
