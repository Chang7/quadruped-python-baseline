
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import mujoco


def make_free_camera(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    lookat: Iterable[float] | None = None,
    distance: float = 1.8,
    azimuth: float = 135.0,
    elevation: float = -20.0,
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    try:
        mujoco.mjv_defaultFreeCamera(m, cam)
    except Exception:
        pass

    if lookat is None:
        trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if trunk_id >= 0:
            look = d.xpos[trunk_id].copy()
        else:
            look = np.array([0.0, 0.0, 0.2], dtype=float)
    else:
        look = np.asarray(list(lookat), dtype=float)

    cam.lookat[:] = look[:3]
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def should_capture(sim_time: float, frame_count: int, fps: int, start_time: float, end_time: float | None) -> bool:
    if sim_time < start_time:
        return False
    if end_time is not None and sim_time > end_time:
        return False
    expected = int(np.floor((sim_time - start_time) * fps)) + 1
    return frame_count < expected


def create_renderer_with_fallback(m: mujoco.MjModel, width: int, height: int):
    req_w = int(width)
    req_h = int(height)
    try:
        return mujoco.Renderer(m, width=req_w, height=req_h), req_w, req_h, None
    except Exception as exc:
        msg = str(exc)
        max_w = 640
        max_h = 480
        import re
        fw = re.search(r"framebuffer width\s+(\d+)", msg)
        fh = re.search(r"framebuffer height\s+(\d+)", msg)
        if fw:
            max_w = int(fw.group(1))
        if fh:
            max_h = int(fh.group(1))
        use_w = min(req_w, max_w)
        use_h = min(req_h, max_h)
        renderer = mujoco.Renderer(m, width=use_w, height=use_h)
        return renderer, use_w, use_h, msg


def capture_rgb_frame(renderer: mujoco.Renderer, d: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(d, camera=camera)
    rgb = renderer.render()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb.copy()


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int) -> str:
    if not frames:
        raise ValueError("No frames were captured; GIF cannot be written.")
    import imageio.v2 as imageio
    path = str(path)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    return path


def save_mp4(frames: list[np.ndarray], path: str | Path, fps: int) -> str:
    if not frames:
        raise ValueError("No frames were captured; MP4 cannot be written.")
    import imageio.v2 as imageio
    path = str(path)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path
