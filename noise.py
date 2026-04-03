"""Minimal fallback noise module for flat-scene simulation on Windows.

The upstream gym_quadruped package imports ``noise`` unconditionally from its
terrain helper even when we only use the flat scene.  On this machine the
compiled ``noise`` package is not available, so we provide the tiny subset of
the API that gym_quadruped touches.
"""


def pnoise2(*args, **kwargs):
    """Return zero-valued Perlin noise.

    This keeps terrain generation deterministic and flat when a rough-terrain
    path is reached accidentally, while primarily serving to satisfy import-time
    availability on Windows.
    """

    return 0.0

