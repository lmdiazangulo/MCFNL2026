"""Create an animation of the 1D wave propagation solved by FDTD1D.

Run this script to generate an animation file (GIF by default) showing how the
field evolves in time.

Usage:
    python animate_fdtd1d.py

The script will create `fdtd1d.gif` in the current folder and optionally show
an interactive matplotlib window.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from fdtd1d import FDTD1D


from pathlib import Path


def make_animation(
    x,
    initial_field,
    t_final=2.0,
    n_frames=200,
    outfile="fdtd1d.gif",
    show=False,
):
    """Create and optionally display an animation of the wave evolution."""

    # Default to placing output next to this script to avoid permission issues.
    outfile = str((Path(__file__).resolve().parent / outfile).resolve())

    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_field)
    # Initialize H to zero so it can be animated as well
    fdtd.load_initial_field(np.zeros_like(x[:-1]))

    fig, (ax_e, ax_h) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    line_e, = ax_e.plot(x, initial_field, lw=2)
    ax_e.set_ylabel("E(x, t)")
    ax_e.grid(alpha=0.3)

    line_h, = ax_h.plot(x[:-1], fdtd.get_h(), lw=2)
    ax_h.set_ylabel("H(x, t)")
    ax_h.set_xlabel("x")
    ax_h.grid(alpha=0.3)

    title = fig.suptitle("")

    # Set reasonable limits based on initial fields
    e_lim = np.max(np.abs(initial_field))
    h_lim = np.max(np.abs(fdtd.get_h()))
    ax_e.set_ylim(-1.1 * e_lim, 1.1 * e_lim)
    ax_h.set_ylim(-1.1 * (h_lim if h_lim > 0 else 1.0), 1.1 * (h_lim if h_lim > 0 else 1.0))

    times = np.linspace(0.0, t_final, n_frames)

    def update(frame_index):
        t = times[frame_index]
        fdtd.run_until(t)
        e = fdtd.get_e()
        h = fdtd.get_h()
        line_e.set_data(x, e)
        line_h.set_data(x[:-1], h)
        title.set_text(f"t = {t:.3f}")
        return line_e, line_h, title

    anim = FuncAnimation(fig, update, frames=len(times), blit=True)

    # Save as GIF; this requires Pillow (usually installed with matplotlib).
    writer = PillowWriter(fps=20)
    anim.save(outfile, writer=writer)

    if show:
        plt.show()

    return outfile


if __name__ == "__main__":
    # Example usage: Gaussian pulse
    x = np.linspace(-1.0, 1.0, 201)
    x0 = 0.0
    sigma = 0.05
    # Use a proper Gaussian peak (decays away from center)
    initial_e = np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    out = make_animation(
        x,
        initial_e,
        t_final=2.0,
        n_frames=200,
        outfile="fdtd1d.gif",
        show=True,
    )
    print(f"Saved animation: {out}")
