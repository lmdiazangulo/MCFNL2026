# %% [markdown]
# # 1-D FDTD Field Movie
#
# Run this file in VSCode with the Jupyter extension (or any IPython-compatible
# interactive environment) by executing each cell with **Shift+Enter**.
#
# The initial condition is a narrow Gaussian pulse centred in the domain.
# Because CFL = 1 the pulse splits into two exact half-amplitude copies that
# travel in opposite directions.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from fdtd1d import FDTD1D

# %% [markdown]
# ## Simulation parameters

# %% Parameters
# Spatial grid
x = np.linspace(-1.0, 1.0, 401)

# Gaussian initial condition
x0    = 0.0   # centre
sigma = 0.08  # width
e0    = np.exp(-0.5 * ((x - x0) / sigma) ** 2)

# Number of animation frames and simulation time step between frames
n_frames     = 120
dt_per_frame = 0.01

# %% [markdown]
# ## Pre-compute all frames

# %% Run simulation
fdtd = FDTD1D(x)
fdtd.load_initial_field(e0)

frames = []  # E-field snapshots
times  = []  # corresponding simulation times

# Store frame 0 (initial condition)
frames.append(fdtd.get_e())
times.append(fdtd.t)

for _ in range(n_frames - 1):
    fdtd.run_until(fdtd.t + dt_per_frame)
    frames.append(fdtd.get_e())
    times.append(fdtd.t)

print(f"Captured {len(frames)} frames  "
      f"(t = {times[0]:.3f} … {times[-1]:.3f})")

# %% [markdown]
# ## Build and display the animation

# %% Animate
fig, ax = plt.subplots(figsize=(8, 4))

ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.6, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("E(x, t)")
ax.set_title("1-D FDTD – Electric field evolution")
ax.grid(True, alpha=0.3)

(line,)  = ax.plot([], [], lw=2, color="royalblue")
time_txt = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=10)


def init():
    line.set_data([], [])
    time_txt.set_text("")
    return line, time_txt


def update(frame_idx):
    line.set_data(x, frames[frame_idx])
    time_txt.set_text(f"t = {times[frame_idx]:.3f}")
    return line, time_txt


anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    init_func=init,
    interval=40,   # ms between frames → ~25 fps
    blit=True,
)

plt.close(fig)         # prevent a static duplicate figure
HTML(anim.to_jshtml()) # display the animation inline
