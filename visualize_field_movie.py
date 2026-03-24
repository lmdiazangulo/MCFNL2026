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

# %% [markdown]
# ## Mur Absorbing Boundary Conditions
#
# A purely **left-traveling** Gaussian pulse (initialized with H = −E)
# propagates toward the left boundary where a first-order Mur ABC is
# applied.  The pulse should be absorbed with negligible reflections.

# %% Parameters (Mur)
x_mur  = np.linspace(-1.0, 1.0, 401)
xH_mur = (x_mur[1:] + x_mur[:-1]) / 2.0

x0_mur    = 0.0
sigma_mur = 0.08

# Left-traveling wave: E = gaussian, H = -gaussian
e0_mur = np.exp(-0.5 * ((x_mur  - x0_mur) / sigma_mur) ** 2)
h0_mur = -np.exp(-0.5 * ((xH_mur - x0_mur) / sigma_mur) ** 2)

n_frames_mur     = 160
dt_per_frame_mur = 0.01

# %% Run simulation (Mur)
fdtd_mur = FDTD1D(x_mur, boundaries=('mur', 'mur'))
fdtd_mur.load_initial_field(e0_mur)
fdtd_mur.h = h0_mur.copy()

frames_e_mur = []
frames_h_mur = []
times_mur    = []

frames_e_mur.append(fdtd_mur.get_e())
frames_h_mur.append(fdtd_mur.get_h())
times_mur.append(fdtd_mur.t)

for _ in range(n_frames_mur - 1):
    fdtd_mur.run_until(fdtd_mur.t + dt_per_frame_mur)
    frames_e_mur.append(fdtd_mur.get_e())
    frames_h_mur.append(fdtd_mur.get_h())
    times_mur.append(fdtd_mur.t)

print(f"Mur ABC – Captured {len(frames_e_mur)} frames  "
      f"(t = {times_mur[0]:.3f} … {times_mur[-1]:.3f})")

# %% Animate (Mur)
fig_mur, ax_mur = plt.subplots(figsize=(8, 4))

ax_mur.set_xlim(x_mur[0], x_mur[-1])
ax_mur.set_ylim(-1.1, 1.1)
ax_mur.set_xlim(-1.1, 1.1)
ax_mur.set_xlabel("x")
ax_mur.set_ylabel("Field amplitude")
ax_mur.set_title("1-D FDTD – Mur ABC – Left-traveling wave absorption")
ax_mur.grid(True, alpha=0.3)

ax_mur.axvline(x_mur[0],  color="red", ls="--", lw=1.5, label="Mur boundary")
ax_mur.axvline(x_mur[-1], color="red", ls="--", lw=1.5)

(line_e_mur,) = ax_mur.plot([], [], lw=2, color="royalblue", label="E(x, t)")
(line_h_mur,) = ax_mur.plot([], [], lw=1.5, color="darkorange", label="H(x, t)")
ax_mur.legend(loc="upper right", fontsize=9)

time_txt_mur = ax_mur.text(0.02, 0.93, "", transform=ax_mur.transAxes, fontsize=10)


def init_mur():
    line_e_mur.set_data([], [])
    line_h_mur.set_data([], [])
    time_txt_mur.set_text("")
    return line_e_mur, line_h_mur, time_txt_mur


def update_mur(frame_idx):
    line_e_mur.set_data(x_mur, frames_e_mur[frame_idx])
    line_h_mur.set_data(xH_mur, frames_h_mur[frame_idx])
    time_txt_mur.set_text(f"t = {times_mur[frame_idx]:.3f}")
    return line_e_mur, line_h_mur, time_txt_mur


anim_mur = FuncAnimation(
    fig_mur,
    update_mur,
    frames=len(frames_e_mur),
    init_func=init_mur,
    interval=40,
    blit=True,
)

plt.close(fig_mur)
HTML(anim_mur.to_jshtml())

# %% [markdown]
# ## Conductive medium – Reflection at a dielectric interface
#
# A purely **right-traveling** Gaussian pulse in vacuum (ε=1) hits an
# interface at x=0 with a denser dielectric (ε=4).
#
# The reflected pulse is **inverted** with amplitude equal to the
# reflection coefficient Γ = (η₂ − η₁)/(η₂ + η₁) = −1/3.
# A transmitted pulse continues into the medium at reduced speed.

# %% Parameters (conductive)
N_cond = 401
x_cond = np.linspace(-2.0, 2.0, N_cond)
xH_cond = (x_cond[1:] + x_cond[:-1]) / 2.0

epsilon_cond = np.ones(N_cond)
epsilon2 = 4.0
interface_idx = N_cond // 2
epsilon_cond[interface_idx:] = epsilon2

x0_cond    = -1.0
sigma_cond = 0.08

# Right-traveling wave: H = -E
e0_cond = np.exp(-0.5 * ((x_cond  - x0_cond) / sigma_cond) ** 2)
h0_cond = -np.exp(-0.5 * ((xH_cond - x0_cond) / sigma_cond) ** 2)

n_frames_cond     = 300
dt_per_frame_cond = 0.01

# %% Run simulation (conductive)
from fdtd1d import gaussian

fdtd_cond = FDTD1D(x_cond, boundaries=('mur', 'mur'),
                    epsilon=epsilon_cond, sigma=0.0)
fdtd_cond.load_initial_field(e0_cond)
fdtd_cond.h = h0_cond.copy()

frames_e_cond = []
frames_h_cond = []
times_cond    = []

frames_e_cond.append(fdtd_cond.get_e())
frames_h_cond.append(fdtd_cond.get_h())
times_cond.append(fdtd_cond.t)

for _ in range(n_frames_cond - 1):
    fdtd_cond.run_until(fdtd_cond.t + dt_per_frame_cond)
    frames_e_cond.append(fdtd_cond.get_e())
    frames_h_cond.append(fdtd_cond.get_h())
    times_cond.append(fdtd_cond.t)

print(f"Conductive – Captured {len(frames_e_cond)} frames  "
      f"(t = {times_cond[0]:.3f} … {times_cond[-1]:.3f})")

# %% Animate (conductive)
eta1 = 1.0 / np.sqrt(1.0)
eta2 = 1.0 / np.sqrt(epsilon2)
Gamma = (eta2 - eta1) / (eta2 + eta1)

fig_cond, ax_cond = plt.subplots(figsize=(8, 4))

ax_cond.set_xlim(x_cond[0], x_cond[-1])
ax_cond.set_ylim(-1.5, 1.5)
ax_cond.set_xlabel("x")
ax_cond.set_ylabel("Field amplitude")
ax_cond.set_title(
    f"1-D FDTD – Reflection at dielectric interface "
    f"(ε₁=1, ε₂={epsilon2:.0f}, Γ={Gamma:.3f})")
ax_cond.grid(True, alpha=0.3)

# Mark the interface and shade the dielectric region
ax_cond.axvline(0.0, color="green", ls="-", lw=2, label="Interface (x=0)")
ax_cond.axvspan(0.0, x_cond[-1], alpha=0.08, color="green")
ax_cond.text(0.15, 1.0, f"ε = {epsilon2:.0f}", fontsize=10, color="green")
ax_cond.text(-1.5, 1.0, "ε = 1 (vacuum)", fontsize=10, color="gray")

# Mur boundaries
ax_cond.axvline(x_cond[0],  color="red", ls="--", lw=1, alpha=0.5)
ax_cond.axvline(x_cond[-1], color="red", ls="--", lw=1, alpha=0.5)

(line_e_cond,) = ax_cond.plot([], [], lw=2, color="royalblue", label="E(x, t)")
(line_h_cond,) = ax_cond.plot([], [], lw=1.5, color="darkorange", label="H(x, t)")
ax_cond.legend(loc="upper right", fontsize=9)

time_txt_cond = ax_cond.text(0.02, 0.93, "", transform=ax_cond.transAxes, fontsize=10)


def init_cond():
    line_e_cond.set_data([], [])
    line_h_cond.set_data([], [])
    time_txt_cond.set_text("")
    return line_e_cond, line_h_cond, time_txt_cond


def update_cond(frame_idx):
    line_e_cond.set_data(x_cond, frames_e_cond[frame_idx])
    line_h_cond.set_data(xH_cond, frames_h_cond[frame_idx])
    time_txt_cond.set_text(f"t = {times_cond[frame_idx]:.3f}")
    return line_e_cond, line_h_cond, time_txt_cond


anim_cond = FuncAnimation(
    fig_cond,
    update_cond,
    frames=len(frames_e_cond),
    init_func=init_cond,
    interval=40,
    blit=True,
)

plt.close(fig_cond)
HTML(anim_cond.to_jshtml())
# %%