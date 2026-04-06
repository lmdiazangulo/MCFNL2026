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
x0    = -0.2   # centre
sigma = 0.15  # width
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

x0_mur    = -0.2
sigma_mur = 0.15

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
# %%

# %% [markdown]
# ## TF/SF Condition Test
# A source is injected at x_tf_sf. You should see a wave 
# traveling only to the right (Total Field) and zero field to the left.

# %% Run simulation (TF/SF)
x_tfsf_grid = np.linspace(-1.0, 1.0, 401)

# Defino una función de perturbación (Gaussiana en el tiempo)
def pulse(t):
    return np.exp(-0.5 * ((t - 0.2) / 0.05) ** 2)

# Inicializo con x_tf_sf en el centro (0.0) y Mur en los bordes
fdtd_ts = FDTD1D(x_tfsf_grid, boundaries=('mur', 'mur'), x_tf_sf=0.0, pert=pulse)

frames_tfsf = []
times_tfsf = []

for _ in range(150):
    fdtd_ts.run_until(fdtd_ts.t + 0.01)
    frames_tfsf.append(fdtd_ts.get_e())
    times_tfsf.append(fdtd_ts.t)

print(f"TF/SF – Capturados {len(frames_tfsf)} frames")

# %% Animate (TF/SF)
fig_ts, ax_ts = plt.subplots(figsize=(8, 4))
ax_ts.set_xlim(x_tfsf_grid[0], x_tfsf_grid[-1])
ax_ts.set_ylim(-1.1, 1.1)
ax_ts.set_xlabel("x")
ax_ts.set_ylabel("E(x, t)")
ax_ts.set_title("1-D FDTD – Total-Field / Scattered-Field Injection")
ax_ts.grid(True, alpha=0.3)

# Marca visual de donde está el punto de inyección
ax_ts.axvline(0.0, color="green", ls="--", label="TF/SF Interface")

(line_ts,) = ax_ts.plot([], [], lw=2, color="seagreen", label="E-field")
time_txt_ts = ax_ts.text(0.02, 0.93, "", transform=ax_ts.transAxes)
ax_ts.legend()

def init_ts():
    line_ts.set_data([], [])
    time_txt_ts.set_text("")
    return line_ts, time_txt_ts

def update_ts(i):
    line_ts.set_data(x_tfsf_grid, frames_tfsf[i])
    time_txt_ts.set_text(f"t = {times_tfsf[i]:.3f}")
    return line_ts, time_txt_ts

anim_ts = FuncAnimation(fig_ts, update_ts, frames=len(frames_tfsf), 
                        init_func=init_ts, interval=40, blit=True)

plt.close(fig_ts)
HTML(anim_ts.to_jshtml())
# %%
# %% Verificación de sondas
x_vals = np.linspace(-1, 1, 400)
sim = FDTD1D(x_vals, boundaries=('mur', 'mur'))

# Defino una fuente gaussiana para TF/SF
def source(t):
    return np.exp(-0.5 * ((t - 0.3) / 0.05)**2)

sim.pert = source
sim.x_tf_sf = 0.0

# Añado una sonda a la izquierda (Scattered Field) y otra a la derecha (Total Field)
sim.add_probe(-0.5)
sim.add_probe(0.5)

# Ejecuto
sim.run_until(1.5)

# Grafico la sonda de x = 0.5+
plt.plot(sim.probe_data[1], label="Sonda en x=0.5")
plt.title("Evolución temporal del campo")
plt.legend()
plt.show()
# %%
