# %% [markdown]
# # Conductive Panel -- FDTD Visualization
#
# Gaussian pulse propagating through a conductive panel (1D FDTD).
# Run each cell with **Shift+Enter** in VSCode/Jupyter.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML, display
import sys, os

_this_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
_root = os.path.join(_this_dir, '..')
if not os.path.isfile(os.path.join(_root, 'fdtd1d.py')):
    _root = _this_dir  # already in root
sys.path.insert(0, os.path.abspath(_root))
from fdtd1d import FDTD1D, gaussian, C
from panel_utils import (
    stack_transfer_matrix,
    RT_from_transfer_matrix, reflection_transmission,
    run_panel_experiment
)

# %% [markdown]
# ## 1. Field Animation -- Pulse hitting a conductive panel

# %% Parameters
N = 2001
L = 4.0
panel_center = 2.0
panel_thickness = 0.3
eps_r = 4.0
sigma_val = 0.5
pulse_x0 = 1.2
pulse_sigma = 0.08

x = np.linspace(0, L, N)
xH = (x[1:] + x[:-1]) / 2.0
panel_left = panel_center - panel_thickness / 2
panel_right = panel_center + panel_thickness / 2

# %% Run simulation and capture frames
t0_pulse = 4.0 * pulse_sigma
pert_fn = lambda t: gaussian(t, t0_pulse, pulse_sigma)

fdtd = FDTD1D(x, boundaries=('mur', 'mur'),
              x_o=pulse_x0, pert=pert_fn, pert_dir=+1)
fdtd.set_panel(panel_center, panel_thickness, eps_r, sigma_val)

n_frames = 250
dt_per_frame = 0.015

frames_e, frames_h, times = [fdtd.get_e()], [fdtd.get_h()], [fdtd.t]
for _ in range(n_frames - 1):
    fdtd.run_until(fdtd.t + dt_per_frame)
    frames_e.append(fdtd.get_e())
    frames_h.append(fdtd.get_h())
    times.append(fdtd.t)

print(f"Captured {len(frames_e)} frames (t = {times[0]:.3f} ... {times[-1]:.3f})")

# %% Animate
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, L); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x"); ax.set_ylabel("Field amplitude")
ax.set_title(f"FDTD -- Pulse through conductive panel "
             f"($\\varepsilon_r$={eps_r}, $\\sigma$={sigma_val}, d={panel_thickness})")
ax.grid(True, alpha=0.3)

ax.add_patch(Rectangle((panel_left, -1.1), panel_thickness, 2.2,
                        color='orange', alpha=0.2, label='Panel'))
ax.axvline(panel_left, color='orange', ls='--', lw=1)
ax.axvline(panel_right, color='orange', ls='--', lw=1)

(line_e,) = ax.plot([], [], lw=2, color='royalblue', label='E(x,t)')
(line_h,) = ax.plot([], [], lw=1.5, color='darkorange', alpha=0.7, label='H(x,t)')
ax.legend(loc='upper right')
time_txt = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=10)

def init():
    line_e.set_data([], []); line_h.set_data([], []); time_txt.set_text("")
    return line_e, line_h, time_txt

def update(i):
    line_e.set_data(x, frames_e[i]); line_h.set_data(xH, frames_h[i])
    time_txt.set_text(f"t = {times[i]:.3f}")
    return line_e, line_h, time_txt

anim = FuncAnimation(fig, update, frames=len(frames_e), init_func=init,
                     interval=40, blit=True)
plt.close(fig)
display(HTML(anim.to_jshtml()))

# %% [markdown]
# ## 2. R(f) and T(f) -- FDTD vs Analytical (single panel)

# %% Compute R,T using run_panel_experiment
print("Running FDTD panel experiment...")
res = run_panel_experiment(
    N=4001, L=4.0, panel_d=panel_thickness,
    eps_r=eps_r, sigma=sigma_val, pulse_sigma=0.06,
)
freq_fdtd = res['freq']
R_fdtd, T_fdtd = res['R'], res['T']

f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
R_anal, T_anal = reflection_transmission(f_anal, panel_thickness, eps_r, sigma_val)

f_bw = 1.0 / (2.0 * np.pi * 0.06)
f_max = min(3.0 * f_bw, freq_fdtd.max())

# %% Plot R,T comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Single conductive panel: $\\varepsilon_r$={eps_r}, '
             f'$\\sigma$={sigma_val}, d={panel_thickness}', fontsize=13)

mask = (freq_fdtd > 0.05) & (freq_fdtd < f_max)
mask_a = (f_anal > 0.05) & (f_anal < f_max)

axes[0].plot(freq_fdtd[mask], np.abs(R_fdtd[mask]), 'b-', alpha=0.6, lw=1, label='FDTD')
axes[0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analytical (TMM)')
axes[0].set_xlabel('Frequency'); axes[0].set_ylabel('|R|')
axes[0].set_title('|R(f)|'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_fdtd[mask], np.abs(T_fdtd[mask]), 'b-', alpha=0.6, lw=1, label='FDTD')
axes[1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analytical (TMM)')
axes[1].set_xlabel('Frequency'); axes[1].set_ylabel('|T|')
axes[1].set_title('|T(f)|'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(freq_fdtd[mask], np.abs(R_fdtd[mask])**2 + np.abs(T_fdtd[mask])**2,
             'b-', alpha=0.6, lw=1, label='FDTD')
axes[2].plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
             'r--', lw=2, label='Analytical (TMM)')
axes[2].axhline(1.0, color='gray', ls=':', alpha=0.5)
axes[2].set_xlabel('Frequency'); axes[2].set_ylabel('$|R|^2 + |T|^2$')
axes[2].set_title('Energy conservation')
axes[2].legend(); axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 1.15)

plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Multi-layer Panel (Bonus)

# %% Multi-layer: analytical + FDTD
layers_ml = [
    {'d': 0.10, 'eps_r': 10.0, 'sigma': 0.0},
    {'d': 0.08, 'eps_r': 1.0, 'sigma': 10.0},
    {'d': 0.15, 'eps_r': 2.0, 'sigma': 0.2},
]

f_ml = np.linspace(0.01, 10.0, 1000)
Phi_ml = stack_transfer_matrix(f_ml, layers_ml)
R_ml, T_ml = RT_from_transfer_matrix(Phi_ml)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Multi-layer panel: 3 layers', fontsize=12)
axes[0].plot(f_ml, np.abs(R_ml), 'b-', lw=1.5); axes[0].set_ylabel('|R|')
axes[0].set_xlabel('Frequency'); axes[0].set_title('|R(f)|'); axes[0].grid(True, alpha=0.3)
axes[1].plot(f_ml, np.abs(T_ml), 'r-', lw=1.5); axes[1].set_ylabel('|T|')
axes[1].set_xlabel('Frequency'); axes[1].set_title('|T(f)|'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3b. Multi-layer Panel Animation

# %% Multi-layer animation
fdtd_ml = FDTD1D(x, boundaries=('mur', 'mur'), x_o=pulse_x0, pert=pert_fn, pert_dir=+1)
fdtd_ml.set_multilayer(panel_center, layers_ml)
fdtd_ml.load_initial_field(np.zeros_like(x))
fdtd_ml.h = np.zeros_like(xH)

n_frames_ml = 220

dt_per_frame_ml = 0.015
frames_e_ml = [fdtd_ml.get_e()]
frames_h_ml = [fdtd_ml.get_h()]
times_ml = [fdtd_ml.t]
for _ in range(n_frames_ml - 1):
    fdtd_ml.run_until(fdtd_ml.t + dt_per_frame_ml)
    frames_e_ml.append(fdtd_ml.get_e())
    frames_h_ml.append(fdtd_ml.get_h())
    times_ml.append(fdtd_ml.t)

print(f"Multilayer animation captured {len(frames_e_ml)} frames "
      f"(t = {times_ml[0]:.3f} ... {times_ml[-1]:.3f})")

fig_ml, ax_ml = plt.subplots(figsize=(10, 5))
ax_ml.set_xlim(0, L)
ax_ml.set_ylim(-1.1, 1.1)
ax_ml.set_xlabel("x")
ax_ml.set_ylabel("Field amplitude")
ax_ml.set_title("FDTD -- Pulse through multilayer panel")
ax_ml.grid(True, alpha=0.3)

panel_width = sum(layer['d'] for layer in layers_ml)
panel_left_ml = panel_center - panel_width / 2
edge = panel_left_ml
for layer in layers_ml:
    ax_ml.add_patch(Rectangle((edge, -1.1), layer['d'], 2.2,
                              color='orange', alpha=0.25))
    edge += layer['d']
ax_ml.axvline(panel_left_ml, color='orange', ls='--', lw=1)
ax_ml.axvline(panel_left_ml + panel_width, color='orange', ls='--', lw=1)

(line_e_ml,) = ax_ml.plot([], [], lw=2, color='royalblue', label='E(x,t)')
(line_h_ml,) = ax_ml.plot([], [], lw=1.5, color='darkorange', alpha=0.7, label='H(x,t)')
ax_ml.legend(loc='upper right')

time_txt_ml = ax_ml.text(0.02, 0.93, "", transform=ax_ml.transAxes, fontsize=10)

def init_ml():
    line_e_ml.set_data([], [])
    line_h_ml.set_data([], [])
    time_txt_ml.set_text("")
    return line_e_ml, line_h_ml, time_txt_ml


def update_ml(i):
    line_e_ml.set_data(x, frames_e_ml[i])
    line_h_ml.set_data(xH, frames_h_ml[i])
    time_txt_ml.set_text(f"t = {times_ml[i]:.3f}")
    return line_e_ml, line_h_ml, time_txt_ml

anim_ml = FuncAnimation(fig_ml, update_ml, frames=len(frames_e_ml),
                        init_func=init_ml, interval=40, blit=True)
plt.close(fig_ml)
display(HTML(anim_ml.to_jshtml()))

# %% [markdown]
# ## 4. Parameter Study -- Effect of conductivity

# %% Parameter sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Effect of conductivity ($\\varepsilon_r$={eps_r}, d={panel_thickness})', fontsize=13)

f_sweep = np.linspace(0.01, 8.0, 1000)
for sig in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
    R_s, T_s = reflection_transmission(f_sweep, panel_thickness, eps_r, sig)
    axes[0].plot(f_sweep, np.abs(R_s), label=f'$\\sigma$={sig}')
    axes[1].plot(f_sweep, np.abs(T_s), label=f'$\\sigma$={sig}')

axes[0].set_xlabel('Frequency'); axes[0].set_ylabel('|R|')
axes[0].set_title('|R(f)|'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Frequency'); axes[1].set_ylabel('|T|')
axes[1].set_title('|T(f)|'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
# %%
