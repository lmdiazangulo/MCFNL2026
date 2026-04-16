# %% [markdown]
# # Coeficiente de Reflexión y Transmisión de un Panel Ligeramente Conductivo
# 
# ## Método FDTD vs Matriz de Transferencia

# %%
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML, display, Markdown

from fdtd1d import FDTD1D, gaussian
from panel_utils import (
    run_panel_experiment,
    reflection_transmission,
    reflection_transmission_stack,
    stack_transfer_matrix,
    RT_from_transfer_matrix,
)

plt.rcParams['figure.dpi'] = 100
print('Entorno listo.')

# %% [markdown]
# ## 2. Visualización: el pulso atraviesa el panel

# %%
# Parámetros del panel y del pulso
N = 2001
L = 4.0
panel_center = 2.0
panel_d = 0.3
eps_r = 4.0
sigma_val = 0.5
pulse_x0 = 1.2
pulse_sigma = 0.08

x = np.linspace(0, L, N)
xH = (x[1:] + x[:-1]) / 2.0
panel_left = panel_center - panel_d / 2
panel_right = panel_center + panel_d / 2

# Fuente TF/SF: pulso gaussiano inyectado en pulse_x0 viajando a la derecha
t0 = 4.0 * pulse_sigma
pert_fn = lambda t: gaussian(t, t0, pulse_sigma)

fdtd = FDTD1D(x, boundaries=('mur', 'mur'), x_o=pulse_x0, pert=pert_fn, pert_dir=+1)
fdtd.set_panel(panel_center, panel_d, eps_r, sigma_val)

n_frames = 250
dt_per_frame = 0.015
frames_e, frames_h, times = [fdtd.get_e()], [fdtd.get_h()], [fdtd.t]
for _ in range(n_frames - 1):
    fdtd.run_until(fdtd.t + dt_per_frame)
    frames_e.append(fdtd.get_e())
    frames_h.append(fdtd.get_h())
    times.append(fdtd.t)

print(f'Capturados {len(frames_e)} frames. Generando animación...')

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.set_xlim(0, L); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('x'); ax.set_ylabel('Amplitud')
ax.set_title(f'Pulso atravesando el panel — $\\varepsilon_r$={eps_r}, $\\sigma$={sigma_val}, d={panel_d}')
ax.grid(True, alpha=0.3)
ax.add_patch(Rectangle((panel_left, -1.1), panel_d, 2.2, color='orange', alpha=0.25, label='Panel'))
ax.axvline(panel_left, color='orange', ls='--', lw=1)
ax.axvline(panel_right, color='orange', ls='--', lw=1)

(line_e,) = ax.plot([], [], lw=2, color='royalblue', label='E(x,t)')
(line_h,) = ax.plot([], [], lw=1.5, color='darkorange', alpha=0.7, label='H(x,t)')
ax.legend(loc='upper right')
time_txt = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=10)

def init():
    line_e.set_data([], []); line_h.set_data([], []); time_txt.set_text('')
    return line_e, line_h, time_txt

def update(i):
    line_e.set_data(x, frames_e[i]); line_h.set_data(xH, frames_h[i])
    time_txt.set_text(f't = {times[i]:.3f}')
    return line_e, line_h, time_txt

anim = FuncAnimation(fig, update, frames=len(frames_e), init_func=init, interval=40, blit=True)
plt.close(fig)
display(HTML(anim.to_jshtml()))

# %% [markdown]
# ## 3. Extracción de R(f) y T(f) desde la simulación FDTD

# %%
print('Corriendo experimento FDTD...')
res = run_panel_experiment(N=4001, L=4.0, panel_d=0.2, eps_r=4.0, sigma=0.5, pulse_sigma=0.06)

# Referencia analítica
freq = res['freq']
f_anal = np.linspace(0.01, freq.max(), 2000)
R_anal, T_anal = reflection_transmission(f_anal, 0.2, 4.0, 0.5)

f_bw = 1.0 / (2.0 * np.pi * 0.06)
f_max = min(3.0 * f_bw, freq.max())
mask = (freq > 0.05) & (freq < f_max)
mask_a = (f_anal > 0.05) & (f_anal < f_max)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('FDTD vs Analítico — panel: $\\varepsilon_r$=4, $\\sigma$=0.5, d=0.2', fontsize=13)

axes[0].plot(freq[mask], np.abs(res['R'][mask]), 'b-', alpha=0.7, lw=1.2, label='FDTD')
axes[0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analítico (TMM)')
axes[0].set_xlabel('Frecuencia'); axes[0].set_ylabel('|R|')
axes[0].set_title('Coef. de reflexión |R(f)|'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(freq[mask], np.abs(res['T'][mask]), 'b-', alpha=0.7, lw=1.2, label='FDTD')
axes[1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analítico (TMM)')
axes[1].set_xlabel('Frecuencia'); axes[1].set_ylabel('|T|')
axes[1].set_title('Coef. de transmisión |T(f)|'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(freq[mask], np.abs(res['R'][mask])**2 + np.abs(res['T'][mask])**2, 'b-', alpha=0.7, lw=1.2, label='FDTD')
axes[2].plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2, 'r--', lw=2, label='Analítico')
axes[2].axhline(1.0, color='gray', ls=':', alpha=0.6, label='Sin pérdidas')
axes[2].set_xlabel('Frecuencia'); axes[2].set_ylabel(r'$|R|^2 + |T|^2$')
axes[2].set_title('Conservación de energía'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 1.15)

plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Estudio paramétrico: efecto de la conductividad

# %%
fig, axes = plt.subplot_mosaic(
    [['R', 'R', 'T', 'T'],
     ['.', 'E', 'E', '.']],
    figsize=(14, 9)
)
fig.suptitle('Estudio paramétrico: efecto de $\\sigma$ (con $\\varepsilon_r$=4, d=0.2)', fontsize=13)

f_sweep = np.linspace(0.01, 8.0, 1000)
for sig in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
    R_s, T_s = reflection_transmission(f_sweep, 0.2, 4.0, sig)
    axes['R'].plot(f_sweep, np.abs(R_s), label=f'$\\sigma$={sig}')
    axes['T'].plot(f_sweep, np.abs(T_s), label=f'$\\sigma$={sig}')
    axes['E'].plot(f_sweep, np.abs(R_s)**2 + np.abs(T_s)**2, label=f'$\\sigma$={sig}')

axes['R'].set_xlabel('Frecuencia'); axes['R'].set_ylabel('|R|')
axes['R'].set_title('|R(f)|'); axes['R'].legend(); axes['R'].grid(True, alpha=0.3)
axes['T'].set_xlabel('Frecuencia'); axes['T'].set_ylabel('|T|')
axes['T'].set_title('|T(f)|'); axes['T'].legend(); axes['T'].grid(True, alpha=0.3)
axes['E'].axhline(1.0, color='gray', ls=':', alpha=0.6, label='Sin pérdidas')
axes['E'].set_xlabel('Frecuencia'); axes['E'].set_ylabel(r'$|R|^2 + |T|^2$')
axes['E'].set_title('Conservación de energía'); axes['E'].legend(fontsize=8); axes['E'].grid(True, alpha=0.3)
axes['E'].set_ylim(0, 1.15)

plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# ## 5. BONUS: Panel multicapa
# 
# | Capa | Material | $\varepsilon_r$ | $\sigma$ | $d$ |
# |------|----------|-----------------|----------|-----|
# | 1 | Dieléctrico | 2.0 | 0.2 | 0.05 |
# | 2 | Conductivo | 6.0 | 1.0 | 0.08 |
# | 3 | Dieléctrico | 2.0 | 0.2 | 0.05 |

# %%
# Animación del panel multicapa
layers = [
    {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
    {'d': 0.08, 'eps_r': 6.0, 'sigma': 1.0},
    {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
]
total_d = sum(l['d'] for l in layers)

x_ml = np.linspace(0, 4.0, 2001)
xH_ml = (x_ml[1:] + x_ml[:-1]) / 2.0
center_ml = 2.0
pulse_sigma_ml = 0.06
t0_ml = 4.0 * pulse_sigma_ml
pert_ml = lambda t: gaussian(t, t0_ml, pulse_sigma_ml)

fdtd_ml = FDTD1D(x_ml, boundaries=('mur', 'mur'), x_o=1.2, pert=pert_ml, pert_dir=+1)
fdtd_ml.set_multilayer(center_ml, layers)

n_frames_ml = 200
dt_frame_ml = 0.02
frames_e_ml, frames_h_ml, times_ml = [fdtd_ml.get_e()], [fdtd_ml.get_h()], [fdtd_ml.t]
for _ in range(n_frames_ml - 1):
    fdtd_ml.run_until(fdtd_ml.t + dt_frame_ml)
    frames_e_ml.append(fdtd_ml.get_e())
    frames_h_ml.append(fdtd_ml.get_h())
    times_ml.append(fdtd_ml.t)

print(f'Capturados {len(frames_e_ml)} frames multicapa.')

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 4.0); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('x'); ax.set_ylabel('Amplitud')
ax.set_title('Panel multicapa — 3 capas distintas')
ax.grid(True, alpha=0.3)

# Sombrear cada capa con color distinto
colors_ml = ['#b0c4de', '#ffb3b3', '#b0c4de']
edge = center_ml - total_d / 2
for lay, col in zip(layers, colors_ml):
    ax.add_patch(Rectangle((edge, -1.1), lay['d'], 2.2, alpha=0.45, color=col))
    edge += lay['d']

(line_e,) = ax.plot([], [], lw=2, color='royalblue', label='E(x,t)')
(line_h,) = ax.plot([], [], lw=1.5, color='darkorange', alpha=0.7, label='H(x,t)')
ax.legend(loc='upper right')
time_txt = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=10)

def init_ml():
    line_e.set_data([], []); line_h.set_data([], []); time_txt.set_text('')
    return line_e, line_h, time_txt

def update_ml(i):
    line_e.set_data(x_ml, frames_e_ml[i]); line_h.set_data(xH_ml, frames_h_ml[i])
    time_txt.set_text(f't = {times_ml[i]:.3f}')
    return line_e, line_h, time_txt

anim_ml = FuncAnimation(fig, update_ml, frames=len(frames_e_ml), init_func=init_ml, interval=40, blit=True)
plt.close(fig)
display(HTML(anim_ml.to_jshtml()))

# %%
# Comparación FDTD vs analítico para el multicapa
print('Corriendo experimento multicapa...')
res_ml = run_panel_experiment(N=4001, L=6.0, layers=layers, pulse_sigma=0.04)

f_anal_ml = np.linspace(0.01, res_ml['freq'].max(), 2000)
R_anal_ml, T_anal_ml = reflection_transmission_stack(f_anal_ml, layers)

f_bw_ml = 1.0 / (2.0 * np.pi * 0.04)
f_max_ml = min(3.0 * f_bw_ml, res_ml['freq'].max())
mask_ml = (res_ml['freq'] > 0.05) & (res_ml['freq'] < f_max_ml)
mask_a_ml = (f_anal_ml > 0.05) & (f_anal_ml < f_max_ml)

fig, axes = plt.subplot_mosaic(
    [['R', 'R', 'T', 'T'],
     ['.', 'E', 'E', '.']],
    figsize=(14, 9)
)
fig.suptitle('Panel multicapa: FDTD vs TMM analítico', fontsize=13)

axes['R'].plot(res_ml['freq'][mask_ml], np.abs(res_ml['R'][mask_ml]), 'b-', alpha=0.7, lw=1.2, label='FDTD')
axes['R'].plot(f_anal_ml[mask_a_ml], np.abs(R_anal_ml[mask_a_ml]), 'r--', lw=2, label='Analítico (stack TMM)')
axes['R'].set_xlabel('Frecuencia'); axes['R'].set_ylabel('|R|')
axes['R'].set_title('|R(f)| — patrón de interferencia complejo'); axes['R'].legend(); axes['R'].grid(True, alpha=0.3)

axes['T'].plot(res_ml['freq'][mask_ml], np.abs(res_ml['T'][mask_ml]), 'b-', alpha=0.7, lw=1.2, label='FDTD')
axes['T'].plot(f_anal_ml[mask_a_ml], np.abs(T_anal_ml[mask_a_ml]), 'r--', lw=2, label='Analítico (stack TMM)')
axes['T'].set_xlabel('Frecuencia'); axes['T'].set_ylabel('|T|')
axes['T'].set_title('|T(f)|'); axes['T'].legend(); axes['T'].grid(True, alpha=0.3)

E_fdtd = np.abs(res_ml['R'][mask_ml])**2 + np.abs(res_ml['T'][mask_ml])**2
E_anal = np.abs(R_anal_ml[mask_a_ml])**2 + np.abs(T_anal_ml[mask_a_ml])**2
axes['E'].plot(res_ml['freq'][mask_ml], E_fdtd, 'b-', alpha=0.7, lw=1.2, label='FDTD')
axes['E'].plot(f_anal_ml[mask_a_ml], E_anal, 'r--', lw=2, label='Analítico (stack TMM)')
axes['E'].axhline(1.0, color='gray', ls=':', alpha=0.6, label='Sin pérdidas')
axes['E'].set_xlabel('Frecuencia'); axes['E'].set_ylabel(r'$|R|^2 + |T|^2$')
axes['E'].set_title('Conservación de energía'); axes['E'].legend(); axes['E'].grid(True, alpha=0.3)
axes['E'].set_ylim(0, 1.15)

plt.tight_layout(); plt.show()
