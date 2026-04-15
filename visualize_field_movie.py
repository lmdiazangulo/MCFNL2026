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
# %%

# ---- PLOTS 1D PARA COMPARAR MESHES ------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from fdtd1d import FDTD1DNonUniform, gaussian
from mesh_utils import (
    uniform_mesh, cosine_mesh, geometric_mesh, step_mesh, custom_mesh,
    mesh_stats,
)

def _run_probe(solver, t_final: float, x_probe: float):
    idx = int(np.argmin(np.abs(solver.x - x_probe)))
    times, e_probe = [], []
    while solver.t < t_final - 1e-15:
        solver._step()
        times.append(solver.t)
        e_probe.append(float(solver.e[idx]))
    return np.array(times), np.array(e_probe), solver.dt


def _spectrum(times: np.ndarray, signal: np.ndarray):
    N  = len(signal)
    dt = float(times[1] - times[0])
    freqs = np.fft.rfftfreq(N, d=dt)
    amp   = np.abs(np.fft.rfft(signal)) / N
    return freqs, amp


def _spectral_correlation(f1, a1, f2, a2, f_max: float = None) -> float:

    f_common = np.union1d(f1, f2)
    if f_max is not None:
        f_common = f_common[f_common <= f_max]
    s1 = np.interp(f_common, f1, a1)
    s2 = np.interp(f_common, f2, a2)
    norm = np.linalg.norm(s1) * np.linalg.norm(s2)
    if norm < 1e-30:
        return 0.0
    return float(np.dot(s1, s2) / norm)

MESH_CONFIGS = [              
    ('Uniform (ref)',
     lambda: uniform_mesh(0, 20, 300),
     'reference'),
    ('Cosine – refined both ends\n(smooth)',
     lambda: cosine_mesh(0, 20, 300, 'both'),
     'high'),
    ('Geometric ratio=1.02\n(smooth)',
     lambda: geometric_mesh(0, 20, 300, 1.02),
     'high'),
    ('Step ratio=2 at centre\n(rough)',
     lambda: step_mesh(0, 20, 300, 0.5, 2.0),
     'low'),
    ('Step ratio=4 at centre\n(very rough)',
     lambda: step_mesh(0, 20, 300, 0.5, 4.0),
     'very low'),
]


def _run_config(label, mesh_fn, L=20.0, sigma=0.5, x_src=3.0, x_prb=12.0, t_final=15.0):
    x = mesh_fn()
    solver = FDTD1DNonUniform(x, boundaries=('mur', 'mur'))
    solver.load_initial_field(gaussian(x, x_src, sigma))
    times, ep, dt = _run_probe(solver, t_final, x_prb)
    freqs, amp    = _spectrum(times, ep)
    return {
        'label':   label,
        'x':       x,
        'times':   times,
        'e_probe': ep,
        'freqs':   freqs,
        'amp':     amp,
        'dt':      dt,
    }


def compare_meshes(save: bool = False):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    sigma = 0.5
    x_src = 3.0
    x_prb = 12.0
    f_max = 2.0 / sigma   

    print("Running simulations …")
    results = [_run_config(lbl, fn) for lbl, fn, _ in MESH_CONFIGS]
    ref = results[0]      

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    # ---- Fig 1: mesh distributions ----------------------------------------
    fig1, axes1 = plt.subplots(len(results), 1,
                               figsize=(10, 1.5 * len(results)), sharex=True)
    fig1.suptitle('Node distributions', fontsize=13)
    for ax, res, c in zip(axes1, results, colors):
        ax.plot(res['x'], np.zeros_like(res['x']), '|', color=c,
                markersize=6, markeredgewidth=0.8)
        stats = {'dx_min': np.diff(res['x']).min(),
                 'dx_max': np.diff(res['x']).max()}
        ax.set_ylabel(res['label'].split('\n')[0], fontsize=7)
        ax.set_yticks([])
        ax.text(0.98, 0.5,
                f"dxₘᵢₙ={stats['dx_min']:.4f}  dxₘₐₓ={stats['dx_max']:.4f}",
                transform=ax.transAxes, ha='right', va='center', fontsize=7)
    axes1[-1].set_xlabel('x')
    fig1.tight_layout()

    # ---- Fig 2: time-domain probe signals ----------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.set_title(f'Probe signal at x = {x_prb}', fontsize=13)
    for res, c in zip(results, colors):
        ax2.plot(res['times'], res['e_probe'],
                 label=res['label'].replace('\n', ' '), color=c,
                 lw=1.5 if res is ref else 1.0,
                 alpha=1.0 if res is ref else 0.75)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('E field')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # ---- Fig 3: frequency spectra ------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    axes3[0].set_title('Amplitude spectrum (linear)', fontsize=11)
    axes3[1].set_title('Amplitude spectrum (log)', fontsize=11)

    for res, c in zip(results, colors):
        mask = res['freqs'] <= f_max * 1.5
        for ax in axes3:
            ax.plot(res['freqs'][mask], res['amp'][mask],
                    label=res['label'].replace('\n', ' '), color=c,
                    lw=2.0 if res is ref else 1.0,
                    alpha=1.0 if res is ref else 0.75)

    axes3[1].set_yscale('log')
    for ax in axes3:
        ax.axvline(f_max, color='k', ls='--', lw=0.8,
                   label=f'BW limit ({f_max:.2f})')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig3.tight_layout()

    # ---- Fig 4: correlation bar chart -------------------------------------
    labels = []
    rhos   = []
    bar_colors = []
    for res, c in zip(results[1:], colors[1:]):   
        rho = _spectral_correlation(
            ref['freqs'], ref['amp'],
            res['freqs'], res['amp'],
            f_max=f_max,
        )
        labels.append(res['label'].replace('\n', '\n'))
        rhos.append(rho)
        bar_colors.append(c)
        print(f"  {res['label'].split(chr(10))[0]:40s}  ρ = {rho:.4f}")

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    bars = ax4.barh(labels, rhos, color=bar_colors, edgecolor='k', linewidth=0.6)
    ax4.set_xlim(0.5, 1.02)
    ax4.axvline(1.0, color='k', lw=0.8, ls='--')
    ax4.axvline(0.97, color='green', lw=1.0, ls=':', label='ρ = 0.97 threshold')
    ax4.axvline(0.85, color='orange', lw=1.0, ls=':', label='ρ = 0.85 threshold')
    for bar, rho in zip(bars, rhos):
        ax4.text(max(rho + 0.002, 0.51), bar.get_y() + bar.get_height() / 2,
                 f'{rho:.4f}', va='center', fontsize=9)
    ax4.set_xlabel('Spectral correlation ρ with uniform reference')
    ax4.set_title('Frequency-domain fidelity vs. uniform mesh', fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, axis='x', alpha=0.3)
    fig4.tight_layout()

    if save:
        fig1.savefig('mesh_distributions.png', dpi=150)
        fig2.savefig('time_domain_comparison.png',  dpi=150)
        fig3.savefig('frequency_spectra.png',       dpi=150)
        fig4.savefig('spectral_correlation.png',    dpi=150)
        print("Figures saved.")
    else:
        plt.show()

# ======================================================================
if __name__ == '__main__':
    # FIX: save=True — evita que se quede bloqueado esperando
    #      una ventana interactiva de matplotlib.
    # Cambia a save=False si tienes display disponible (escritorio, Jupyter…)
    compare_meshes(save=True)



# %% Animación

# ── Parámetros ────────────────────────────────────────────────────────────────
X_SRC    = 3.0
SIGMA    = 0.5
T_FINAL  = 18.0
N_FRAMES = 120
FPS      = 25

CONFIGS = [
    ('Uniforme',           uniform_mesh(0, 20, 300),          '#2C3E50'),
    ('Geométrica r=1.02',  geometric_mesh(0, 20, 300, 1.02),  '#E74C3C'),
    ('Coseno (ambos)',     cosine_mesh(0, 20, 300, 'both'),   '#27AE60'),
    ('Step ratio=2',       step_mesh(0, 20, 300, 0.5, 2.0),  '#3498DB'),
    ('Step ratio=4',       step_mesh(0, 20, 300, 0.5, 4.0),  '#9B59B6'),
]

frame_times = np.linspace(0.0, T_FINAL, N_FRAMES + 1)[1:]

# ── Simular y capturar frames ─────────────────────────────────────────────────
print("Simulando y capturando frames...")

all_frames = []
for label, x, color in CONFIGS:
    print(f"  {label}...", end=' ', flush=True)
    s = FDTD1DNonUniform(x, boundaries=('mur', 'mur'))
    s.load_initial_field(gaussian(x, X_SRC, SIGMA))

    frames = []
    for t_target in frame_times:
        while s.t < t_target - 0.5 * s.dt:
            s._step()
        frames.append(s.e.copy())

    all_frames.append(frames)
    print(f"OK  (dt={s.dt:.5f})")

print("Simulaciones completadas.\n")

# ── Construir figura ──────────────────────────────────────────────────────────
print("Generando animación...")

n_configs = len(CONFIGS)
fig = plt.figure(figsize=(13, 10))
fig.patch.set_facecolor('#0F1923')

gs = gridspec.GridSpec(n_configs + 1, 1,
                       hspace=0.08,
                       height_ratios=[1] * n_configs + [0.35],
                       top=0.93, bottom=0.06, left=0.08, right=0.97)

wave_axes = []
for i in range(n_configs):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor('#0F1923')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=8)
    wave_axes.append(ax)

ax_mesh = fig.add_subplot(gs[-1])
ax_mesh.set_facecolor('#0F1923')
for spine in ax_mesh.spines.values():
    spine.set_color('#334155')
ax_mesh.set_yticks([])
ax_mesh.tick_params(colors='#94A3B8', labelsize=8)
ax_mesh.set_xlabel('x', color='#94A3B8', fontsize=10)

for i, (label, x, color) in enumerate(CONFIGS):
    ax_mesh.plot(x, np.full_like(x, i), '|',
                 color=color, markersize=5, markeredgewidth=0.8, alpha=0.7)
ax_mesh.set_xlim(0, 20)
ax_mesh.set_ylim(-0.5, n_configs - 0.5)
ax_mesh.set_yticks(range(n_configs))
ax_mesh.set_yticklabels([cfg[0] for cfg in CONFIGS], fontsize=7.5, color='#94A3B8')

fig.text(0.5, 0.97,
         'Propagación de onda FDTD 1D — Mallas no uniformes',
         ha='center', va='top', fontsize=13, color='white', fontweight='bold')
time_txt = fig.text(0.5, 0.945, 't = 0.000',
                    ha='center', va='top', fontsize=11, color='#94A3B8')

lines, dots = [], []
for i, (label, x, color) in enumerate(CONFIGS):
    ax = wave_axes[i]
    ax.set_xlim(0, 20)
    ax.set_ylim(-1.25, 1.25)

    # Fondo: densidad de celda
    dx   = np.diff(x)
    dx_n = (dx - dx.min()) / (dx.max() - dx.min() + 1e-12)
    for j in range(len(dx)):
        ax.axvspan(x[j], x[j+1], color=color, alpha=0.04 + 0.1 * dx_n[j], linewidth=0)

    ax.axhline(0,    color='#334155', lw=0.6)
    ax.axhline( 1.0, color='#334155', lw=0.4, ls='--')
    ax.axhline(-1.0, color='#334155', lw=0.4, ls='--')
    ax.axvline(X_SRC, color='#F59E0B', lw=0.8, ls=':', alpha=0.7)

    dx_min, dx_max = dx.min(), dx.max()
    ax.text(0.01, 0.88,
            f'{label}   Δxₘᵢₙ={dx_min:.4f}  Δxₘₐₓ={dx_max:.4f}',
            transform=ax.transAxes, fontsize=8.5, color=color,
            fontweight='bold', va='top')

    if i < n_configs - 1:
        ax.set_xticklabels([])
    ax.set_ylabel('E', color='#94A3B8', fontsize=8)

    line, = ax.plot([], [], color=color, lw=1.8, solid_capstyle='round')
    dot,  = ax.plot([], [], 'o', color='white', ms=5, zorder=5)
    lines.append(line)
    dots.append(dot)

progress_bar, = wave_axes[0].plot([], [], color='#F59E0B', lw=3, zorder=7)


def animate(fi):
    t = frame_times[fi]
    time_txt.set_text(f't = {t:.3f}')
    progress_bar.set_data([0, 20.0 * t / T_FINAL], [1.22, 1.22])

    for i, (_, x, _) in enumerate(CONFIGS):
        e = all_frames[i][fi]
        lines[i].set_data(x, e)
        idx_max = int(np.argmax(np.abs(e)))
        dots[i].set_data([x[idx_max]], [e[idx_max]])

    return lines + dots + [progress_bar, time_txt]


ani = animation.FuncAnimation(fig, animate,
                               frames=N_FRAMES,
                               interval=1000 / FPS,
                               blit=False)

writer = animation.PillowWriter(fps=FPS, metadata={'loop': 0})
ani.save('wave_animation.gif', writer=writer, dpi=100)
print("Animación guardada en wave_animation.gif")
# %%
