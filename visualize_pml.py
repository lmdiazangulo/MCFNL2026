# %% [markdown]
# # Visualización de Perfect Matched Layers (PML)
# 
# Este archivo genera las gráficas para la presentación del trabajo final.
# Ejecutar cada celda con **Shift+Enter** en VSCode/Jupyter.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from fdtd1d import FDTD1D, gaussian, C
from fdtd2d import FDTD2D, gaussian2d

plt.rcParams['figure.figsize'] = [12, 5]
plt.rcParams['font.size'] = 12

# %% [markdown]
# ## 1. Comparación 1D: PML vs Sin PML (PEC)
# 
# Una onda viajera hacia la izquierda: con PML se absorbe, con PEC se refleja.

# %% Comparación 1D: PML vs PEC
x = np.linspace(-1.0, 1.0, 401)
xH = (x[1:] + x[:-1]) / 2.0

x0 = 0.0
sigma = 0.08
initial_e = gaussian(x, x0, sigma)
initial_h = -gaussian(xH, x0, sigma)  # Onda viajando a la izquierda

# Tiempos: 0, propagando, llegando al borde, después de absorción
t_values = [0.0, 0.6, 1.2, 1.8]

fig, axes = plt.subplots(2, 4, figsize=(16, 6))

# Crear simuladores NUEVOS para cada fila
for i, t in enumerate(t_values):
    # Reiniciar simuladores para cada tiempo (evitar acumulación de errores)
    if i == 0:
        # Con PML
        fdtd_pml = FDTD1D(x, boundaries=('PML', 'PML'), n_pml=20, pml_order=3, pml_R0=1e-6)
        fdtd_pml.load_initial_field(initial_e.copy())
        fdtd_pml.h[fdtd_pml.Npml_left:fdtd_pml.Npml_left + len(initial_h)] = initial_h.copy()
        
        # Sin PML (PEC)
        fdtd_pec = FDTD1D(x, boundaries=('PEC', 'PEC'))
        fdtd_pec.load_initial_field(initial_e.copy())
        fdtd_pec.h = initial_h.copy()
    
    # Avanzar al tiempo t
    fdtd_pml.run_until(t)
    fdtd_pec.run_until(t)
    
    # Obtener campos
    e_pml = fdtd_pml.get_e()
    e_pec = fdtd_pec.get_e()
    
    # Debug
    print(f"t={t:.1f}: PML max|E|={np.max(np.abs(e_pml)):.4f}, PEC max|E|={np.max(np.abs(e_pec)):.4f}")
    
    # PML (arriba) — mostrar dominio extendido para ver la absorción
    x_full = fdtd_pml.x
    e_pml_full = fdtd_pml.get_e_full()
    axes[0, i].plot(x_full, e_pml_full, 'b-', lw=2, label='E')
    axes[0, i].axvspan(x_full[0], x[0], alpha=0.3, color='orange', label='PML')
    axes[0, i].axvspan(x[-1], x_full[-1], alpha=0.3, color='orange')
    axes[0, i].axvline(x[0], color='orange', lw=1.5, ls='--')
    axes[0, i].axvline(x[-1], color='orange', lw=1.5, ls='--')
    axes[0, i].set_xlim(x_full[0], x_full[-1])
    axes[0, i].set_ylim(-1.2, 1.2)
    axes[0, i].set_title(f't = {t:.1f}')
    axes[0, i].grid(True, alpha=0.3)
    if i == 0:
        axes[0, i].set_ylabel('Con PML\n(absorbe)')
        axes[0, i].legend(loc='upper right', fontsize=9)
    
    # PEC (abajo)
    axes[1, i].plot(x, e_pec, 'r-', lw=2, label='E')
    axes[1, i].axvline(x[0], color='k', lw=2, label='PEC')
    axes[1, i].axvline(x[-1], color='k', lw=2)
    axes[1, i].set_xlim(-1.1, 1.1)
    axes[1, i].set_ylim(-1.2, 1.2)
    axes[1, i].set_xlabel('x')
    axes[1, i].grid(True, alpha=0.3)
    if i == 0:
        axes[1, i].set_ylabel('Sin PML (PEC)\n(refleja)')
        axes[1, i].legend(loc='upper right', fontsize=9)

plt.suptitle('Comparación 1D: Onda viajera (H = -E → viaja a la izquierda)', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Perfil de conductividad σ(x) en la PML

# %% Perfil de sigma
x = np.linspace(-1.0, 1.0, 201)
fdtd = FDTD1D(x, boundaries=('PML', 'PML'), n_pml=20, pml_order=3, pml_R0=1e-6)

fig, ax = plt.subplots(figsize=(10, 4))

x_full = fdtd.x
ax.plot(x_full, fdtd.sig, 'b-', lw=2, label='σ(x)')
ax.axvspan(x_full[0], x_full[fdtd.Npml_left], alpha=0.3, color='orange', label='PML izquierda')
ax.axvspan(x_full[-fdtd.Npml_right], x_full[-1], alpha=0.3, color='orange', label='PML derecha')
ax.axvspan(x_full[fdtd.Npml_left], x_full[-fdtd.Npml_right], alpha=0.2, color='green', label='Dominio físico')

ax.set_xlabel('x')
ax.set_ylabel('σ [S/m normalizado]')
ax.set_title('Perfil de conductividad en la PML (orden m=3)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Animación 1D: PML absorbiendo onda en ambos sentidos
#
# Condición inicial: pulso Gaussiano puro en E, H=0 → se descompone en dos
# frentes viajeros (izquierda y derecha) de amplitud 0.5 cada uno.
# Ambos entran en su PML correspondiente y son absorbidos.

# %% Animación 1D
x = np.linspace(-1.0, 1.0, 401)
xH = (x[1:] + x[:-1]) / 2.0

# H=0 → onda se divide en mitad izquierda y mitad derecha
initial_e = gaussian(x, 0.0, 0.08)
initial_h = np.zeros(len(xH))

fdtd = FDTD1D(x, boundaries=('PML', 'PML'), n_pml=20, pml_order=3)
fdtd.load_initial_field(initial_e.copy())
fdtd.h[fdtd.Npml_left:fdtd.Npml_left + len(initial_h)] = initial_h.copy()

n_frames = 180
dt_per_frame = 0.015

x_full_anim = fdtd.x  # dominio extendido incluyendo PML
frames_e = [fdtd.get_e_full().copy()]
times_anim = [fdtd.t]

for _ in range(n_frames - 1):
    fdtd.run_until(fdtd.t + dt_per_frame)
    frames_e.append(fdtd.get_e_full().copy())
    times_anim.append(fdtd.t)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(x_full_anim[0], x_full_anim[-1])
ax.set_ylim(-0.7, 1.1)
ax.set_xlabel('x')
ax.set_ylabel('E(x, t)')
ax.set_title('FDTD 1D con PML – Pulso dividiéndose y atenuándose en ambos lados')
ax.grid(True, alpha=0.3)

# Marcar regiones PML (fuera del dominio físico [-1, 1])
ax.axvspan(x_full_anim[0], x[0], alpha=0.3, color='orange', label='PML')
ax.axvspan(x[-1], x_full_anim[-1], alpha=0.3, color='orange')
ax.axvline(x[0], color='orange', lw=1.5, ls='--')
ax.axvline(x[-1], color='orange', lw=1.5, ls='--')

line, = ax.plot([], [], 'b-', lw=2, label='E(x,t)')
time_txt = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=11)
ax.legend(loc='upper right')

def init():
    line.set_data([], [])
    time_txt.set_text('')
    return line, time_txt

def update(i):
    line.set_data(x_full_anim, frames_e[i])
    time_txt.set_text(f't = {times_anim[i]:.2f}')
    return line, time_txt

anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=40, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())

# %% [markdown]
# ## 4. Comparación 2D: Con PML vs Sin PML

# %% Comparación 2D
x = np.linspace(-1.0, 1.0, 101)
y = np.linspace(-1.0, 1.0, 101)
X, Y = np.meshgrid(x, y, indexing='ij')

initial_Ez = gaussian2d(X, Y, 0.0, 0.0, 0.1)

# Tiempos más largos para ver absorción
t_values = [0.0, 0.5, 1.0, 1.8]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

vmax = 0.8

for i, t in enumerate(t_values):
    if i == 0:
        # Con PML
        fdtd_pml = FDTD2D(x, y, boundaries=('PML', 'PML', 'PML', 'PML'), n_pml=15)
        fdtd_pml.load_initial_field(initial_Ez.copy())
        
        # Sin PML (PEC)
        fdtd_pec = FDTD2D(x, y, boundaries=('PEC', 'PEC', 'PEC', 'PEC'))
        fdtd_pec.load_initial_field(initial_Ez.copy())
    
    fdtd_pml.run_until(t)
    fdtd_pec.run_until(t)
    
    Ez_pml = fdtd_pml.get_Ez()
    Ez_pec = fdtd_pec.get_Ez()

    print(f"2D t={t:.1f}: PML max|Ez|={np.max(np.abs(Ez_pml)):.4f}, PEC max|Ez|={np.max(np.abs(Ez_pec)):.4f}")

    # Escala dinámica por columna: usa el máximo entre PML y PEC para comparación justa
    vmax_col = max(np.max(np.abs(Ez_pml)), np.max(np.abs(Ez_pec)), 1e-6)

    # Con PML (arriba)
    im1 = axes[0, i].imshow(Ez_pml.T, extent=[-1, 1, -1, 1],
                            origin='lower', cmap='RdBu_r', vmin=-vmax_col, vmax=vmax_col)
    axes[0, i].set_title(f't = {t:.1f}')
    if i == 0:
        axes[0, i].set_ylabel('Con PML\n(absorbe)')
    axes[0, i].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, i], shrink=0.8)

    # Sin PML (abajo)
    im2 = axes[1, i].imshow(Ez_pec.T, extent=[-1, 1, -1, 1],
                            origin='lower', cmap='RdBu_r', vmin=-vmax_col, vmax=vmax_col)
    axes[1, i].set_xlabel('x')
    if i == 0:
        axes[1, i].set_ylabel('Sin PML (PEC)\n(refleja)')
    axes[1, i].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1, i], shrink=0.8)

plt.suptitle('Comparación 2D: Pulso Gaussiano', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Perfil de σ en 2D (mostrando esquinas)

# %% Perfil sigma 2D
x = np.linspace(-1.0, 1.0, 51)
y = np.linspace(-1.0, 1.0, 51)
n_pml = 8

fdtd = FDTD2D(x, y, boundaries=('PML', 'PML', 'PML', 'PML'), n_pml=n_pml)

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(fdtd.sig.T, extent=[fdtd.x[0], fdtd.x[-1], fdtd.y[0], fdtd.y[-1]], 
               origin='lower', cmap='hot')
plt.colorbar(im, ax=ax, label='σ [normalizado]')

# Marcar dominio físico
rect = plt.Rectangle((x[0], y[0]), x[-1]-x[0], y[-1]-y[0], 
                      fill=False, edgecolor='cyan', linewidth=2, linestyle='--', label='Dominio físico')
ax.add_patch(rect)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Conductividad σ en PML 2D\n(Esquinas: σ = σₓ + σᵧ)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Animación 2D: Pulso con PML

# %% Animación 2D
x = np.linspace(-1.0, 1.0, 101)
y = np.linspace(-1.0, 1.0, 101)
X, Y = np.meshgrid(x, y, indexing='ij')

initial_Ez = gaussian2d(X, Y, 0.0, 0.0, 0.1)

n_pml_2d = 15
fdtd = FDTD2D(x, y, boundaries=('PML', 'PML', 'PML', 'PML'), n_pml=n_pml_2d)
fdtd.load_initial_field(initial_Ez.copy())

n_frames = 120
dt_per_frame = 0.025

# Usar dominio extendido para ver la atenuación en la PML
x_full_2d = fdtd.x
y_full_2d = fdtd.y
frames_Ez = [fdtd.get_Ez_full().copy()]
times_2d = [fdtd.t]

for _ in range(n_frames - 1):
    fdtd.run_until(fdtd.t + dt_per_frame)
    frames_Ez.append(fdtd.get_Ez_full().copy())
    times_2d.append(fdtd.t)

fig, ax = plt.subplots(figsize=(8, 7))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('FDTD 2D con PML – dominio extendido')

extent_full = [x_full_2d[0], x_full_2d[-1], y_full_2d[0], y_full_2d[-1]]

# Colormap intenso: 'seismic' con rango reducido para mayor saturación
im = ax.imshow(frames_Ez[0].T, extent=extent_full, origin='lower',
               cmap='seismic', vmin=-0.3, vmax=0.3, animated=True)
plt.colorbar(im, ax=ax, label='Ez')

# Sombrear región PML con naranja semi-transparente
dx2d = x[1] - x[0]
dy2d = y[1] - y[0]
d_pml_x = n_pml_2d * dx2d
d_pml_y = n_pml_2d * dy2d

# Franjas PML: izquierda, derecha, abajo, arriba
from matplotlib.patches import Rectangle as Rect
pml_patches = [
    Rect((x_full_2d[0],  y_full_2d[0]),  d_pml_x,              y_full_2d[-1] - y_full_2d[0]),  # izq
    Rect((x[-1],         y_full_2d[0]),  d_pml_x,              y_full_2d[-1] - y_full_2d[0]),  # der
    Rect((x[0],          y_full_2d[0]),  x[-1] - x[0],         d_pml_y),                        # inf
    Rect((x[0],          y[-1]),         x[-1] - x[0],         d_pml_y),                        # sup
]
for p in pml_patches:
    p.set(facecolor='orange', alpha=0.25, linewidth=0)
    ax.add_patch(p)

# Borde del dominio físico (interior de la PML)
rect_phys = plt.Rectangle((x[0], y[0]), x[-1] - x[0], y[-1] - y[0],
                           fill=False, edgecolor='lime', linewidth=2,
                           linestyle='--', label='Dominio físico')
ax.add_patch(rect_phys)

# Borde exterior de la PML
rect_pml = plt.Rectangle((x_full_2d[0], y_full_2d[0]),
                          x_full_2d[-1] - x_full_2d[0],
                          y_full_2d[-1] - y_full_2d[0],
                          fill=False, edgecolor='orange', linewidth=2,
                          linestyle='-', label='PML')
ax.add_patch(rect_pml)

ax.legend(loc='upper right', fontsize=9)

time_txt = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

def init():
    im.set_array(frames_Ez[0].T)
    time_txt.set_text('')
    return im, time_txt

def update(i):
    im.set_array(frames_Ez[i].T)
    time_txt.set_text(f't = {times_2d[i]:.2f}')
    return im, time_txt

anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=50, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())

# %% [markdown]
# ## 7. Convergencia PML 1D: Reflexión vs anchura y vs Δx
#
# Panel izquierdo: se varía el número de celdas PML (Δx fijo) para ver cómo decrece la reflexión.
# Panel derecho: se fija el espesor físico de la PML y se refina la malla para ver la convergencia espacial.

# %% Convergencia
x_domain = (-1.0, 1.0)

# --- Parte 1: n_pml variable, Δx fijo ---
x = np.linspace(x_domain[0], x_domain[1], 201)
xH = (x[1:] + x[:-1]) / 2.0

initial_e = gaussian(x, 0.0, 0.05)
initial_h = -gaussian(xH, 0.0, 0.05)

n_pml_values = [5, 10, 15, 20, 25, 30]
residuals = []

for n_pml in n_pml_values:
    fdtd = FDTD1D(x, boundaries=('PML', 'PML'), n_pml=n_pml, pml_order=3, pml_R0=1e-6)
    fdtd.load_initial_field(initial_e.copy())
    fdtd.h[fdtd.Npml_left:fdtd.Npml_left + len(initial_h)] = initial_h.copy()

    dx = x[1] - x[0]
    d_pml = n_pml * dx
    t_final = ((x[-1] - x[0]) / 2.0 + 2.0 * d_pml) / C
    fdtd.run_until(t_final)
    residuals.append(np.max(np.abs(fdtd.get_e())))

# --- Parte 2: espesor físico PML fijo, Δx variable ---
sigma_gauss = 0.05
pml_order_fixed = 3
pml_R0_fixed = 1e-6

# Espesor físico de PML constante: igual al de la referencia (n_pml=20, N=201)
dx_ref = (x_domain[1] - x_domain[0]) / (201 - 1)
d_pml_fixed = 20 * dx_ref   # espesor físico fijo

N_values = [51, 101, 201, 401, 801]
dx_values = []
residuals_dx = []

for N in N_values:
    x_conv = np.linspace(x_domain[0], x_domain[1], N)
    xH_conv = (x_conv[1:] + x_conv[:-1]) / 2.0
    dx_conv = x_conv[1] - x_conv[0]

    # n_pml se ajusta para mantener el espesor físico constante
    n_pml_conv = max(1, round(d_pml_fixed / dx_conv))

    initial_e_conv = gaussian(x_conv,  0.0, sigma_gauss)
    initial_h_conv = -gaussian(xH_conv, 0.0, sigma_gauss)

    fdtd_conv = FDTD1D(x_conv, boundaries=('PML', 'PML'),
                       n_pml=n_pml_conv, pml_order=pml_order_fixed, pml_R0=pml_R0_fixed)
    fdtd_conv.load_initial_field(initial_e_conv.copy())
    fdtd_conv.h[fdtd_conv.Npml_left:fdtd_conv.Npml_left + len(initial_h_conv)] = initial_h_conv.copy()

    t_final_conv = ((x_conv[-1] - x_conv[0]) / 2.0 + 2.0 * d_pml_fixed) / C
    fdtd_conv.run_until(t_final_conv)

    dx_values.append(dx_conv)
    residuals_dx.append(np.max(np.abs(fdtd_conv.get_e())))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: n_pml variable, dx fijo
axes[0].semilogy(n_pml_values, residuals, 'bo-', lw=2, markersize=10)
axes[0].set_xlabel('Número de celdas PML  (Δx fijo)')
axes[0].set_ylabel('Residuo máximo |E|')
axes[0].set_title('Reflexión vs anchura PML\n(Δx fijo, n_pml variable)')
axes[0].grid(True, which='both', alpha=0.3)

# Panel derecho: dx variable, espesor físico PML fijo
axes[1].loglog(dx_values, residuals_dx, 'rs-', lw=2, markersize=10)
axes[1].set_xlabel('Δx  (d_pml fijo = {:.3f})'.format(d_pml_fixed))
axes[1].set_ylabel('Residuo máximo |E|')
axes[1].set_title('Reflexión vs resolución espacial\n(d_pml físico fijo, Δx variable)')
axes[1].invert_xaxis()  # Δx decreciente → resolución creciente hacia la derecha
axes[1].grid(True, which='both', alpha=0.3)

plt.suptitle('Convergencia PML 1D', fontsize=14)
plt.tight_layout()
plt.show()

print("Convergencia espacial (d_pml fijo = {:.3f}):".format(d_pml_fixed))
for dx_v, r in zip(dx_values, residuals_dx):
    print(f"  dx = {dx_v:.5f}  →  residuo = {r:.2e}")

# %%
