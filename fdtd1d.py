import numpy as np
import matplotlib.pyplot as plt

C = 1.0


def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)


class FDTD1D:
    mu0 = 1.0
    eps0 = 1.0

    def __init__(self, x, boundaries=None, x_o=None, pert=None, pert_dir=False):
        self.x = x
        self.xH = (self.x[1:] + self.x[:-1]) / 2.0
        self.dx = x[1] - x[0]
        self.dt = self.dx / C
        self.N = len(x)
        self.e = np.zeros(self.N)
        self.h = np.zeros(self.N - 1)
        self.t = 0.0
        self.boundaries = boundaries

        self.sig = np.zeros(self.N)
        self.eps_r = np.ones(self.N)
        self.eps = self.eps0 * self.eps_r
        self.x_o = x_o
        self.pert = pert
        self.pert_dir = pert_dir

    def load_initial_field(self, e0):
        self.e = e0.copy()

    def set_panel(self, center, d, eps_r=1.0, sigma=0.0):
        """Set a single homogeneous panel on the grid."""
        left = center - d / 2
        right = center + d / 2
        self.eps_r = np.where(
            (self.x >= left) & (self.x <= right), eps_r, 1.0)
        self.sig = np.where(
            (self.x >= left) & (self.x <= right), sigma, 0.0)

    def set_multilayer(self, center, layers):
        """Set a multilayer stack centered at `center`."""
        total_d = sum(lay['d'] for lay in layers)
        edge = center - total_d / 2
        self.eps_r = np.ones_like(self.x)
        self.sig = np.zeros_like(self.x)
        for lay in layers:
            mask = (self.x >= edge) & (self.x < edge + lay['d'])
            self.eps_r[mask] = lay.get('eps_r', 1.0)
            self.sig[mask] = lay.get('sigma', 0.0)
            edge += lay['d']

    def _step(self):
        r = self.dt / self.dx
        self.eps = self.eps0 * self.eps_r

        ca = (2 * self.eps - self.sig * self.dt) / (2 * self.eps + self.sig * self.dt)
        cb = (2 * self.dt / self.dx) / (2 * self.eps + self.sig * self.dt)

        # 1. Guardar valores para Mur
        if self.boundaries is not None:
            if self.boundaries[0] == 'mur':
                e_old_left_0 = self.e[0]
                e_old_left_1 = self.e[1]
            if self.boundaries[1] == 'mur':
                e_old_right_0 = self.e[-1]
                e_old_right_1 = self.e[-2]

        # 2. Actualizar E
        self.e[1:-1] = ca[1:-1] * self.e[1:-1] - cb[1:-1] * (self.h[1:] - self.h[:-1])

        # --- CORRECCIÓN PARA E ---

        if self.pert is not None and self.x_o is not None:
            idx_e = np.argmin(np.abs(self.x - self.x_o))
            val_pert = self.pert(self.t + self.dt)
            
            if self.pert_dir: 
                self.e[idx_e] += cb[idx_e] * val_pert
            else: 
                self.e[idx_e] += val_pert


        # 3. Aplicar condiciones de contorno
        if self.boundaries is not None:
            if self.boundaries[0] == 'PEC': self.e[0] = 0.0
            if self.boundaries[1] == 'PEC': self.e[-1] = 0.0
            if self.boundaries[0] == 'periodic':
                self.e[0] = ca[0] * self.e[0] - cb[0] * (self.h[0] - self.h[-1])
                self.e[-1] = self.e[0]
            if self.boundaries[0] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[0] = e_old_left_1 + mur_coeff * (self.e[1] - e_old_left_0)
            if self.boundaries[1] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[-1] = e_old_right_1 + mur_coeff * (self.e[-2] - e_old_right_0)
            if self.boundaries[0] == 'PMC': self.e[0] -= 2 * r * self.h[0]
            if self.boundaries[1] == 'PMC': self.e[-1] += 2 * r * self.h[-1]

        # 4. Actualizar H
        self.h -= r * (self.e[1:] - self.e[:-1])

        # --- CORRECCIÓN PARA H ---
        if self.pert is not None and self.x_o is not None and self.pert_dir:
            idx_e = np.argmin(np.abs(self.x - self.x_o))
            val_pert = self.pert(self.t + self.dt)
            
            if self.pert_dir == +1:
                self.h[idx_e - 1] += r * val_pert
            elif self.pert_dir == -1:
                self.h[idx_e] -= r * val_pert
                


        self.t += self.dt

    def run_until(self, t_final):
        n_steps = round((t_final - self.t) / self.dt)
        for _ in range(n_steps):
            self._step()
        self.t = t_final

    def get_e(self):
        return self.e.copy()

    def get_h(self):
        return self.h.copy()


# ── Panel experiment helpers ──

def run_panel_experiment(
    N=4001, L=4.0, panel_center=None, panel_d=0.2,
    eps_r=4.0, sigma=0.5, layers=None,
    pulse_x0=0.8, pulse_sigma=0.06,
    t_final=None, obs_offset=0.4,
):
    """
    Run dual FDTD simulations (with and without panel) and record E-field
    time series at observation points on each side of the panel.

    Uses a temporal perturbation source with pert_dir=+1 at pulse_x0 to
    generate an incident pulse travelling in the +x direction. Since c=1,
    the spatial and temporal widths of the Gaussian are equal.

    Parameters
    ----------
    layers : list of dict or None
        If provided, use set_multilayer(). Otherwise use set_panel() with
        eps_r, sigma, panel_d.

    Returns
    -------
    dict with keys: 'freq', 'R', 'T', 'dt', 'n_steps',
                    'E_left_panel', 'E_left_ref',
                    'E_right_panel', 'E_right_ref',
                    'panel_left', 'panel_right'
    """
    x = np.linspace(0, L, N)
    if panel_center is None:
        panel_center = L / 2

    if layers is not None:
        total_d = sum(lay['d'] for lay in layers)
    else:
        total_d = panel_d

    panel_left  = panel_center - total_d / 2
    panel_right = panel_center + total_d / 2

    obs_left_idx  = np.argmin(np.abs(x - (panel_left  - obs_offset)))
    obs_right_idx = np.argmin(np.abs(x - (panel_right + obs_offset)))

    # Temporal Gaussian pulse: peaks at t0 = 4*sigma so it starts from ~zero
    t0_pulse = 4.0 * pulse_sigma
    pert_fn = lambda t: gaussian(t, t0_pulse, pulse_sigma)

    if t_final is None:
        t_final = 2.5 * L

    # ── Simulation WITH panel ──
    fdtd = FDTD1D(x, boundaries=('mur', 'mur'),
                  x_o=pulse_x0, pert=pert_fn, pert_dir=+1)
    if layers is not None:
        fdtd.set_multilayer(panel_center, layers)
    else:
        fdtd.set_panel(panel_center, panel_d, eps_r, sigma)

    n_steps = round(t_final / fdtd.dt)
    E_left_panel  = np.zeros(n_steps)
    E_right_panel = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd._step()
        E_left_panel[i]  = fdtd.e[obs_left_idx]
        E_right_panel[i] = fdtd.e[obs_right_idx]

    # ── Reference simulation WITHOUT panel ──
    fdtd_ref = FDTD1D(x, boundaries=('mur', 'mur'),
                      x_o=pulse_x0, pert=pert_fn, pert_dir=+1)

    E_left_ref  = np.zeros(n_steps)
    E_right_ref = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_ref._step()
        E_left_ref[i]  = fdtd_ref.e[obs_left_idx]
        E_right_ref[i] = fdtd_ref.e[obs_right_idx]

    # ── Extract R(f), T(f) via FFT ──
    dt = fdtd.dt
    freq, R, T = compute_RT_fdtd(
        E_left_panel, E_left_ref, E_right_panel, E_right_ref, dt)

    return {
        'freq': freq, 'R': R, 'T': T,
        'dt': dt, 'n_steps': n_steps,
        'E_left_panel': E_left_panel, 'E_left_ref': E_left_ref,
        'E_right_panel': E_right_panel, 'E_right_ref': E_right_ref,
        'panel_left': panel_left, 'panel_right': panel_right,
    }


def compute_RT_fdtd(E_left_panel, E_left_ref, E_right_panel, E_right_ref, dt):
    """
    Extract R(f) and T(f) from time-domain observation signals via FFT.

    R = FFT(E_reflected) / FFT(E_incident)
    T = FFT(E_transmitted) / FFT(E_incident)
    """
    n = len(E_left_panel)
    E_ref_fft   = np.fft.rfft(E_left_panel - E_left_ref)
    E_trans_fft = np.fft.rfft(E_right_panel)
    E_inc_fft   = np.fft.rfft(E_right_ref)
    freq = np.fft.rfftfreq(n, d=dt)

    valid = np.abs(E_inc_fft) > 1e-10 * np.max(np.abs(E_inc_fft))
    R = np.zeros_like(freq, dtype=complex)
    T = np.zeros_like(freq, dtype=complex)
    R[valid] = E_ref_fft[valid] / E_inc_fft[valid]
    T[valid] = E_trans_fft[valid] / E_inc_fft[valid]

    return freq, R, T