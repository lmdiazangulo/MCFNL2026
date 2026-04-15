import numpy as np
from fdtd1d import C, FDTD1D, gaussian

def panel_transfer_matrix(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    omega = 2.0 * np.pi * freq

    eps_c = eps_r - 1j * sigma / omega
    gamma = 1j * omega * np.sqrt(mu_r * eps_c)
    eta = np.sqrt(mu_r / eps_c)

    gd = gamma * d
    ch = np.cosh(gd)
    sh = np.sinh(gd)

    Phi = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi[:, 0, 0] = ch
    Phi[:, 0, 1] = eta * sh
    Phi[:, 1, 0] = sh / eta
    Phi[:, 1, 1] = ch
    return Phi


def stack_transfer_matrix(freq, layers):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    Phi_total = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi_total[:, 0, 0] = 1.0
    Phi_total[:, 1, 1] = 1.0

    for layer in layers:
        Phi_i = panel_transfer_matrix(
            freq,
            d=layer['d'],
            eps_r=layer.get('eps_r', 1.0),
            sigma=layer.get('sigma', 0.0),
            mu_r=layer.get('mu_r', 1.0),
        )
        Phi_new = np.zeros_like(Phi_total)
        Phi_new[:, 0, 0] = Phi_total[:, 0, 0] * Phi_i[:, 0, 0] + Phi_total[:, 0, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 0, 1] = Phi_total[:, 0, 0] * Phi_i[:, 0, 1] + Phi_total[:, 0, 1] * Phi_i[:, 1, 1]
        Phi_new[:, 1, 0] = Phi_total[:, 1, 0] * Phi_i[:, 0, 0] + Phi_total[:, 1, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 1, 1] = Phi_total[:, 1, 0] * Phi_i[:, 0, 1] + Phi_total[:, 1, 1] * Phi_i[:, 1, 1]
        Phi_total = Phi_new

    return Phi_total


def RT_from_transfer_matrix(Phi):
    A = Phi[:, 0, 0]
    B = Phi[:, 0, 1]
    C = Phi[:, 1, 0]
    D = Phi[:, 1, 1]
    denom = A + B + C + D
    R = (A + B - C - D) / denom
    T = 2.0 / denom
    return R, T


def reflection_transmission(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    Phi = panel_transfer_matrix(freq, d, eps_r, sigma, mu_r)
    return RT_from_transfer_matrix(Phi)


def reflection_transmission_stack(freq, layers):
    Phi = stack_transfer_matrix(freq, layers)
    return RT_from_transfer_matrix(Phi)

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

    Uses a temporal H-field perturbation source (pert_dir=True) at pulse_x0
    to generate the incident pulse. Since c=1, the spatial and temporal
    widths of the Gaussian are equal.

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

    panel_left = panel_center - total_d / 2
    panel_right = panel_center + total_d / 2

    obs_left_idx = np.argmin(np.abs(x - (panel_left - obs_offset)))
    obs_right_idx = np.argmin(np.abs(x - (panel_right + obs_offset)))

    # Temporal Gaussian pulse: peaks at t0 = 4*sigma so it starts from ~zero
    t0_pulse = 4.0 * pulse_sigma
    pert_fn = lambda t: gaussian(t, t0_pulse, pulse_sigma)

    if t_final is None:
        t_final = 2.5 * L

    # ── Simulation WITH panel ──
    fdtd = FDTD1D(x, boundaries=('mur', 'mur'),
                  x_o=pulse_x0, pert=pert_fn, pert_dir=True)
    if layers is not None:
        fdtd.set_multilayer(panel_center, layers)
    else:
        fdtd.set_panel(panel_center, panel_d, eps_r, sigma)

    n_steps = round(t_final / fdtd.dt)
    E_left_panel = np.zeros(n_steps)
    E_right_panel = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd._step()
        E_left_panel[i] = fdtd.e[obs_left_idx]
        E_right_panel[i] = fdtd.e[obs_right_idx]

    # ── Reference simulation WITHOUT panel ──
    fdtd_ref = FDTD1D(x, boundaries=('mur', 'mur'),
                      x_o=pulse_x0, pert=pert_fn, pert_dir=True)

    E_left_ref = np.zeros(n_steps)
    E_right_ref = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_ref._step()
        E_left_ref[i] = fdtd_ref.e[obs_left_idx]
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
    E_ref_fft = np.fft.rfft(E_left_panel - E_left_ref)
    E_trans_fft = np.fft.rfft(E_right_panel)
    E_inc_fft = np.fft.rfft(E_right_ref)
    freq = np.fft.rfftfreq(n, d=dt)

    valid = np.abs(E_inc_fft) > 1e-10 * np.max(np.abs(E_inc_fft))
    R = np.zeros_like(freq, dtype=complex)
    T = np.zeros_like(freq, dtype=complex)
    R[valid] = E_ref_fft[valid] / E_inc_fft[valid]
    T[valid] = E_trans_fft[valid] / E_inc_fft[valid]

    return freq, R, T
