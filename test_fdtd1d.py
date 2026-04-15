import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd1d import FDTD1D, FDTD1DNonUniform, C, gaussian, panel_transfer_matrix, RT_from_transfer_matrix, stack_transfer_matrix
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


def _spectral_correlation(f1, a1, f2, a2,
                           f_max: float = None) -> float:
    f_common = np.union1d(f1, f2)
    if f_max is not None:
        f_common = f_common[f_common <= f_max]
    s1 = np.interp(f_common, f1, a1)
    s2 = np.interp(f_common, f2, a2)
    norm = np.linalg.norm(s1) * np.linalg.norm(s2)
    if norm < 1e-30:
        return 0.0
    return float(np.dot(s1, s2) / norm)


def _make_solver(x, sigma_pulse=0.3, x0_pulse=None, boundaries=('mur', 'mur')):
    if x0_pulse is None:
        x0_pulse = x[len(x) // 4]
    solver = FDTD1DNonUniform(x, boundaries=boundaries)
    e0 = gaussian(x, x0_pulse, sigma_pulse)
    solver.load_initial_field(e0)
    return solver




class TestWaveSpeed:
    @pytest.mark.parametrize("mesh_fn, label", [
        (lambda: cosine_mesh(0, 20, 400, 'both'),   'cosine-both'),
        (lambda: geometric_mesh(0, 20, 400, 1.02),  'geometric-1.02'),  
        (lambda: step_mesh(0, 20, 400, 0.5, 2.0),   'step-ratio2'),
    ])
    def test_pulse_travel_time(self, mesh_fn, label):
        x  = mesh_fn()
        x0 = x[len(x) // 4]
        x1 = x[3 * len(x) // 4]
        d  = x1 - x0

        solver = _make_solver(x, sigma_pulse=0.3, x0_pulse=x0)

        idx_probe = int(np.argmin(np.abs(x - x1)))
        t_travel  = d / C
        t_margin  = 1.5 * t_travel
        e_probe   = []
        times     = []
        while solver.t < t_margin:
            solver._step()
            e_probe.append(float(solver.e[idx_probe]))
            times.append(solver.t)

        t_peak = times[int(np.argmax(np.abs(e_probe)))]
        rel_err = abs(t_peak - t_travel) / t_travel

        assert rel_err < 0.05, (
            f"[{label}] Peak arrived at t={t_peak:.4f}, expected {t_travel:.4f} "
            f"(rel error {rel_err:.3%} > 5%)"
        )


class TestEnergyConservation:
    @pytest.mark.parametrize("mesh_fn, label, tol", [
        (lambda: uniform_mesh(0, 20, 400),          'uniform',      0.01),
        (lambda: cosine_mesh(0, 20, 400, 'both'),   'cosine',       0.05),
        (lambda: step_mesh(0, 20, 400, 0.5, 2.0),   'step-ratio2',  0.10),
    ])
    def test_energy_non_increasing(self, mesh_fn, label, tol):

        x      = mesh_fn()
        solver = _make_solver(x, sigma_pulse=0.4, boundaries=('mur', 'mur'))

        def total_energy(s):
            dx_h = np.diff(s.x)
            dx_e = np.diff(s.xH)
            e_energy = 0.5 * s.eps0 * np.dot(s.e[1:-1] ** 2, dx_e)
            h_energy = 0.5 * s.mu0  * np.dot(s.h         ** 2, dx_h)
            return e_energy + h_energy

        e0 = total_energy(solver)
        solver.run_until(15.0)
        e1 = total_energy(solver)

        assert e1 <= e0 * (1.0 + tol), (
            f"[{label}] Energy increased from {e0:.4e} to {e1:.4e} "
            f"(ratio {e1/e0:.4f} > {1+tol:.4f})"
        )


class TestBoundaryConditions:
    def test_pec_field_zero_at_boundary(self):
        x      = cosine_mesh(0, 10, 201, 'both')
        solver = _make_solver(x, sigma_pulse=0.3,
                              boundaries=('PEC', 'PEC'))
        solver.run_until(2.0)
        assert solver.e[0] == pytest.approx(0.0, abs=1e-12)
        assert solver.e[-1] == pytest.approx(0.0, abs=1e-12)

    def test_mur_reduces_boundary_reflection(self):

        x_pec = cosine_mesh(0, 10, 201, 'both')
        x_mur = cosine_mesh(0, 10, 201, 'both')

        s_pec = _make_solver(x_pec, sigma_pulse=0.3, boundaries=('PEC', 'PEC'))
        s_mur = _make_solver(x_mur, sigma_pulse=0.3, boundaries=('mur', 'mur'))

        t_exit = 10.0 / C + 1.0

        def rms_e(s):
            return float(np.sqrt(np.mean(s.e ** 2)))

        s_pec.run_until(t_exit)
        s_mur.run_until(t_exit)

        assert rms_e(s_mur) < 0.2 * rms_e(s_pec), (
            f"rms_e PEC={rms_e(s_pec):.4e}, Mur={rms_e(s_mur):.4e}"
        )


class TestFrequencyCorrelation:

    @pytest.mark.parametrize("mesh_fn, rho_min, label", [
        (lambda: cosine_mesh(0, 20, 300, 'both'),   0.97, 'cosine-both'),
        (lambda: cosine_mesh(0, 20, 300, 'left'),   0.97, 'cosine-left'),
        (lambda: geometric_mesh(0, 20, 300, 1.02),  0.97, 'geometric-1.02'),
        (lambda: step_mesh(0, 20, 300, 0.5, 2.0),   0.85, 'step-ratio2'),
    ])
    def test_spectral_correlation(self, mesh_fn, rho_min, label):

        L      = 20.0
        N_ref  = 300
        sigma  = 0.5
        x_src  = 3.0
        x_prb  = 12.0
        t_final = 15.0

        # ---- Reference (uniform) ----------------------------------------
        x_ref   = uniform_mesh(0, L, N_ref)
        s_ref   = FDTD1DNonUniform(x_ref, boundaries=('mur', 'mur'))
        e0_ref  = gaussian(x_ref, x_src, sigma)
        s_ref.load_initial_field(e0_ref)
        t_ref, ep_ref, _ = _run_probe(s_ref, t_final, x_prb)
        f_ref, a_ref     = _spectrum(t_ref, ep_ref)

        # ---- Non-uniform ------------------------------------------------
        x_nu   = mesh_fn()
        s_nu   = FDTD1DNonUniform(x_nu, boundaries=('mur', 'mur'))
        e0_nu  = gaussian(x_nu, x_src, sigma)
        s_nu.load_initial_field(e0_nu)
        t_nu, ep_nu, _ = _run_probe(s_nu, t_final, x_prb)
        f_nu, a_nu     = _spectrum(t_nu, ep_nu)

        f_max = 2.0 / sigma
        rho   = _spectral_correlation(f_ref, a_ref, f_nu, a_nu, f_max=f_max)

        assert rho >= rho_min, (
            f"[{label}] Spectral correlation {rho:.4f} < {rho_min} (threshold)"
        )


def test_fdtd_solves_basic_propagation():
    x = np.linspace(-1, 1, 201)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)

    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_e)

    t_final = 0.2
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()

    e_expected = 0.5 * gaussian(x, -t_final*C, sigma) \
     + 0.5 * gaussian(x, t_final*C, sigma)

    plt.plot(x, e_solved)
    plt.plot(x, e_expected)

    assert np.corrcoef(e_solved, e_expected)[0,1] > 0.99


def test_fdtd_PEC_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('PEC', 'PEC')

    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = - initial_e
    h_expected = np.zeros_like(h_solved)

    assert np.corrcoef(e_solved, e_expected)[0,1] > 0.99
    assert np.allclose(h_solved, h_expected, atol=0.01)


def test_fdtd_periodic_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('periodic', 'periodic')

    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    initial_h = np.zeros_like(initial_e[:-1])

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = initial_e
    h_expected = initial_h

    assert np.corrcoef(e_solved, e_expected)[0, 1] > 0.99
    assert np.allclose(h_solved, h_expected, atol=1e-10)

def test_fdtd_PMC_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('PMC', 'PMC')
    
    x0 = 0.0
    sigma = 0.05
    initial_h = gaussian(xH, x0, sigma)
    initial_e = np.zeros_like(x)

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    h_expected = - initial_h
    e_expected = np.zeros_like(e_solved)

    assert np.corrcoef(h_solved, h_expected)[0,1] > 0.99
    assert np.max(np.abs(e_solved - e_expected)) < 1e-8

def test_fdtd_perturbation():
    xMax = 1
    xMin = -1
    L = xMax - xMin

    x = np.linspace(xMin,xMax,201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur','mur') 
    x_o = 0
    initial_e = np.zeros_like(x)

    def my_pert(t):
        return np.sin(2*np.pi*2*C/L*t)
    
    fdtd = FDTD1D(x,boundaries,x_o,pert = lambda t: my_pert(t))
    fdtd.load_initial_field(initial_e)
    t_final = L / (2*C) 
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()
    
    e_expected = my_pert(L/C/2 - np.abs(x)/C)/2
    h_expected = my_pert(L/C/2 - xH/C + fdtd.dt/2)/2
    # plt.plot(xH,h_expected)
    # plt.show()

    assert np.allclose(e_solved, e_expected, atol=1e-1)
    assert np.allclose(h_solved, h_expected, atol=1e-1)

def test_fdtd_eh_perturbation():
    xMax = 1
    xMin = -1
    L = xMax - xMin

    x = np.linspace(xMin,xMax,201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur','mur') 
    x_o = 0
    initial_e = np.zeros_like(x)

    def my_pert(t):
        #return np.sin(2*np.pi*2*C/L*t)
        return gaussian(t, 0.1, 0.02)
    fdtd = FDTD1D(x,boundaries,x_o,pert = lambda t: my_pert(t), pert_dir = True)
    fdtd.load_initial_field(initial_e)
    t_final = L / (2*C) 
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()
    
    e_expected = my_pert(L/C/2 - np.abs(x)/C)
    h_expected = my_pert(L/C/2 - xH/C+ + fdtd.dt/2)

    e_expected[x<0.1] = 0
    h_expected[xH<0.1] = 0
    # plt.plot(x,e_expected)
    # plt.show()

    assert np.allclose(e_solved, e_expected, atol=1e-3)
    assert np.allclose(h_solved, h_expected, atol=1e-3)

def test_fdtd_mur_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')

    x0 = 0.0
    sigma = 0.05

    initial_e = gaussian(x, x0, sigma)
    initial_h = -gaussian(xH, x0, sigma)

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    t_final = 1.2
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    assert np.allclose(e_solved, 0.0, atol=1e-2)
    assert np.allclose(h_solved, 0.0, atol=1e-2)

def test_fdtd_dissipative_exact():
    xMax = 1.0
    xMin = -1.0
    L = xMax - xMin

    x = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('periodic', 'periodic')

    x0 = 0.0
    k = np.pi  
    initial_e = np.sin((x-x0)*k)
    initial_h = np.zeros_like(initial_e[:-1])

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()
    
    sigma_val = 1.0
    eps_r_val = 2.0
    
    fdtd.sig = np.full_like(x, sigma_val)       
    fdtd.eps_r = np.full_like(x, eps_r_val)
    fdtd.eps = fdtd.eps0 * fdtd.eps_r

    gamma = sigma_val / (2 * fdtd.eps0 * eps_r_val)
    w0_sq = (k**2) / (fdtd.mu0 * fdtd.eps0 * eps_r_val)
    wd = np.sqrt(w0_sq - gamma**2)

    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()
    
    t_func_E = np.exp(-gamma * t_final) * (np.cos(wd * t_final) - (gamma / wd) * np.sin(wd * t_final))
    t_func_H = np.exp(-gamma * t_final) * (k / (fdtd.mu0 * wd)) * np.sin(wd * t_final)

    e_expected = t_func_E * np.sin(k * (x - x0))
    h_expected = -t_func_H * np.cos(k * (xH - x0)) 

    assert np.allclose(e_solved, e_expected, atol=1e-2)
    assert np.allclose(h_solved, h_expected, atol=1e-2)

def test_fdtd_conductive_panel_reflection():
    L = 4.0
    N = 4001
    x = np.linspace(0, L, N)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')

    pulse_x0 = 0.8
    sigma = 0.06

    panel_center = L / 2
    panel_d = 0.2
    panel_left = panel_center - panel_d / 2
    panel_right = panel_center + panel_d / 2
    eps_r_panel = 4.0
    sigma_panel = 0.5

    obs_left_idx = np.argmin(np.abs(x - (panel_left - 0.4)))
    obs_right_idx = np.argmin(np.abs(x - (panel_right + 0.4)))

    initial_e = gaussian(x, pulse_x0, sigma)
    initial_h = gaussian(xH, pulse_x0, sigma)

    t_final = 2.5 * L

    # Simulation WITH panel
    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()
    fdtd.eps_r = np.where((x >= panel_left) & (x <= panel_right), eps_r_panel, 1.0)
    fdtd.sig = np.where((x >= panel_left) & (x <= panel_right), sigma_panel, 0.0)

    n_steps = round(t_final / fdtd.dt)
    E_left_panel = np.zeros(n_steps)
    E_right_panel = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd._step()
        E_left_panel[i] = fdtd.e[obs_left_idx]
        E_right_panel[i] = fdtd.e[obs_right_idx]

    # Reference simulation WITHOUT panel (free space)
    fdtd_ref = FDTD1D(x, boundaries)
    fdtd_ref.load_initial_field(initial_e)
    fdtd_ref.h = initial_h.copy()

    E_left_ref = np.zeros(n_steps)
    E_right_ref = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_ref._step()
        E_left_ref[i] = fdtd_ref.e[obs_left_idx]
        E_right_ref[i] = fdtd_ref.e[obs_right_idx]

    # Extract R(f), T(f) via FFT
    dt = fdtd.dt
    E_ref_fft = np.fft.rfft(E_left_panel - E_left_ref)
    E_trans_fft = np.fft.rfft(E_right_panel)
    E_inc_fft = np.fft.rfft(E_right_ref)
    freq = np.fft.rfftfreq(n_steps, d=dt)

    valid = np.abs(E_inc_fft) > 1e-10 * np.max(np.abs(E_inc_fft))
    R_fdtd = np.zeros_like(freq, dtype=complex)
    T_fdtd = np.zeros_like(freq, dtype=complex)
    R_fdtd[valid] = E_ref_fft[valid] / E_inc_fft[valid]
    T_fdtd[valid] = E_trans_fft[valid] / E_inc_fft[valid]

    # Analytical R(f), T(f) via Transfer Matrix
    f_bw = 1.0 / (2.0 * np.pi * sigma)
    band = (freq > 0.1) & (freq < 1.5 * f_bw)

    Phi = panel_transfer_matrix(freq[band], panel_d, eps_r_panel, sigma_panel)
    R_anal, T_anal = RT_from_transfer_matrix(Phi)

    assert np.corrcoef(np.abs(R_fdtd[band]), np.abs(R_anal))[0, 1] > 0.99
    assert np.corrcoef(np.abs(T_fdtd[band]), np.abs(T_anal))[0, 1] > 0.99


def test_fdtd_multilayer_panel_reflection():
    L = 4.0
    N = 4001
    x = np.linspace(0, L, N)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')

    pulse_x0 = 0.8
    sigma = 0.06

    # Define 3-layer stack centered at L/2
    layers = [
        {'d': 0.08, 'eps_r': 4.0, 'sigma': 0.0},
        {'d': 0.04, 'eps_r': 1.0, 'sigma': 0.5},
        {'d': 0.08, 'eps_r': 4.0, 'sigma': 0.0},
    ]
    total_d = sum(lay['d'] for lay in layers)
    panel_center = L / 2
    panel_left = panel_center - total_d / 2
    panel_right = panel_center + total_d / 2

    obs_left_idx = np.argmin(np.abs(x - (panel_left - 0.4)))
    obs_right_idx = np.argmin(np.abs(x - (panel_right + 0.4)))

    initial_e = gaussian(x, pulse_x0, sigma)
    initial_h = gaussian(xH, pulse_x0, sigma)

    t_final = 2.5 * L

    # Build spatially-varying eps_r and sigma for the multilayer
    eps_r_arr = np.ones_like(x)
    sig_arr = np.zeros_like(x)
    edge = panel_left
    for lay in layers:
        mask = (x >= edge) & (x < edge + lay['d'])
        eps_r_arr[mask] = lay['eps_r']
        sig_arr[mask] = lay['sigma']
        edge += lay['d']

    # Simulation WITH multilayer panel
    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()
    fdtd.eps_r = eps_r_arr
    fdtd.sig = sig_arr

    n_steps = round(t_final / fdtd.dt)
    E_left_panel = np.zeros(n_steps)
    E_right_panel = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd._step()
        E_left_panel[i] = fdtd.e[obs_left_idx]
        E_right_panel[i] = fdtd.e[obs_right_idx]

    # Reference simulation WITHOUT panel (free space)
    fdtd_ref = FDTD1D(x, boundaries)
    fdtd_ref.load_initial_field(initial_e)
    fdtd_ref.h = initial_h.copy()

    E_left_ref = np.zeros(n_steps)
    E_right_ref = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_ref._step()
        E_left_ref[i] = fdtd_ref.e[obs_left_idx]
        E_right_ref[i] = fdtd_ref.e[obs_right_idx]

    # Extract R(f), T(f) via FFT
    dt = fdtd.dt
    E_ref_fft = np.fft.rfft(E_left_panel - E_left_ref)
    E_trans_fft = np.fft.rfft(E_right_panel)
    E_inc_fft = np.fft.rfft(E_right_ref)
    freq = np.fft.rfftfreq(n_steps, d=dt)

    valid = np.abs(E_inc_fft) > 1e-10 * np.max(np.abs(E_inc_fft))
    R_fdtd = np.zeros_like(freq, dtype=complex)
    T_fdtd = np.zeros_like(freq, dtype=complex)
    R_fdtd[valid] = E_ref_fft[valid] / E_inc_fft[valid]
    T_fdtd[valid] = E_trans_fft[valid] / E_inc_fft[valid]

    # Analytical R(f), T(f) via stack transfer matrix
    f_bw = 1.0 / (2.0 * np.pi * sigma)
    band = (freq > 0.1) & (freq < 1.5 * f_bw)

    Phi = stack_transfer_matrix(freq[band], layers)
    R_anal, T_anal = RT_from_transfer_matrix(Phi)

    assert np.corrcoef(np.abs(R_fdtd[band]), np.abs(R_anal))[0, 1] > 0.90
    assert np.corrcoef(np.abs(T_fdtd[band]), np.abs(T_anal))[0, 1] > 0.90


if __name__ == "__main__":
    pytest.main([__file__])