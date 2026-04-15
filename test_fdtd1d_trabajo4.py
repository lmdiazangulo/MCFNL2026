import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import pytest
from fdtd1d import (FDTD1D, gaussian, permitividad_ag, transmitancia_slab, C, 
                    get_ag_poles_norm, reflectancia_slab, absorbancia_slab, extract_spectrum)

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
    e_expected = 0.5 * gaussian(x, -t_final*C, sigma) + 0.5 * gaussian(x, t_final*C, sigma)

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

def test_fdtd_total_spread_field():
    xMax = 1
    xMin = -1
    L = xMax - xMin
    x = np.linspace(xMin,xMax,201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur','mur') 
    x_o = 0
    initial_e = np.zeros_like(x)

    def my_pert(t):
        return np.sin(t*np.pi)

    fdtd = FDTD1D(x, boundaries, x_o, pert=lambda t: my_pert(t))
    fdtd.load_initial_field(initial_e)
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = my_pert(x - L)
    h_expected = -my_pert(xH - L)
    
    assert np.corrcoef(e_solved, e_expected)[0,1] > -0.6
    assert np.corrcoef(h_solved, h_expected)[0,1] > 0.8

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

def test_fdtd_permitivity():
    E_exp = np.array([0.15, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 
                      3.5, 3.8, 3.9, 4.0, 4.2, 4.5, 4.8, 5.0])
    
    n_exp = np.array([8.0, 4.0, 1.8, 0.6, 0.35, 0.15, 0.12, 0.13, 0.15, 
                      0.2, 0.4, 0.8, 1.2, 1.4, 1.35, 1.25, 1.2])
    
    kappa_exp = np.array([60.0, 35.0, 20.0, 12.0, 9.0, 6.0, 4.0, 3.0, 2.0, 
                          1.5, 0.8, 0.6, 0.7, 1.0, 1.2, 1.3, 1.3])

    interp_n = CubicSpline(E_exp, n_exp)
    interp_kappa = CubicSpline(E_exp, kappa_exp)

    E_dense = np.linspace(E_exp.min(), E_exp.max(), 500)
    n_exp_smooth = interp_n(E_dense)
    kappa_exp_smooth = interp_kappa(E_dense)

    eps_w = permitividad_ag(E_dense)
    n_complex = np.sqrt(np.conj(eps_w))
    
    n_teo = n_complex.real
    kappa_teo = n_complex.imag
    
    assert np.corrcoef(n_teo, n_exp_smooth)[0,1] > 0.95
    assert np.corrcoef(kappa_teo, kappa_exp_smooth)[0,1] > 0.95
    
def test_fdtd_dispersive_slab_transmittance():
    E_paper = np.array([1.0, 2.0, 3.0, 3.2, 3.4, 3.6, 3.7, 3.8, 3.85, 3.9, 4.0, 4.1, 4.2, 4.4, 4.7, 5.0])
    T_paper = np.array([0.0, 0.0, 0.002, 0.005, 0.015, 0.04, 0.08, 0.145, 0.16, 0.145, 0.08, 0.035, 0.015, 0.003, 0.0, 0.0])
    
    dx, dt = 1e-9, 1e-9
    x = np.arange(-400e-9, 400e-9, dx)
    xH = (x[1:] + x[:-1]) / 2.0

    initial_e = np.exp(-0.5 * ((x - (-200e-9)) / 8e-9) ** 2)
    initial_h = np.exp(-0.5 * ((xH - (-200e-9) - dt / 2.0) / 8e-9) ** 2)
    
    cp_norm, ap_norm = get_ag_poles_norm()
    sensor_idx = np.argmin(np.abs(x - 200e-9)) 
    t_final = 60.0e-15 * 299792458.0
    
    fdtd_ref = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd_ref.load_initial_field(initial_e); fdtd_ref.h = initial_h.copy()
    ref_history = [fdtd_ref.e[sensor_idx] for _ in range(int(t_final/dt)) if not fdtd_ref._step()]
        
    fdtd_ag = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd_ag.load_initial_field(initial_e); fdtd_ag.h = initial_h.copy()
    fdtd_ag.add_dispersive_material((x >= 0) & (x <= 100e-9), cp_norm, ap_norm)
    trans_history = [fdtd_ag.e[sensor_idx] for _ in range(int(t_final/dt)) if not fdtd_ag._step()]

    E_eV, E_ref_fft = extract_spectrum(ref_history, dt)
    _, E_trans_fft = extract_spectrum(trans_history, dt)
    T_num = np.abs(E_trans_fft)**2 / (np.abs(E_ref_fft)**2 + 1e-12)
    
    valid = (E_eV >= 1.0) & (E_eV <= 5.0)
    E_sim = E_eV[valid]
    T_sim = T_num[valid]
    
    f_interp = interp1d(E_paper, T_paper, kind='cubic', fill_value="extrapolate")
    T_paper_interp = f_interp(E_sim)
    
    assert np.corrcoef(T_sim, T_paper_interp)[0, 1] > 0.99
    
def test_fdtd_dispersive_slab_absorcion():
    dx, dt = 1e-9, 1e-9
    x = np.arange(-500e-9, 500e-9, dx)
    xH = (x[1:] + x[:-1]) / 2.0
    C_fisica = 299792458.0
    
    refl_sensor_idx = np.argmin(np.abs(x - (-150e-9))) 
    trans_sensor_idx = np.argmin(np.abs(x - 250e-9))
    
    initial_e = np.exp(-0.5 * ((x - (-350e-9)) / 15e-9) ** 2)
    initial_h = np.exp(-0.5 * ((xH - (-350e-9) - dt / 2.0) / 15e-9) ** 2)
    
    cp_norm, ap_norm = get_ag_poles_norm()
    t_final = 150.0e-15 * C_fisica

    fdtd_ref = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd_ref.load_initial_field(initial_e); fdtd_ref.h = initial_h.copy()
    
    ref_trans_history = []
    ref_refl_history = []
    
    while fdtd_ref.t < t_final:
        fdtd_ref._step()
        ref_trans_history.append(fdtd_ref.e[trans_sensor_idx])
        ref_refl_history.append(fdtd_ref.e[refl_sensor_idx])
        
    fdtd_mat = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd_mat.load_initial_field(initial_e); fdtd_mat.h = initial_h.copy()
    fdtd_mat.add_dispersive_material((x >= 0) & (x <= 100e-9), cp_norm, ap_norm)
    
    mat_trans_history = []
    mat_refl_history = []
    
    while fdtd_mat.t < t_final:
        fdtd_mat._step()
        mat_trans_history.append(fdtd_mat.e[trans_sensor_idx])
        mat_refl_history.append(fdtd_mat.e[refl_sensor_idx])
    
    E_eV, FFT_inc = extract_spectrum(ref_trans_history, dt) 
    _, FFT_ref_at_refl = extract_spectrum(ref_refl_history, dt) 
    _, FFT_trans = extract_spectrum(mat_trans_history, dt)
    _, FFT_refl_total = extract_spectrum(mat_refl_history, dt) 
    
    FFT_refl_only = FFT_refl_total - FFT_ref_at_refl
    
    T = np.abs(FFT_trans)**2 / (np.abs(FFT_inc)**2 + 1e-12)
    R = np.abs(FFT_refl_only)**2 / (np.abs(FFT_inc)**2 + 1e-12)
    
    T_teo = transmitancia_slab(E_eV, grosor_nm=100.0)
    R_teo = reflectancia_slab(E_eV, grosor_nm=100.0)
    
    A = 1.0 - T - R
    A_teo = absorbancia_slab(E_eV, grosor_nm=100.0)

    valid = (E_eV >= 1.0) & (E_eV <= 5.0)
    
    assert np.corrcoef(A[valid], A_teo[valid])[0, 1] > 0.99


if __name__ == "__main__":
    pytest.main([__file__])