import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd1d import FDTD1D, C, gaussian




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

def test_fdtd_sinus_perturbation():
    xMax =  1.0
    xMin = -1.0
    L = xMax - xMin
    x  = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    x_o = 0.0

    def my_pert(t):
        return np.sin(2*np.pi*2*C/L*t)

    fdtd = FDTD1D(x, boundaries, x_o, pert=lambda t: my_pert(t), pert_dir=+1)
    fdtd.load_initial_field(np.zeros_like(x))

    t_final = L / (2 * C)
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = my_pert(t_final - (x  - x_o) / C)
    h_expected = my_pert(t_final + fdtd.dt/2 - (xH - x_o) / C) 
    
    e_expected[x < x_o] = 0.0
    h_expected[xH < x_o] = 0.0

    np.testing.assert_allclose(e_solved, e_expected, atol=1e-3)
    np.testing.assert_allclose(h_solved, h_expected, atol=1e-3)

def test_fdtd_gauss_perturbation():
    xMax =  1.0
    xMin = -1.0
    L = xMax - xMin
    x  = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    x_o = 0.0

    def my_pert(t):
        return gaussian(t, 0.4, 0.1)

    fdtd = FDTD1D(x, boundaries, x_o, pert=lambda t: my_pert(t), pert_dir=+1)
    fdtd.load_initial_field(np.zeros_like(x))

    t_final = L / (2 * C)
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = my_pert(t_final - (x  - x_o) / C)
    h_expected = my_pert(t_final + fdtd.dt/2 - (xH - x_o) / C) 
    
    e_expected[x < x_o] = 0.0
    h_expected[xH < x_o] = 0.0

    np.testing.assert_allclose(e_solved, e_expected, atol=1e-3)
    np.testing.assert_allclose(h_solved, h_expected, atol=1e-3)

def test_fdtd_tf_sf_right():
    xMax = 1.0
    xMin = -1.0
    x = np.linspace(xMin, xMax, 401)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    x_o = -0.5

    def my_pert(t):
        return gaussian(t, 0.2, 0.05)

    fdtd = FDTD1D(x, boundaries, x_o, pert=my_pert, pert_dir=+1)
    fdtd.load_initial_field(np.zeros_like(x))

    t_final = 0.5
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    sf_region_e = e_solved[x < x_o]
    sf_region_h = h_solved[xH < x_o]

    assert np.allclose(sf_region_e, 0.0, atol=1e-10), "Fuga detectada en E (región SF)"
    assert np.allclose(sf_region_h, 0.0, atol=1e-10), "Fuga detectada en H (región SF)"

    e_expected = np.zeros_like(e_solved)
    mask_tf = x >= x_o
    e_expected[mask_tf] = my_pert(t_final - (x[mask_tf] - x_o) / C)

    assert np.allclose(e_solved[mask_tf], e_expected[mask_tf], atol=1e-3)


def test_fdtd_tf_sf_left():
    xMax = 1.0
    xMin = -1.0
    x = np.linspace(xMin, xMax, 401)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    x_o = 0.5

    def my_pert(t):
        return gaussian(t, 0.2, 0.05)

    fdtd = FDTD1D(x, boundaries, x_o, pert=my_pert, pert_dir=-1)
    fdtd.load_initial_field(np.zeros_like(x))

    t_final = 0.5
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    sf_region_e = e_solved[x > x_o]
    sf_region_h = h_solved[xH > x_o]

    assert np.allclose(sf_region_e, 0.0, atol=1e-10), "Fuga detectada en E (región SF)"
    assert np.allclose(sf_region_h, 0.0, atol=1e-10), "Fuga detectada en H (región SF)"

    e_expected = np.zeros_like(e_solved)
    mask_tf = x <= x_o
    e_expected[mask_tf] = my_pert(t_final - (x_o - x[mask_tf]) / C)

    assert np.allclose(e_solved[mask_tf], e_expected[mask_tf], atol=1e-3)


def test_fdtd_soft_source():
    xMax = 1.0
    xMin = -1.0
    x = np.linspace(xMin, xMax, 401)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    x_o = 0.0

    def my_pert(t):
        return gaussian(t, 0.2, 0.05)

    fdtd = FDTD1D(x, boundaries, x_o, pert=my_pert, pert_dir=None)
    fdtd.load_initial_field(np.zeros_like(x))

    t_final = 0.5
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    # Soft source should inject energy into both E and H at x_o without a directional bias
    assert np.any(np.abs(e_solved) > 0.0)
    assert np.any(np.abs(h_solved) > 0.0)

    # Expected soft source waveform for both left and right-going pulses
    e_expected = np.zeros_like(e_solved)
    mask_right = x >= x_o
    mask_left = x <= x_o
    e_expected[mask_right] = 0.5 * my_pert(t_final - (x[mask_right] - x_o) / C)
    e_expected[mask_left] = 0.5 * my_pert(t_final - (x_o - x[mask_left]) / C)
    e_expected[np.argmin(np.abs(x - x_o))] = my_pert(t_final)

    mask_near = np.abs(x - x_o) < 0.3
    assert np.max(e_solved[mask_near]) > 0.1
    assert np.max(np.abs(h_solved[np.abs(xH - x_o) < 0.3])) > 0.1
    assert np.allclose(e_solved, e_expected, atol=0.15)


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

if __name__ == "__main__":
    pytest.main([__file__])