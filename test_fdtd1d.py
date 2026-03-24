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
    assert np.allclose(h_solved, h_expected, atol=1e-2)




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


def test_fdtd_conductive_medium():

    N = 401
    x = np.linspace(-2, 2, N)
    xH = (x[1:] + x[:-1]) / 2.0

    epsilon = np.ones(N)
    epsilon2 = 4.0
    interface_idx = N // 2
    epsilon[interface_idx:] = epsilon2

    x0 = -1.0
    sig = 0.08

    initial_e = gaussian(x, x0, sig)
    initial_h = -gaussian(xH, x0, sig)

    fdtd = FDTD1D(x, ('mur', 'mur'), epsilon=epsilon, sigma=0.0)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    t_final = 2.0
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()

    eta1 = 1.0 / np.sqrt(1.0)
    eta2 = 1.0 / np.sqrt(epsilon2)
    Gamma = (eta2 - eta1) / (eta2 + eta1)

    left_mask = x < -0.5
    e_left = e_solved[left_mask]
    peak_idx = np.argmax(np.abs(e_left))
    measured_peak = e_left[peak_idx]

    # Reflected pulse must be inverted (negative)
    assert measured_peak < 0

    # Amplitude must match |Gamma| within 5%
    assert abs(measured_peak / Gamma - 1) < 0.05


if __name__ == "__main__":
    pytest.main([__file__])