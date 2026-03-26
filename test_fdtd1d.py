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
    
def test_fdtd_PMC_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('PMC', 'PMC')
    
    x0 = 0.0
    sigma = 0.05
    # H is stored on staggered cells of size N-1
    x_h = x[:-1]
    initial_h = gaussian(x_h, x0, sigma)

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_h)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    h_solved = fdtd.get_h()
    e_solved = fdtd.get_e()

    h_expected = - initial_h
    e_expected = np.zeros_like(e_solved)
    
    assert np.corrcoef(h_solved, h_expected)[0,1] > 0.98
    assert np.allclose(e_solved, e_expected, atol=0.01)


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

def test_fdtd_ABC_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('PEC', 'ABC')
    
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

    e_expected = np.zeros_like(e_solved)
    h_expected = np.zeros_like(h_solved)
    
    assert np.allclose(e_solved, e_expected, atol=0.01)
    assert np.allclose(h_solved, h_expected, atol=0.01)



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
    
def test_reflection():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    
    L = xMax - xMin
    t_final = 2.0*L/C
    
    epsilon_r = 4.0
    
    epsilon = np.ones_like(x)
    epsilon[x > L] = epsilon_r
    
    n1 = 1.0
    n2 = np.sqrt(epsilon_r)
    
    R = (n1-n2)/(n1+n2)
    T = 2.0*n1/(n1+n2)
    
    fdtd = FDTD1D(x, epsilon=epsilon)
    fdtd.load_initial_field(initial_e)
    fdtd.run_until(t_final)
    
    e_trans_solved, e_reflex_solved = fdtd.get_e()
    
    e_reflex_expected = initial_e*R
    e_trans_expected = initial_e*T
    
    assert np.corrcoef(e_reflex_solved, e_reflex_expected)[0,1] > 0.99
    assert np.corrcoef(e_trans_solved, e_trans_expected)[0,1] > 0.99

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
    h_expected = t_func_H * np.cos(k * (xH - x0)) 

    assert np.allclose(e_solved, e_expected, atol=1e-2)
    assert np.allclose(h_solved, h_expected, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
