import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pytest
from fdtd1d import FDTD1D, gaussian, permitividad_ag, transmitancia_slab, C 

# -------------------------------------------------------------
# Tus Tests Originales (Intactos)
# -------------------------------------------------------------
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
    
def test_analytical_transmittance_matches_paper():
    E_extraido = np.array([1.0, 2.0, 3.0, 3.2, 3.4, 3.6, 3.7, 3.8, 3.85, 3.9, 4.0, 4.1, 4.2, 4.4, 4.7, 5.0])
    T_extraido = np.array([0.0, 0.0, 0.002, 0.005, 0.015, 0.04, 0.08, 0.145, 0.16, 0.145, 0.08, 0.035, 0.015, 0.003, 0.0, 0.0])

    interp_spline = CubicSpline(E_extraido, T_extraido)
    E_denso = np.linspace(1.0, 5.0, 500)
    T_interpolado = np.clip(interp_spline(E_denso), 0, None)

    T_teorico = transmitancia_slab(E_denso, grosor_nm=100.0)
    assert np.corrcoef(T_teorico, T_interpolado)[0, 1] > 0.99


# -------------------------------------------------------------
# NUEVO TEST: PROPOSAL 4 (Material Dispersivo y Transmitancia)
# -------------------------------------------------------------
import numpy as np

def test_fdtd_dispersive_slab_transmittance():
    """
    Simula la placa de Ag en FDTD con un pulso espacial ultracorto,
    calcula su FFT y valida el resultado de transmitancia numérica con el analítico
    propuesto en la Figura 2 del artículo. (Versión con unidades normalizadas C=1).
    """
    # --- 1. Constantes Normalizadas y de Conversión ---
    eps0 = 1.0
    mu0 = 1.0
    Z0 = np.sqrt(mu0 / eps0)  # Impedancia normalizada (es 1.0)
    
    C_fisica = 299792458.0  # Usada SOLO para conversiones con el mundo exterior
    
    # Malla en el espacio físico un poco más fina para alta frecuencia
    dx = 1e-9  # 1 nm
    xMin = -400e-9
    xMax =  400e-9
    x = np.arange(xMin, xMax, dx)
    xH = (x[1:] + x[:-1]) / 2.0
    
    dt = dx / C  # En FDTD normalizado, dt es numéricamente igual a dx
    
    # --- 2. Pulso Gaussiano espacial ultracorto ---
    # (sigma=8 nm) para inyectar una "banda ancha" que cubra de 0 a 5 eV
    sigma = 8e-9
    x0_pulse = -200e-9
    initial_e = np.exp(-0.5 * ((x - x0_pulse) / sigma) ** 2)
    
    # H se evalúa espacialmente desplazado dx/2 y temporalmente dt/2 
    initial_h = np.exp(-0.5 * ((xH - x0_pulse - C * dt / 2.0) / sigma) ** 2) / Z0
    
    # --- 3. Datos Ag de la Tabla I escalados a la normalización ---
    eV_to_rads = 1.519267e15
    # Factor clave: Convierte (rad/s) a (rad/m) para cuadrar con C=1
    factor_conversion = eV_to_rads / C_fisica 
    
    cp_eV = np.array([
        0.5987 + 4195j, -0.2211 + 0.2680j, -4.240 + 732.4j, 
        0.6391 - 0.07186j, 1.806 + 4.563j, 1.443 - 82.19j
    ])
    ap_eV = np.array([
        -0.02502 - 0.008626j, -0.2021 - 0.9407j, -14.67 - 1.338j, 
        -0.2997 - 4.034j, -1.896 - 4.808j, -9.396 - 6.477j
    ])
    
    cp_norm = cp_eV * factor_conversion
    ap_norm = ap_eV * factor_conversion
    
    # Sensor detrás de la placa
    x_sensor = 200e-9
    sensor_idx = np.argmin(np.abs(x - x_sensor))
    
    # Tiempo de simulación: 40 fs convertidos a "distancia FDTD"
    t_final_fisico = 40.0e-15 
    t_final = t_final_fisico * C_fisica
    
    # --- 4. SIMULACIÓN DE REFERENCIA (VACÍO) ---
    fdtd_ref = FDTD1D(x, boundaries=('mur', 'mur'), C=C, eps0=eps0, mu0=mu0)
    fdtd_ref.load_initial_field(initial_e)
    fdtd_ref.h = initial_h.copy() 
    
    E_ref_history = []
    while fdtd_ref.t < t_final:
        fdtd_ref._step()
        E_ref_history.append(fdtd_ref.e[sensor_idx])
        
    # --- 5. SIMULACIÓN CON SLAB DE AG (100 nm) ---
    fdtd_ag = FDTD1D(x, boundaries=('mur', 'mur'), C=C, eps0=eps0, mu0=mu0)
    fdtd_ag.load_initial_field(initial_e)
    fdtd_ag.h = initial_h.copy()
    
    slab_mask = (x >= 0) & (x <= 100e-9)
    fdtd_ag.add_dispersive_material(slab_mask, cp_norm, ap_norm)
    
    E_trans_history = []
    while fdtd_ag.t < t_final:
        fdtd_ag._step()
        E_trans_history.append(fdtd_ag.e[sensor_idx])
        
    # --- 6. ANÁLISIS EN FRECUENCIA (FFT) ---
    # dt ahora es espacial, así que freqs está en ciclos/metro
    freqs = np.fft.fftfreq(len(E_ref_history), d=dt)
    w_rads_norm = 2 * np.pi * freqs  # Frecuencia angular espacial (rad/m)
    
    # Recuperamos eV: (rad/m * m/s) / (rad/s/eV) = eV
    E_eV_array = (w_rads_norm * C_fisica) / eV_to_rads
    
    E_ref_fft = np.fft.fft(E_ref_history)
    E_trans_fft = np.fft.fft(E_trans_history)
    
    # Espectro de transmisión numérico
    T_num = np.abs(E_trans_fft)**2 / np.abs(E_ref_fft)**2
    
    # Nos centramos solo en la banda de análisis propuesta
    valid_idx = (E_eV_array >= 1.5) & (E_eV_array <= 4.5)
    E_valid = E_eV_array[valid_idx]
    T_valid_num = T_num[valid_idx]
    
    # --- 7. COMPARACIÓN CON RESULTADO ANALÍTICO ---
    # (Asegúrate de que transmitancia_slab esté disponible en tu archivo test)
    T_teo = transmitancia_slab(E_valid, grosor_nm=100.0)
    
    # La correlación validará el espectro banda ancha normalizado
    assert np.corrcoef(T_valid_num, T_teo)[0, 1] > 0.99
if __name__ == "__main__":
    pytest.main([__file__])