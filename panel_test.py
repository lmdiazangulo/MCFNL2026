import numpy as np
import pytest

from fdtd1d import FDTD1D, gaussian
from panel_utils import (
    panel_transfer_matrix,
    stack_transfer_matrix,
    RT_from_transfer_matrix,
    reflection_transmission,
    reflection_transmission_stack,
    run_panel_experiment
)


def test_set_panel_assigns_eps_r():
    x = np.linspace(0, 4, 401)
    fdtd = FDTD1D(x)
    fdtd.set_panel(center=2.0, d=0.5, eps_r=4.0, sigma=0.0)

    inside = (x >= 1.75) & (x <= 2.25)
    assert np.all(fdtd.eps_r[inside] == 4.0)
    assert np.all(fdtd.eps_r[~inside] == 1.0)


def test_set_panel_assigns_sigma():
    x = np.linspace(0, 4, 401)
    fdtd = FDTD1D(x)
    fdtd.set_panel(center=2.0, d=0.5, eps_r=1.0, sigma=2.0)

    inside = (x >= 1.75) & (x <= 2.25)
    assert np.all(fdtd.sig[inside] == 2.0)
    assert np.all(fdtd.sig[~inside] == 0.0)


def test_set_multilayer_assigns_properties():
    x = np.linspace(0, 4, 4001)
    layers = [
        {'d': 0.1, 'eps_r': 2.0, 'sigma': 0.0},
        {'d': 0.2, 'eps_r': 6.0, 'sigma': 1.0},
        {'d': 0.1, 'eps_r': 2.0, 'sigma': 0.0},
    ]
    fdtd = FDTD1D(x)
    fdtd.set_multilayer(center=2.0, layers=layers)

    mid_layer2 = np.argmin(np.abs(x - 2.0))
    assert fdtd.eps_r[mid_layer2] == 6.0
    assert fdtd.sig[mid_layer2] == 1.0

    mid_layer1 = np.argmin(np.abs(x - 1.85))
    assert fdtd.eps_r[mid_layer1] == 2.0
    assert fdtd.sig[mid_layer1] == 0.0

    outside = np.argmin(np.abs(x - 0.5))
    assert fdtd.eps_r[outside] == 1.0
    assert fdtd.sig[outside] == 0.0


def test_vacuum_panel_no_reflection():
    freq = np.linspace(0.1, 5.0, 200)
    R, T = reflection_transmission(freq, d=0.2, eps_r=1.0, sigma=0.0)
    assert np.allclose(np.abs(R), 0.0, atol=1e-12)
    assert np.allclose(np.abs(T), 1.0, atol=1e-12)


def test_lossless_energy_conservation():
    freq = np.linspace(0.1, 10.0, 500)
    R, T = reflection_transmission(freq, d=0.2, eps_r=4.0, sigma=0.0)
    energy = np.abs(R)**2 + np.abs(T)**2
    assert np.allclose(energy, 1.0, atol=1e-10)


def test_lossy_panel_absorbs_energy():
    freq = np.linspace(0.1, 10.0, 500)
    R, T = reflection_transmission(freq, d=0.2, eps_r=4.0, sigma=1.0)
    energy = np.abs(R)**2 + np.abs(T)**2
    assert np.all(energy < 1.0 + 1e-10)
    assert np.mean(energy) < 0.99


def test_stack_single_layer_matches_panel():
    freq = np.linspace(0.1, 8.0, 300)
    R_panel, T_panel = reflection_transmission(freq, d=0.2, eps_r=4.0, sigma=0.5)
    layers = [{'d': 0.2, 'eps_r': 4.0, 'sigma': 0.5}]
    R_stack, T_stack = reflection_transmission_stack(freq, layers)
    assert np.allclose(np.abs(R_panel), np.abs(R_stack), atol=1e-12)
    assert np.allclose(np.abs(T_panel), np.abs(T_stack), atol=1e-12)


# ── Test de Integracion

def test_fdtd_single_panel_vs_analytical():
    res = run_panel_experiment(
        N=4001, L=4.0, panel_d=0.2,
        eps_r=4.0, sigma=0.5, pulse_sigma=0.06,
    )
    freq = res['freq']
    R_fdtd, T_fdtd = res['R'], res['T']

    f_bw = 1.0 / (2.0 * np.pi * 0.06)
    band = (freq > 0.1) & (freq < 1.5 * f_bw)

    R_anal, T_anal = reflection_transmission(freq[band], d=0.2, eps_r=4.0, sigma=0.5)

    assert np.corrcoef(np.abs(R_fdtd[band]), np.abs(R_anal))[0, 1] > 0.99
    assert np.corrcoef(np.abs(T_fdtd[band]), np.abs(T_anal))[0, 1] > 0.99


def test_fdtd_multilayer_vs_analytical():
    layers = [
        {'d': 0.08, 'eps_r': 4.0, 'sigma': 0.0},
        {'d': 0.04, 'eps_r': 1.0, 'sigma': 0.5},
        {'d': 0.08, 'eps_r': 4.0, 'sigma': 0.0},
    ]
    res = run_panel_experiment(
        N=4001, L=4.0, layers=layers, pulse_sigma=0.06,
    )
    freq = res['freq']
    R_fdtd, T_fdtd = res['R'], res['T']

    f_bw = 1.0 / (2.0 * np.pi * 0.06)
    band = (freq > 0.1) & (freq < 1.5 * f_bw)

    R_anal, T_anal = reflection_transmission_stack(freq[band], layers)

    assert np.corrcoef(np.abs(R_fdtd[band]), np.abs(R_anal))[0, 1] > 0.99
    assert np.corrcoef(np.abs(T_fdtd[band]), np.abs(T_anal))[0, 1] > 0.99


def test_fdtd_free_space_no_reflection():
    res = run_panel_experiment(
        N=4001, L=4.0, panel_d=0.2,
        eps_r=1.0, sigma=0.0, pulse_sigma=0.06,
    )
    freq = res['freq']
    f_bw = 1.0 / (2.0 * np.pi * 0.06)
    band = (freq > 0.1) & (freq < 1.5 * f_bw)
    assert np.max(np.abs(res['R'][band])) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
