"""
Tests for the conductive panel reflection/transmission project.
All in normalized units (c=1, eps_0=1, mu_0=1, eta_0=1).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transfer_matrix import (
    panel_transfer_matrix,
    RT_from_transfer_matrix,
    reflection_transmission,
    reflection_transmission_stack,
)
from fdtd_panel import run_fdtd_panel, run_fdtd_reference, compute_RT_fdtd


# ---------- Transfer Matrix Tests ----------

def test_lossless_energy_conservation():
    """For a lossless panel, |R|^2 + |T|^2 = 1."""
    freq = np.linspace(0.1, 10.0, 500)
    R, T = reflection_transmission(freq, d=0.1, eps_r=4.0, sigma=0.0)
    energy = np.abs(R)**2 + np.abs(T)**2
    assert np.allclose(energy, 1.0, atol=1e-10)


def test_half_wave_slab_transparent():
    """A half-wave slab (n*d = lambda/2) should have R=0."""
    eps_r = 4.0
    n = np.sqrt(eps_r)
    f0 = 5.0
    lam = 1.0 / (f0 * n)  # c=1 in normalized units
    d = lam / 2

    R, T = reflection_transmission(np.array([f0]), d=d, eps_r=eps_r, sigma=0.0)
    assert np.abs(R[0]) < 1e-10


def test_lossy_panel_absorbs_energy():
    """For a lossy panel, |R|^2 + |T|^2 < 1."""
    freq = np.linspace(0.1, 10.0, 500)
    R, T = reflection_transmission(freq, d=0.1, eps_r=4.0, sigma=1.0)
    energy = np.abs(R)**2 + np.abs(T)**2
    assert np.all(energy < 1.0)
    assert np.all(energy > 0.0)


def test_vacuum_panel_no_reflection():
    """A vacuum panel (eps_r=1, sigma=0) should have R=0, T=1."""
    freq = np.linspace(0.1, 10.0, 100)
    R, T = reflection_transmission(freq, d=0.1, eps_r=1.0, sigma=0.0)
    assert np.allclose(np.abs(R), 0.0, atol=1e-10)
    assert np.allclose(np.abs(T), 1.0, atol=1e-10)


def test_stack_single_layer_equals_panel():
    """A stack with one layer should equal a single panel."""
    freq = np.linspace(0.1, 10.0, 200)
    d, eps_r, sigma = 0.1, 4.0, 0.5

    R1, T1 = reflection_transmission(freq, d=d, eps_r=eps_r, sigma=sigma)
    R2, T2 = reflection_transmission_stack(freq, [{'d': d, 'eps_r': eps_r, 'sigma': sigma}])

    assert np.allclose(R1, R2, atol=1e-12)
    assert np.allclose(T1, T2, atol=1e-12)


# ---------- FDTD Tests ----------

def test_fdtd_free_space_no_reflection():
    """FDTD with no panel should give R~0."""
    N = 2001
    L = 4.0
    panel_res = run_fdtd_panel(
        N=N, L=L, panel_thickness=0.1, eps_r=1.0, sigma=0.0,
        pulse_sigma=0.06, t_final=2.0 * L,
    )
    ref_res = run_fdtd_reference(
        N=N, L=L, panel_thickness=0.1, pulse_sigma=0.06, t_final=2.0 * L,
    )
    freq, R, T = compute_RT_fdtd(panel_res, ref_res)

    f_bw = 1.0 / (2.0 * np.pi * 0.06)
    mask = (freq > 0.1) & (freq < 2.0 * f_bw)
    assert np.max(np.abs(R[mask])) < 0.05


def test_fdtd_vs_analytical_dielectric():
    """FDTD R,T should match analytical for a lossless dielectric panel."""
    N, L, d, eps_r, sigma, pulse_sigma = 4001, 4.0, 0.2, 4.0, 0.0, 0.06

    panel_res = run_fdtd_panel(
        N=N, L=L, panel_thickness=d, eps_r=eps_r, sigma=sigma,
        pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    ref_res = run_fdtd_reference(
        N=N, L=L, panel_thickness=d, pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    freq, R_fdtd, T_fdtd = compute_RT_fdtd(panel_res, ref_res)

    f_bw = 1.0 / (2.0 * np.pi * pulse_sigma)
    mask = (freq > 0.1) & (freq < 1.5 * f_bw)

    Phi = panel_transfer_matrix(freq[mask], d, eps_r, sigma)
    R_anal, T_anal = RT_from_transfer_matrix(Phi)

    assert np.corrcoef(np.abs(R_fdtd[mask]), np.abs(R_anal))[0, 1] > 0.95
    assert np.corrcoef(np.abs(T_fdtd[mask]), np.abs(T_anal))[0, 1] > 0.95


def test_fdtd_vs_analytical_conductive():
    """FDTD R,T should match analytical for a conductive panel."""
    N, L, d, eps_r, sigma, pulse_sigma = 4001, 4.0, 0.2, 4.0, 0.5, 0.06

    panel_res = run_fdtd_panel(
        N=N, L=L, panel_thickness=d, eps_r=eps_r, sigma=sigma,
        pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    ref_res = run_fdtd_reference(
        N=N, L=L, panel_thickness=d, pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    freq, R_fdtd, T_fdtd = compute_RT_fdtd(panel_res, ref_res)

    f_bw = 1.0 / (2.0 * np.pi * pulse_sigma)
    mask = (freq > 0.1) & (freq < 1.5 * f_bw)

    Phi = panel_transfer_matrix(freq[mask], d, eps_r, sigma)
    R_anal, T_anal = RT_from_transfer_matrix(Phi)

    assert np.corrcoef(np.abs(R_fdtd[mask]), np.abs(R_anal))[0, 1] > 0.90
    assert np.corrcoef(np.abs(T_fdtd[mask]), np.abs(T_anal))[0, 1] > 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
