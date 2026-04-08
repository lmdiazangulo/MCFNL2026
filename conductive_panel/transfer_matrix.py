"""
Analytical reflection and transmission coefficients using the Transfer Matrix Method.

All computations use NORMALIZED units (c=1, eps_0=1, mu_0=1, eta_0=1),
consistent with the FDTD1D class in the repository.

For a homogeneous panel of thickness d with complex permittivity
    eps_c = eps_r - j * sigma / omega
the transfer (ABCD) matrix is (notes eq. 1.19):
    Phi = [[cosh(gamma*d),  eta*sinh(gamma*d)],
           [sinh(gamma*d)/eta,  cosh(gamma*d)]]
where
    gamma = j * omega * sqrt(mu_r * eps_c)
    eta   = sqrt(mu_r / eps_c)

T and R are then obtained from (notes eqs. 1.24, 1.25):
    T = 2 / (Phi11 + Phi12 + Phi21 + Phi22)
    R = (Phi11 + Phi12 - Phi21 - Phi22) / (Phi11 + Phi12 + Phi21 + Phi22)

(with eta_0 = 1 in normalized units, the formulas simplify).

References:
    [1] Class notes, section 1.4.3
    [2] Orfanidis, "Electromagnetic Waves and Antennas", Chapter 4-5
"""

import numpy as np


def panel_transfer_matrix(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    """
    Compute the 2x2 ABCD transfer matrix for a single homogeneous panel.

    Parameters
    ----------
    freq : array_like
        Frequencies (normalized units).
    d : float
        Panel thickness (normalized units).
    eps_r : float
        Relative permittivity.
    sigma : float
        Conductivity (normalized units).
    mu_r : float
        Relative permeability.

    Returns
    -------
    Phi : ndarray, shape (len(freq), 2, 2)
    """
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
    """
    Compute the transfer matrix for a stack of panels.

    Parameters
    ----------
    freq : array_like
        Frequencies (normalized units).
    layers : list of dict
        Each dict has keys: 'd', 'eps_r', 'sigma', 'mu_r' (optional).

    Returns
    -------
    Phi_total : ndarray, shape (len(freq), 2, 2)
    """
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
    """
    Compute reflection (R) and transmission (T) from the ABCD matrix.

    In normalized units eta_0 = 1, so the formulas simplify.

    Parameters
    ----------
    Phi : ndarray, shape (N, 2, 2)

    Returns
    -------
    R, T : complex ndarrays, shape (N,)
    """
    A = Phi[:, 0, 0]
    B = Phi[:, 0, 1]
    C = Phi[:, 1, 0]
    D = Phi[:, 1, 1]

    denom = A + B + C + D

    T = 2.0 / denom
    R = (A + B - C - D) / denom

    return R, T


def reflection_transmission(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    """Compute R(f) and T(f) for a single panel."""
    Phi = panel_transfer_matrix(freq, d, eps_r, sigma, mu_r)
    return RT_from_transfer_matrix(Phi)


def reflection_transmission_stack(freq, layers):
    """Compute R(f) and T(f) for a stack of panels."""
    Phi = stack_transfer_matrix(freq, layers)
    return RT_from_transfer_matrix(Phi)
