"""
Compare R(f) and T(f) between Transfer Matrix Method (analytical)
and FDTD simulation (numerical) for conductive/dielectric panels.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fdtd1d import FDTD1D
from panel_utils import (
    reflection_transmission,
    reflection_transmission_stack,
    run_panel_experiment
)


def compare_single_panel(d=0.1, eps_r=4.0, sigma=0.5, N=4001, L=4.0,
                         pulse_sigma=0.06, save_fig=None):
    """Run FDTD and analytical comparison for a single conductive panel."""
    print(f"Panel: d={d}, eps_r={eps_r}, sigma={sigma}")

    res = run_panel_experiment(
        N=N, L=L, panel_d=d, eps_r=eps_r, sigma=sigma,
        pulse_sigma=pulse_sigma,
    )
    freq_fdtd, R_fdtd, T_fdtd = res['freq'], res['R'], res['T']

    f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
    R_anal, T_anal = reflection_transmission(f_anal, d, eps_r, sigma)

    f_bw = 1.0 / (2.0 * np.pi * pulse_sigma)
    f_max = min(3.0 * f_bw, freq_fdtd.max())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Conductive Panel: d={d}, $\\varepsilon_r$={eps_r}, $\\sigma$={sigma}',
                 fontsize=14)

    mask = (freq_fdtd > 0.01) & (freq_fdtd < f_max)
    mask_a = f_anal < f_max

    axes[0, 0].plot(freq_fdtd[mask], np.abs(R_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    axes[0, 0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analytical')
    axes[0, 0].set_xlabel('Frequency'); axes[0, 0].set_ylabel('|R|')
    axes[0, 0].set_title('Reflection |R(f)|'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freq_fdtd[mask], np.abs(T_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    axes[0, 1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analytical')
    axes[0, 1].set_xlabel('Frequency'); axes[0, 1].set_ylabel('|T|')
    axes[0, 1].set_title('Transmission |T(f)|'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(freq_fdtd[mask], np.abs(R_fdtd[mask])**2 + np.abs(T_fdtd[mask])**2,
                    'b-', alpha=0.7, label='FDTD')
    axes[1, 0].plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
                    'r--', lw=2, label='Analytical')
    axes[1, 0].set_xlabel('Frequency'); axes[1, 0].set_ylabel('$|R|^2 + |T|^2$')
    axes[1, 0].set_title('Energy conservation'); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_ylim(0, 1.1)

    axes[1, 1].plot(np.arange(len(res['E_left_panel'])) * res['dt'],
                    res['E_left_panel'], 'b-', alpha=0.7, label='E left (inc+ref)')
    axes[1, 1].plot(np.arange(len(res['E_right_panel'])) * res['dt'],
                    res['E_right_panel'], 'g-', alpha=0.7, label='E right (trans)')
    axes[1, 1].plot(np.arange(len(res['E_left_ref'])) * res['dt'],
                    res['E_left_ref'], 'r--', alpha=0.5, label='E left (ref)')
    axes[1, 1].set_xlabel('Time'); axes[1, 1].set_ylabel('E field')
    axes[1, 1].set_title('Time-domain signals'); axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_fig}")
    plt.show()


def compare_multilayer(layers=None, N=4001, L=6.0, pulse_sigma=0.04, save_fig=None):
    """Run FDTD and analytical comparison for a multi-layer panel."""
    if layers is None:
        layers = [
            {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
            {'d': 0.08, 'eps_r': 6.0, 'sigma': 1.0},
            {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
        ]

    print(f"Multi-layer: {len(layers)} layers")
    res = run_panel_experiment(N=N, L=L, layers=layers, pulse_sigma=pulse_sigma)
    freq_fdtd, R_fdtd, T_fdtd = res['freq'], res['R'], res['T']

    f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
    R_anal, T_anal = reflection_transmission_stack(f_anal, layers)

    f_bw = 1.0 / (2.0 * np.pi * pulse_sigma)
    f_max = min(3.0 * f_bw, freq_fdtd.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layer_str = ', '.join([f"(d={l['d']}, er={l.get('eps_r',1)}, s={l.get('sigma',0)})"
                           for l in layers])
    fig.suptitle(f'Multi-layer: {layer_str}', fontsize=11)

    mask = (freq_fdtd > 0.01) & (freq_fdtd < f_max)
    mask_a = f_anal < f_max

    axes[0].plot(freq_fdtd[mask], np.abs(R_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    axes[0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analytical')
    axes[0].set_xlabel('Frequency'); axes[0].set_ylabel('|R|')
    axes[0].set_title('|R(f)|'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(freq_fdtd[mask], np.abs(T_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    axes[1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analytical')
    axes[1].set_xlabel('Frequency'); axes[1].set_ylabel('|T|')
    axes[1].set_title('|T(f)|'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq_fdtd[mask], np.abs(R_fdtd[mask])**2 + np.abs(T_fdtd[mask])**2,
                 'b-', alpha=0.7, label='FDTD')
    axes[2].plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
                 'r--', lw=2, label='Analytical')
    axes[2].set_xlabel('Frequency'); axes[2].set_ylabel('$|R|^2 + |T|^2$')
    axes[2].set_title('Energy check'); axes[2].legend()
    axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_fig}")
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Case 1: Slightly conductive panel")
    print("=" * 60)
    compare_single_panel(d=0.1, eps_r=4.0, sigma=0.5)

    print("\n" + "=" * 60)
    print("Case 2: More conductive panel")
    print("=" * 60)
    compare_single_panel(d=0.1, eps_r=1.0, sigma=2.0)

    print("\n" + "=" * 60)
    print("Case 3: Pure dielectric (no loss)")
    print("=" * 60)
    compare_single_panel(d=0.1, eps_r=4.0, sigma=0.0)

    print("\n" + "=" * 60)
    print("Case 4: Multi-layer panel (bonus)")
    print("=" * 60)
    compare_multilayer()
