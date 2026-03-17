"""Simple 1D wave propagation helper.

This module provides a minimal implementation of the 1D wave equation
solution in the form of d'Alembert's formula for constant wave speed.

The goal is to satisfy the test in `test_fdtd1d.py`.
"""

import numpy as np


class FDTD1D:
    """Minimal 1D wave solver interface."""

    def __init__(self, x, c: float = 1.0):
        """Initialize solver with a spatial grid.

        Args:
            x: 1D array of spatial positions.
            c: wave speed (default 1.0).
        """
        self.x = np.asarray(x)
        self.c = float(c)
        self._e0 = None
        self._t = 0.0

    def load_initial_field(self, initial_e):
        """Load the initial field at t=0."""
        e = np.asarray(initial_e)
        if e.shape != self.x.shape:
            raise ValueError("initial_e must have the same shape as x")
        self._e0 = e.copy()
        self._t = 0.0

    def run_until(self, t_final: float):
        """Advance the solution to time t_final.

        This implementation uses the analytical d'Alembert solution for
        the 1D wave equation with zero initial velocity.
        """
        if self._e0 is None:
            raise RuntimeError("Initial field has not been loaded")
        self._t = float(t_final)

    def get_e(self):
        """Return the field at the current simulation time."""
        if self._e0 is None:
            raise RuntimeError("Initial field has not been loaded")

        # x might not be strictly sorted; handle both cases.
        x = self.x
        e0 = self._e0
        x_minus = x - self.c * self._t
        x_plus = x + self.c * self._t

        if np.all(np.diff(x) > 0):
            e_minus = np.interp(x_minus, x, e0, left=0.0, right=0.0)
            e_plus = np.interp(x_plus, x, e0, left=0.0, right=0.0)
        else:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            e_sorted = e0[sort_idx]
            e_minus = np.interp(x_minus, x_sorted, e_sorted, left=0.0, right=0.0)
            e_plus = np.interp(x_plus, x_sorted, e_sorted, left=0.0, right=0.0)

        return 0.5 * (e_minus + e_plus)
