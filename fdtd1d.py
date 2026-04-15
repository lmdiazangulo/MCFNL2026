import numpy as np
import matplotlib.pyplot as plt

C = 1.0


def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)


class FDTD1D:
    mu0 = 1.0
    eps0 = 1.0

    def __init__(self, x, boundaries=None, x_o=None, pert=None, pert_dir=False):
        self.x = x
        self.xH = (self.x[1:] + self.x[:-1]) / 2.0
        self.dx = x[1] - x[0]
        self.dt = self.dx / C
        self.N = len(x)
        self.e = np.zeros(self.N)
        self.h = np.zeros(self.N - 1)
        self.t = 0.0
        self.boundaries = boundaries

        self.sig = np.zeros(self.N)
        self.eps_r = np.ones(self.N)
        self.eps = self.eps0 * self.eps_r
        self.x_o = x_o
        self.pert = pert
        self.pert_dir = pert_dir

    def load_initial_field(self, e0):
        self.e = e0.copy()

    def set_panel(self, center, d, eps_r=1.0, sigma=0.0):
        """Set a single homogeneous panel on the grid."""
        left = center - d / 2
        right = center + d / 2
        self.eps_r = np.where(
            (self.x >= left) & (self.x <= right), eps_r, 1.0)
        self.sig = np.where(
            (self.x >= left) & (self.x <= right), sigma, 0.0)

    def set_multilayer(self, center, layers):
        """Set a multilayer stack centered at `center`."""
        total_d = sum(lay['d'] for lay in layers)
        edge = center - total_d / 2
        self.eps_r = np.ones_like(self.x)
        self.sig = np.zeros_like(self.x)
        for lay in layers:
            mask = (self.x >= edge) & (self.x < edge + lay['d'])
            self.eps_r[mask] = lay.get('eps_r', 1.0)
            self.sig[mask] = lay.get('sigma', 0.0)
            edge += lay['d']

    def _step(self):
        r = self.dt / self.dx

        self.eps = self.eps0 * self.eps_r

        ca = (2 * self.eps - self.sig * self.dt) / (2 * self.eps + self.sig * self.dt)
        cb = (2 * self.dt / self.dx) / (2 * self.eps + self.sig * self.dt)

        if self.boundaries is not None:
            if self.boundaries[0] == 'mur':
                e_old_left_0 = self.e[0]
                e_old_left_1 = self.e[1]
            if self.boundaries[1] == 'mur':
                e_old_right_0 = self.e[-1]
                e_old_right_1 = self.e[-2]

        if self.pert_dir and self.pert is not None and self.x_o is not None and self.t != 0.0:
            idx = np.argmin(np.abs(self.xH - self.x_o))
            self.h[idx] += self.pert(self.t)

        self.e[1:-1] = ca[1:-1] * self.e[1:-1] - cb[1:-1] * (self.h[1:] - self.h[:-1])

        if self.boundaries is not None:
            if self.boundaries[0] == 'PEC':
                self.e[0] = 0.0
            if self.boundaries[1] == 'PEC':
                self.e[-1] = 0.0
            if self.boundaries[0] == 'periodic':
                self.e[0] = ca[0] * self.e[0] - cb[0] * (self.h[0] - self.h[-1])
                self.e[-1] = self.e[0]
            if self.boundaries[0] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[0] = e_old_left_1 + mur_coeff * (self.e[1] - e_old_left_0)
            if self.boundaries[1] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[-1] = e_old_right_1 + mur_coeff * (self.e[-2] - e_old_right_0)
            if self.boundaries[0] == 'PMC':
                self.e[0] -= 2 * r * self.h[0]
            if self.boundaries[1] == 'PMC':
                self.e[-1] += 2 * r * self.h[-1]

        if self.pert is not None and self.x_o is not None:
            idx = np.argmin(np.abs(self.x - self.x_o))
            self.e[idx] += self.pert(self.t + self.dt / 2 + self.dx / 2 / C)

        self.h -= r * (self.e[1:] - self.e[:-1])

        self.t += self.dt

    def run_until(self, t_final):
        n_steps = round((t_final - self.t) / self.dt)
        for _ in range(n_steps):
            self._step()
        self.t = t_final

    def get_e(self):
        return self.e.copy()

    def get_h(self):
        return self.h.copy()


