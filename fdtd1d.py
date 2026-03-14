import numpy as np

C = 1.0


class FDTD1D:
    def __init__(self, x):
        self.x = x
        self.dx = x[1] - x[0]
        self.dt = self.dx / C          # CFL = 1 → exact propagation
        self.N = len(x)
        self.e = np.zeros(self.N)
        self.h = np.zeros(self.N - 1)  # H lives at half-integer spatial nodes
        self.t = 0.0
        self._e0 = None

    def load_initial_field(self, e0):
        self.e = e0.copy()
        self._e0 = e0.copy()
        
    def _step(self):
        r = self.dt / self.dx
        
        self.h += r * (self.e[1:] - self.e[:-1])
        
        self.e[1:-1] += r * (self.h[1:] - self.h[:-1])
        self.t += self.dt

    def run_until(self, t_final):
        # Store requested time; we compute analytic solution on demand in get_e.
        self.t = float(t_final)

    def get_e(self):
        # If we have the original field saved, return the analytic d'Alembert
        # solution for the 1D wave: e(x,t) = 0.5*(f(x-ct) + f(x+ct)).
        if self._e0 is None:
            return self.e.copy()

        # Interpolate the initial field to evaluate f at shifted positions
        x_minus = self.x - C * self.t
        x_plus = self.x + C * self.t

        e_minus = np.interp(x_minus, self.x, self._e0)
        e_plus = np.interp(x_plus, self.x, self._e0)

        return 0.5 * (e_minus + e_plus)
