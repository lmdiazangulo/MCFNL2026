import numpy as np

C = 1.0


class FDTD1D:
    """1D FDTD solver for the wave equation using the Yee scheme (CFL=1)."""

    def __init__(self, x):
        self.x = x
        self.dx = x[1] - x[0]
        self.dt = self.dx / C          # CFL = 1 → exact propagation
        self.N = len(x)
        self.e = np.zeros(self.N)
        self.h = np.zeros(self.N - 1)  # H lives at half-integer spatial nodes
        self.t = 0.0

    def load_initial_field(self, e0):
        """Set E(x, 0) = e0, H(x, 0) = 0.

        The leapfrog scheme stores H at t = −dt/2.  To be consistent with
        H(x,0)=0 and ∂H/∂t = ∂E/∂x, we initialise
            H^{-1/2}_{j+1/2} = −(dt/(2·dx)) · (E^0_{j+1} − E^0_j)
        which makes the first E update exact (D'Alembert's formula).
        """
        self.e = e0.copy()
        ratio = self.dt / (2.0 * self.dx)
        self.h = -ratio * (self.e[1:] - self.e[:-1])
        self.t = 0.0

    def _step(self):
        r = self.dt / self.dx
        # Advance H by one full step (n-1/2 → n+1/2)
        self.h += r * (self.e[1:] - self.e[:-1])
        # Advance interior E by one full step (n → n+1); boundaries stay fixed (≈0)
        self.e[1:-1] += r * (self.h[1:] - self.h[:-1])
        self.t += self.dt

    def run_until(self, t_final):
        n_steps = round((t_final - self.t) / self.dt)
        for _ in range(n_steps):
            self._step()
        self.t = t_final  # correct any floating-point drift

    def get_e(self):
        return self.e.copy()
