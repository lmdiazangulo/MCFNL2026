import numpy as np

C = 1.0


class FDTD1D:
    def __init__(self, x, boundaries=None):
        self.x = x
        self.xH = (self.x[:1] + self.x[:-1]) / 2.0
        self.dx = x[1] - x[0]
        self.dt = self.dx / C  
        self.N = len(x)
        self.e = np.zeros(self.N)
        self.h = np.zeros(self.N - 1) 
        self.t = 0.0
        self.boundaries = boundaries

    def load_initial_field(self, field0):
        """Load an initial field into the solver.

        If the provided array has the same length as the E-field grid, it is
        treated as an initial electric field. If it matches the H-field grid
        length, it is treated as an initial magnetic field.
        """
        field0 = np.asarray(field0)

        if field0.shape == self.e.shape:
            self.e = field0.copy()
        elif field0.shape == self.h.shape:
            self.h = field0.copy()
        else:
            raise ValueError(
                "Initial field must have length {} (E field) or {} (H field), "
                "got {}".format(self.e.size, self.h.size, field0.size)
            )

    def _step(self):
        r = self.dt / self.dx
    
        self.e[1:-1] += r * (self.h[1:] - self.h[:-1])

        if self.boundaries is not None:
            left_bc, right_bc = self.boundaries

            # Perfect Electric Conductor (PEC): enforce E=0 at the boundaries
            if left_bc == 'PEC':
                self.e[0] = 0.0
            if right_bc == 'PEC':
                self.e[-1] = 0.0

            # Periodic boundary conditions for E
            if left_bc == 'periodic':
                self.e[0] += r * (self.h[0] - self.h[-1])
                self.e[-1] = self.e[0]

        self.h += r * (self.e[1:] - self.e[:-1])

        if self.boundaries is not None:
            left_bc, right_bc = self.boundaries

            # Perfect Magnetic Conductor (PMC): enforce H=0 at the boundaries
            if left_bc == 'PMC':
                self.h[0] = 0.0
            if right_bc == 'PMC':
                self.h[-1] = 0.0
    
        self.t += self.dt   

    def run_until(self, t_final):
        n_steps = round((t_final - self.t) / self.dt)
        for _ in range(n_steps):
            self._step()
        self.t = t_final  # correct any floating-point drift

    def get_e(self):
        return self.e.copy()

    def get_h(self):
        return self.h.copy()

