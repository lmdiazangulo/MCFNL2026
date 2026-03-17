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

    def load_initial_field(self, e0):
        self.e = e0.copy()
        
    def _step(self):
        r = self.dt / self.dx
        
        self.e[1:-1] += r * (self.h[1:] - self.h[:-1])

        if self.boundaries is not None:
            if self.boundaries[0] == 'PEC':
                self.e[0] = 0.0
            if self.boundaries[1] == 'PEC':
                self.e[-1] = 0.0

        self.h += r * (self.e[1:] - self.e[:-1])
        
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
