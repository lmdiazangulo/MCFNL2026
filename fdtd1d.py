import numpy as np

C = 1.0

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)


class FDTD1D:
    def __init__(self, x, boundaries=None, epsilon=1.0, sigma=0.0):
        self.x = x
        self.xH = (self.x[:1] + self.x[:-1]) / 2.0
        self.dx = x[1] - x[0]
        self.dt = self.dx / C  
        self.N = len(x)
        self.e = np.zeros(self.N)
        self.h = np.zeros(self.N - 1) 
        self.t = 0.0
        self.boundaries = boundaries

        # Allow epsilon and sigma to be scalars or arrays (per node)
        if np.isscalar(epsilon):
            self.epsilon = np.full(self.N, epsilon)
        else:
            self.epsilon = np.array(epsilon, dtype=float)
        if np.isscalar(sigma):
            self.sigma = np.full(self.N, sigma)
        else:
            self.sigma = np.array(sigma, dtype=float)

    def load_initial_field(self, e0):
        self.e = e0.copy()
        
    def _step(self):
        r = self.dt / self.dx
    
        # Save boundary values before E update (needed for Mur ABC)
        if self.boundaries is not None:
            if self.boundaries[0] == 'mur':
                e_old_left_0 = self.e[0]
                e_old_left_1 = self.e[1]
            if self.boundaries[1] == 'mur':
                e_old_right_0 = self.e[-1]
                e_old_right_1 = self.e[-2]

        eps = self.epsilon[1:-1]
        sig = self.sigma[1:-1]
        self.e[1:-1] = (
            (eps - sig * self.dt / 2) / 
            (eps + sig * self.dt / 2) * self.e[1:-1]
            + self.dt / (eps + sig * self.dt / 2) 
            * (self.h[1:] - self.h[:-1]) / self.dx
        )

        if self.boundaries is not None:
            if self.boundaries[0] == 'PEC':
                self.e[0] = 0.0
            if self.boundaries[1] == 'PEC':
                self.e[-1] = 0.0
            if self.boundaries[0] == 'periodic':
                self.e[0] += r * (self.h[0] - self.h[-1])
                self.e[-1] = self.e[0]
            if self.boundaries[0] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[0] = e_old_left_1 + mur_coeff * (self.e[1] - e_old_left_0)
            if self.boundaries[1] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[-1] = e_old_right_1 + mur_coeff * (self.e[-2] - e_old_right_0)

        self.h += r * (self.e[1:] - self.e[:-1])
    
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