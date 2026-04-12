import numpy as np
import matplotlib.pyplot as plt

C = 1.0

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

class FDTD1D:
    mu0 = 1.0
    eps0 = 1.0  
    
    def __init__(self, x, boundaries=None, x_o=None, pert=None):
        self.x = x
        self.xH = (self.x[1:] + self.x[:-1]) / 2.0
        self.dx = x[1] - x[0]
        self.dt = self.dx/C
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

    # Cambia el estado del campo inicial a lo que se le pase
    def load_initial_field(self, e0):
        self.e = e0.copy()
    
    # Función de actualización
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
                self.e[0] -= 2*r*self.h[0] 
            if self.boundaries[1] == 'PMC':
                self.e[-1] += 2*r*self.h[-1] 

        if self.pert is not None and self.x_o is not None:
            idx = np.argmin(np.abs(self.x - self.x_o))
            self.e[idx] = self.pert(self.t)

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
    

    
def permitividad_ag(E_eV):
        eps0 = 1.0
        eps_inf = 1.0

        # Parámetros extraídos de la Tabla I del paper (con el signo de c4 corregido)

        cp = np.array([
            0.5987 + 4195j,
            -0.2211 + 0.2680j,
            -4.240 + 732.4j,
            0.6391 - 0.07186j, 
            1.806 + 4.563j,
            1.443 - 82.19j
        ])
    
        ap = np.array([
            -0.02502 - 0.008626j,
            -0.2021 - 0.9407j,
            -14.67 - 1.338j,
            -0.2997 - 4.034j,
            -1.896 - 4.808j,
            -9.396 - 6.477j
        ])


        eps = np.full_like(E_eV, eps0 * eps_inf, dtype=complex)

        # Convención FDTD estándar: exp(jωt) en el dominio de Laplace s = jω

        s = 1j * E_eV  

        for c, a in zip(cp, ap):
            
            eps += eps0 * (c/(s - a) + np.conj(c)/(s - np.conj(a)))

        return eps



def transmitancia_slab(E_eV, grosor_nm=100.0):

        """Calcula la transmitancia teórica de un slab usando Matriz de Transferencia."""

        hc = 1239.84193

        eps_c = permitividad_ag(E_eV)

        n_c = np.sqrt(np.conj(eps_c))

        k0 = (2 * np.pi * E_eV) / hc

        phi = k0 * n_c * grosor_nm

        t_total = 1.0 / (np.cos(phi) - 0.5j * (n_c + 1.0 / n_c) * np.sin(phi))

        return np.abs(t_total)**2