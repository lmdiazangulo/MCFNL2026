import numpy as np
import matplotlib.pyplot as plt

C = 1.0

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

class FDTD1D:
    mu0 = 1.0
    eps0 = 1.0  
    
    def __init__(
                self,
                x, 
                boundaries=None,
                x_o=None, 
                pert=None,
                pert_dir = None,
                panel_bool = False, 
                panel_center = None, 
                panel_thickness = None,
                panel_eps_r = None,
                panel_sigma = None,
                panel_mu_r = None,
                obs_bool = [False,False],
                obs_left_offset = None,
                obs_right_offset = None,
        ):

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
        self.pert_dir = pert_dir

        self.panel_bool = panel_bool
        self.panel_sigma = panel_sigma
        self.panel_center = panel_center
        self.panel_thickness = panel_thickness
        self.panel_eps_r = panel_eps_r
        self.panel_mu_r = panel_mu_r
        self.obs_bool = obs_bool
        self.obs_left_offset = obs_left_offset
        self.obs_right_offset = obs_right_offset

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
                self.e[0] -= 2*r*self.h[0] 
            if self.boundaries[1] == 'PMC':
                self.e[-1] += 2*r*self.h[-1] 
        
        if self.pert is not None and self.x_o is not None:
            idx = np.argmin(np.abs(self.x - self.x_o))
            self.e[idx] += self.pert(self.t + self.dt/2 + self.dx/2/C) 
        
        self.h -= r * (self.e[1:] - self.e[:-1])
        
        self.t += self.dt   

    def run_until(self, t_final):
        
        if self.panel_bool == True and self.panel_center is not None and self.panel_thickness is not None:
            panel_left = self.panel_center - self.panel_thickness / 2
            panel_right = self.panel_center + self.panel_thickness / 2
            self.eps_r = np.where((self.x >= panel_left) & (self.x <= panel_right), self.panel_eps_r, 1.0)
            self.sig = np.where((self.x >= panel_left) & (self.x <= panel_right), self.panel_sigma, 0.0)
        
        if self.obs_bool[0] == True:
            obs_left_x = panel_left - self.obs_left_offset
            obs_left_idx = np.argmin(np.abs(self.x - obs_left_x))
        if self.obs_bool[1] == True:
            obs_right_x = panel_right + self.obs_right_offset
            obs_right_idx = np.argmin(np.abs(self.x - obs_right_x))
        
        n_steps = round((t_final - self.t) / self.dt)
        
        if self.obs_bool[0] == True or self.obs_bool[1] == True:
            t_array = np.zeros(n_steps)
        if self.obs_bool[0] == True: 
            E_left = np.zeros(n_steps)
        if self.obs_bool[1] == True:
            E_right = np.zeros(n_steps)

        for i in range(n_steps):

            self._step()
            plt.clf()
            plt.plot(self.x, self.get_e(), label="E")
            plt.plot(self.xH, self.get_h(), label="H")

            plt.ylim(-1.2, 1.2)
            plt.legend()
            plt.pause(0.001)

            if self.obs_bool[0] == True or self.obs_bool[1] == True:
                t_array[i] = self.t
            if self.obs_bool[0] == True:
                E_left[i] = self.e[obs_left_idx]
            if self.obs_bool[1] == True:
                E_right[i] = self.e[obs_right_idx]

        self.t = t_final  
        
    def get_e(self):
        return self.e.copy()

    def get_h(self):
        return self.h.copy()