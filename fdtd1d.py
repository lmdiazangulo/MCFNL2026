import numpy as np
import matplotlib.pyplot as plt

C = 1.0

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

def panel_transfer_matrix(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    omega = 2.0 * np.pi * freq

    eps_c = eps_r - 1j * sigma / omega
    gamma = 1j * omega * np.sqrt(mu_r * eps_c)
    eta = np.sqrt(mu_r / eps_c)

    gd = gamma * d
    ch = np.cosh(gd)
    sh = np.sinh(gd)

    Phi = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi[:, 0, 0] = ch
    Phi[:, 0, 1] = eta * sh
    Phi[:, 1, 0] = sh / eta
    Phi[:, 1, 1] = ch
    return Phi


# ── BONUS: multilayer panel ──
def stack_transfer_matrix(freq, layers):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    Phi_total = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi_total[:, 0, 0] = 1.0
    Phi_total[:, 1, 1] = 1.0

    for layer in layers:
        Phi_i = panel_transfer_matrix(
            freq, d=layer['d'],
            eps_r=layer.get('eps_r', 1.0),
            sigma=layer.get('sigma', 0.0),
            mu_r=layer.get('mu_r', 1.0),
        )
        Phi_new = np.zeros_like(Phi_total)
        Phi_new[:, 0, 0] = Phi_total[:, 0, 0]*Phi_i[:, 0, 0] + Phi_total[:, 0, 1]*Phi_i[:, 1, 0]
        Phi_new[:, 0, 1] = Phi_total[:, 0, 0]*Phi_i[:, 0, 1] + Phi_total[:, 0, 1]*Phi_i[:, 1, 1]
        Phi_new[:, 1, 0] = Phi_total[:, 1, 0]*Phi_i[:, 0, 0] + Phi_total[:, 1, 1]*Phi_i[:, 1, 0]
        Phi_new[:, 1, 1] = Phi_total[:, 1, 0]*Phi_i[:, 0, 1] + Phi_total[:, 1, 1]*Phi_i[:, 1, 1]
        Phi_total = Phi_new
    return Phi_total
# ── END BONUS ──


def RT_from_transfer_matrix(Phi):
    A, B, C_, D = Phi[:, 0, 0], Phi[:, 0, 1], Phi[:, 1, 0], Phi[:, 1, 1]
    denom = A + B + C_ + D
    return (A + B - C_ - D) / denom, 2.0 / denom


def reflection_transmission(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    return RT_from_transfer_matrix(panel_transfer_matrix(freq, d, eps_r, sigma, mu_r))

class FDTD1D:
    mu0 = 1.0
    eps0 = 1.0  
    
    def __init__(self, x, boundaries=None, x_o=None, pert=None, pert_dir = False):
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

    def load_initial_field(self, e0):
        self.e = e0.copy()
    
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
        n_steps = round((t_final - self.t) / self.dt)
        for _ in range(n_steps):
            self._step()
            # plt.clf()
            # plt.plot(self.x, self.get_e(), label="E")
            # plt.plot((self.x[1:] + self.x[:-1]) / 2.0, self.get_h(), label="H")

            # plt.ylim(-1.2, 1.2)
            # plt.legend()
            # plt.pause(0.001)
        self.t = t_final  
        
    def get_e(self):
        return self.e.copy()

    def get_h(self):
        return self.h.copy()


class FDTD1DNonUniform(FDTD1D):

    def __init__(self, x, boundaries=None, x_o=None, pert=None):

        super().__init__(x, boundaries=boundaries, x_o=x_o, pert=pert)

        self.dx_h = np.diff(x)          
        self.dx_e = np.diff(self.xH)     
        self.dt = float(np.min(self.dx_h)) / C

    def _step(self):

        eps = self.eps0 * self.eps_r 

        e_old_left_0 = e_old_left_1 = None
        e_old_right_0 = e_old_right_1 = None
        if self.boundaries is not None:
            if self.boundaries[0] == 'mur':
                e_old_left_0 = self.e[0]
                e_old_left_1 = self.e[1]
            if self.boundaries[1] == 'mur':
                e_old_right_0 = self.e[-1]
                e_old_right_1 = self.e[-2]


        denom_int = 2.0 * eps[1:-1] + self.sig[1:-1] * self.dt
        ca_int    = (2.0 * eps[1:-1] - self.sig[1:-1] * self.dt) / denom_int
        cb_int    = (2.0 * self.dt / self.dx_e) / denom_int 

        self.e[1:-1] = (ca_int * self.e[1:-1]
                        - cb_int * (self.h[1:] - self.h[:-1]))

        if self.boundaries is not None:
            bc_l, bc_r = self.boundaries

            if bc_l == 'PEC':
                self.e[0] = 0.0
            if bc_r == 'PEC':
                self.e[-1] = 0.0

            if bc_l == 'periodic':

                dx_periodic = 0.5 * (self.dx_h[0] + self.dx_h[-1])
                denom0 = 2.0 * eps[0] + self.sig[0] * self.dt
                ca0    = (2.0 * eps[0] - self.sig[0] * self.dt) / denom0
                cb0    = (2.0 * self.dt / dx_periodic) / denom0
                self.e[0]  = ca0 * self.e[0] - cb0 * (self.h[0] - self.h[-1])
                self.e[-1] = self.e[0]

            if bc_l == 'mur':
                mc = (C * self.dt - self.dx_h[0]) / (C * self.dt + self.dx_h[0])
                self.e[0] = e_old_left_1 + mc * (self.e[1] - e_old_left_0)
            if bc_r == 'mur':
                mc = (C * self.dt - self.dx_h[-1]) / (C * self.dt + self.dx_h[-1])
                self.e[-1] = e_old_right_1 + mc * (self.e[-2] - e_old_right_0)

            if bc_l == 'PMC':
                self.e[0]  -= 2.0 * (self.dt / self.dx_h[0])  * self.h[0]
            if bc_r == 'PMC':
                self.e[-1] += 2.0 * (self.dt / self.dx_h[-1]) * self.h[-1]

        if self.pert is not None and self.x_o is not None:
            idx = int(np.argmin(np.abs(self.x - self.x_o)))
            self.e[idx] = self.pert(self.t)


        self.h -= (self.dt / self.dx_h) * (self.e[1:] - self.e[:-1])

        self.t += self.dt