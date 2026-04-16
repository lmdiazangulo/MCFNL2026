import numpy as np
import matplotlib.pyplot as plt

C = 1.0
def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

class FDTD1D:
    def __init__(self, x, boundaries=None, x_o=None, pert=None, C=1.0, eps0=1.0, mu0=1.0):
        self.C = C
        self.eps0 = eps0
        self.mu0 = mu0
        
        self.x = x
        self.xH = (self.x[1:] + self.x[:-1]) / 2.0
        self.dx = x[1] - x[0]
        # En 1D, el límite de Courant es dx / C exacto
        self.dt = self.dx / self.C 
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
        
        # Almacenamiento para los polos dispersivos (ADE)
        self.dispersive_poles = []

    def add_dispersive_material(self, mask, cp_array, ap_array):
        """
        Añade un material dispersivo a las regiones definidas por 'mask'.
        Los arrays cp_array y ap_array deben contener los coeficientes 
        complejos en unidades físicas compatibles (ej. rad/s para simulación SI).
        """
        for c, a in zip(cp_array, ap_array):
            # Coeficientes k_p y beta_p según la Ec. (6) y (7) del paper
            k_p = (1 + a * self.dt / 2.0) / (1 - a * self.dt / 2.0)
            beta_p = (self.eps0 * c * self.dt) / (1 - a * self.dt / 2.0)
            
            self.dispersive_poles.append({
                'mask': mask,
                'k_p': k_p,
                'beta_p': beta_p,
                'J_p': np.zeros(self.N, dtype=complex) # Variable auxiliar J_p
            })

    def load_initial_field(self, e0):
        self.e = e0.copy()
    
    def _step(self):
        r = self.dt / (self.mu0 * self.dx)
        self.eps = self.eps0 * self.eps_r

        # 1. Sumatorias para el material dispersivo 
        sum_Re_beta = np.zeros(self.N)
        sum_J_term = np.zeros(self.N)
        
        for pole in self.dispersive_poles:
            mask = pole['mask']
            sum_Re_beta[mask] += np.real(pole['beta_p'])
            sum_J_term[mask] += np.real((1 + pole['k_p']) * pole['J_p'][mask])

        # 2. Coeficientes actualizados con dispersión (Ecuación 6)
        denom = 2 * self.eps + 2 * sum_Re_beta + self.sig * self.dt
        
        ca = (2 * self.eps + 2 * sum_Re_beta - self.sig * self.dt) / denom
        cb = (2 * self.dt / self.dx) / denom
        cc = (2 * self.dt) / denom

        if self.boundaries is not None:
            if self.boundaries[0] == 'mur':
                e_old_left_0 = self.e[0]
                e_old_left_1 = self.e[1]
            if self.boundaries[1] == 'mur':
                e_old_right_0 = self.e[-1]
                e_old_right_1 = self.e[-2]

        # 3. Calculamos el nuevo E temporalmente
        e_new = self.e.copy()
        e_new[1:-1] = (ca[1:-1] * self.e[1:-1] 
                       - cb[1:-1] * (self.h[1:] - self.h[:-1]) 
                       - cc[1:-1] * sum_J_term[1:-1])

        # 4. Actualizamos las variables auxiliares J_p (Ecuación 7)
        for pole in self.dispersive_poles:
            mask = pole['mask']
            deriv_E = (e_new[mask] - self.e[mask]) / self.dt
            pole['J_p'][mask] = pole['k_p'] * pole['J_p'][mask] + pole['beta_p'] * deriv_E

        # 5. Aplicamos E definitivamente
        self.e = e_new

        # 6. Condiciones de Frontera y Perturbaciones
        if self.boundaries is not None:
            if self.boundaries[0] == 'PEC':
                self.e[0] = 0.0
            if self.boundaries[1] == 'PEC':
                self.e[-1] = 0.0
            if self.boundaries[0] == 'periodic':
                self.e[0] = ca[0] * self.e[0] - cb[0] * (self.h[0] - self.h[-1])
                self.e[-1] = self.e[0]
            if self.boundaries[0] == 'mur':
                mur_coeff = (self.C * self.dt - self.dx) / (self.C * self.dt + self.dx)
                self.e[0] = e_old_left_1 + mur_coeff * (self.e[1] - e_old_left_0)
            if self.boundaries[1] == 'mur':
                mur_coeff = (self.C * self.dt - self.dx) / (self.C * self.dt + self.dx)
                self.e[-1] = e_old_right_1 + mur_coeff * (self.e[-2] - e_old_right_0)
            if self.boundaries[0] == 'PMC':
                self.e[0] -= 2 * (self.dt/self.dx) * self.h[0] 
            if self.boundaries[1] == 'PMC':
                self.e[-1] += 2 * (self.dt/self.dx) * self.h[-1] 

        if self.pert is not None and self.x_o is not None:
            idx = np.argmin(np.abs(self.x - self.x_o))
            self.e[idx] = self.pert(self.t)

        # 7. Actualización de H (con la permeabilidad magnética correcta)
        r_mag = self.dt / (self.mu0 * self.dx)
        self.h -= r_mag * (self.e[1:] - self.e[:-1])
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
    """
    Calcula la permitividad compleja de la plata (Ag) en función de la energía (eV)
    usando el modelo ADE con pares de polos y residuos conjugados[cite: 73, 74].
    """
    eps0 = 1.0
    eps_inf = 1.0
    
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
    s = 1j * E_eV  
    
    for c, a in zip(cp, ap):
        eps += eps0 * (c/(s - a) + np.conj(c)/(s - np.conj(a)))
        
    return eps

def transmitancia_slab(E_eV, grosor_nm=100.0):
    hc = 1239.84193 
    
    eps_c = permitividad_ag(E_eV)
    n_c = np.sqrt(np.conj(eps_c))
    
    k0 = (2 * np.pi * E_eV) / hc
    phi = k0 * n_c * grosor_nm
    
    t_total = 1.0 / (np.cos(phi) - 0.5j * (n_c + 1.0 / n_c) * np.sin(phi))
    return np.abs(t_total)**2

def get_ag_poles_norm(C_fisica=299792458.0):
    eV_to_rads = 1.519267e15
    factor_conversion = eV_to_rads / C_fisica 
    
    cp_eV = np.array([
        0.5987 + 4195j, -0.2211 + 0.2680j, -4.240 + 732.4j, 
        0.6391 - 0.07186j, 1.806 + 4.563j, 1.443 - 82.19j
    ])
    ap_eV = np.array([
        -0.02502 - 0.008626j, -0.2021 - 0.9407j, -14.67 - 1.338j, 
        -0.2997 - 4.034j, -1.896 - 4.808j, -9.396 - 6.477j
    ])
    return cp_eV * factor_conversion, ap_eV * factor_conversion

def reflectancia_slab(E_eV, grosor_nm=100.0):
    hc = 1239.84193 
    eps_c = permitividad_ag(E_eV)
    n_c = np.sqrt(np.conj(eps_c))
    k0 = (2 * np.pi * E_eV) / hc
    phi = k0 * n_c * grosor_nm
    
    r01 = (1.0 - n_c) / (1.0 + n_c)
    r_total = r01 * (1.0 - np.exp(2j * phi)) / (1.0 - r01**2 * np.exp(2j * phi))
    return np.abs(r_total)**2

def absorbancia_slab(E_eV, grosor_nm=100.0):
    T = transmitancia_slab(E_eV, grosor_nm)
    R = reflectancia_slab(E_eV, grosor_nm)
    return 1.0 - T - R

def extract_spectrum(E_time, dt, C_fisica=299792458.0):
    eV_to_rads = 1.519267e15
    freqs = np.fft.fftfreq(len(E_time), d=dt)
    w_rads_norm = 2 * np.pi * freqs
    E_eV_array = (w_rads_norm * C_fisica) / eV_to_rads
    E_fft = np.fft.fft(E_time)
    return E_eV_array, E_fft