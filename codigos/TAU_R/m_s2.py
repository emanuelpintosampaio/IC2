import numpy as np

def calcular_rho_ww2(T_eq):
    A0 = 999.8396
    A1 = 18.224944
    A2 = -7.922210e-3
    B = 1.8159725e-2
    numerator = A0 + A1 * T_eq + A2 * T_eq**2
    denominator = 1 + B * T_eq
    rho_ww2 = numerator / denominator
    return rho_ww2

def calcular_ms2(rho_ww2,r_tau_r2,s):
    m_salt2 = (4/3) * np.pi * rho_ww2 * (r_tau_r2 ** 3) * (s / (1 - s))
    return m_salt2