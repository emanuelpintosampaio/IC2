import numpy as np

def calcular_rho_ww(T_gota):
    A0 = 999.8396
    A1 = 18.224944
    A2 = -7.922210e-3
    B = 1.8159725e-2
    numerator = A0 + A1 * T_gota + A2 * T_gota**2
    denominator = 1 + B * T_gota
    rho_ww = numerator / denominator
    return rho_ww

def calcular_ms(rho_ww,r,s):
    m_salt = (4/3) * np.pi * rho_ww * (r ** 3) * (s / (1 - s))
    return m_salt

