import numpy as np

# Função para calcular o volume molar aparente de uma solução aquosa (v_a)
def calcular_v_a2(T_eq, m_s2, M_NaCl, r_tau_r2):
    v_a02 = (12.97 + 0.2340 * T_eq - 4.210e-3 * T_eq**2 + 2.857e-5 * T_eq**3) * 1e-6    
    S_v_star2 = (2.982 - 4.970e-2 * T_eq + 6.032e-4 * T_eq**2) * 1e-6
    c2 = 10e-3 * (m_s2 / M_NaCl) / (4/3) * np.pi * r_tau_r2**3  
    v_a2 = v_a02  + S_v_star2 * np.sqrt(c2)
    return v_a2

# Função para calcular a massa de uma gotícula esférica (m_w)
def calcular_massa_agua2(r_tau_r2, rho_ww2):
    volume = (4/3) * np.pi * (r_tau_r2 ** 3)    
    m_ww2 = rho_ww2 * volume
    return m_ww2

# Função para calcular a densidade da água do mar (rho_spray)
def calcular_rho_spray2(rho_ww2, m_s2, m_ww2, v_a2, M_NaCl):
    rho_spray2 = rho_ww2 * (1 + (m_s2 / m_ww2)) / (1 + v_a2 * (rho_ww2 / M_NaCl) * (m_s2 / m_ww2))
    return rho_spray2
