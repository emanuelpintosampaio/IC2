import numpy as np

# Função para calcular o volume molar aparente de uma solução aquosa (v_a)
def calcular_v_a(T_gota, m_s, M_NaCl, r_i):
    v_a0 = (12.97 + 0.2340 * T_gota - 4.210e-3 * T_gota**2 + 2.857e-5 * T_gota**3) * 1e-6    
    S_v_star = (2.982 - 4.970e-2 * T_gota + 6.032e-4 * T_gota**2) * 1e-6
    c = 10e-3 * (m_s / M_NaCl) / (4/3) * np.pi * r_i**3  
    v_a = v_a0 + S_v_star * np.sqrt(c)
    return v_a

# Função para calcular a massa de uma gotícula esférica (m_w)
def calcular_massa_agua(r_i, rho_ww):
    volume = (4/3) * np.pi * (r_i ** 3)
    m_ww = rho_ww * volume
    return m_ww

# Função para calcular a densidade da água do mar (rho_spray)
def calcular_rho_spray(rho_ww, m_s, m_ww, v_a, M_NaCl):
    rho_spray = rho_ww * (1 + (m_s / m_ww)) / (1 + v_a * (rho_ww / M_NaCl) * (m_s / m_ww))
    return rho_spray

