import numpy as np

def calcular_exp_y(M_H2O , sigma_s , T_a_em_k , rho_w , R, v_ion , Phi_s , m_s , r_i , rho_s ,  M_NaCl):
    termo1 = (2 * M_H2O * sigma_s) / (R * T_a_em_k * rho_w * r_i)
    termo2 = (v_ion * Phi_s * m_s*( M_H2O / M_NaCl)) / ((4/3) * np.pi * rho_s * r_i**3 - m_s)
    Y = termo1 - termo2
    exp_yy = np.exp(Y)
    return exp_yy

