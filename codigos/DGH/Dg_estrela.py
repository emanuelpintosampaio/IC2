import numpy as np

def calcular_Dg_estrela(r_i, T_a_em_k,R):
    Dg=0.16
    alpha_epsilon=14e-5
    M_CO2=0.044
    v_epsilon = np.sqrt(8 * R * T_a_em_k / (np.pi * M_CO2))
    return Dg / (1 + (4 * Dg) / (r_i * alpha_epsilon * v_epsilon))
