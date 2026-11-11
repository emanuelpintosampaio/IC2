import math

def calculate_K_linha_a2(T_eq,P, r_tau_r2 , R , T0 , P0, M_a = 28.9644e-3 , alpha_T = 0.7 , delta_T = 2.16e-7, c_pa = 1.006e3):
    rho_a2 = 1.2923 * (T0 / T_eq) * (P / P0)  
    k_a2 = 2.411e-2 * (1 + 3.309e-3 * T_eq - 1.441e-6 * T_eq**2)  
    denominator2 = r_tau_r2/(r_tau_r2 + delta_T) + (k_a2 / (r_tau_r2 * alpha_T * rho_a2 * c_pa)) * ((2 * math.pi * M_a) / (R*T_eq))**0.5
    k_linha_aa2 = k_a2 / denominator2
    return k_linha_aa2