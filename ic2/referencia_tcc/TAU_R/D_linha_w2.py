import math

def calculate_Dw_prime2(T_eq_em_k, R,P, r_tau_r2, M_H2O , T0 , P0 , alpha_c = 0.036 , Delta_w = 8e-8):
    D_w2 = 2.11e-5 * (T_eq_em_k / T0)**1.94 * (P0 / P)
    denominator2 = r_tau_r2 / (r_tau_r2 + Delta_w) + (D_w2 / (r_tau_r2 * alpha_c)) * ((2 * math.pi * M_H2O)/(R*T_eq_em_k))**0.5
    D_linha_water2 = D_w2 / denominator2
    return D_linha_water2