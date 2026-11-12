import numpy as np

def calculate_Dw_prime(T_gota_em_k, R,P, r_i, M_H2O , T0 , P0 , alpha_c = 0.036 , Delta_w = 8e-8):
    # Difusividade do vapor de água modificada para efeitos de não-continuo
    D_w = 2.11e-5 * (T_gota_em_k / T0)**1.94 * (P0 / P)
    denominator = r_i / (r_i + Delta_w) + (D_w / (r_i * alpha_c)) * ((2 * np.pi * M_H2O)/(R*T_gota_em_k))**0.5
    # Difusividade molecular do vapor de água no ar
    D_linha_water = D_w / denominator
    a = (2 * np.pi * M_H2O)/(R*T_gota_em_k)
    b = T_gota_em_k / T0
    #print(f"T_gota_em_k = {T_gota_em_k}   |   dentro da raíz D_w = {b}  |   dentro da raíz denominador = {a}")
    return D_linha_water

