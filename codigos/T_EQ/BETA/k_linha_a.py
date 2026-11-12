import numpy as np
# Condutividade térmica modificada para efeitos de não-continuum
def calculate_K_linha_a(T_gota,T_gota_em_k,P, r_i , R , T0 , P0, M_a = 28.9644e-3 , alpha_T = 0.7 , 
                        delta_T = 2.16e-7, c_pa = 1.006e3):
    rho_a = 1.2923 * (T0 / T_gota_em_k) * (P / P0)  
    k_a = 2.411e-2 * (1 + 3.309e-3 * T_gota - 1.441e-6 * T_gota**2)  
    denominator = r_i/(r_i + delta_T) + (k_a / (r_i * alpha_T * rho_a * c_pa)) * ((2 * np.pi * M_a) / (R*T_gota_em_k))**0.5
    a = (2 * np.pi * M_a) / (R*T_gota_em_k)
    #print(f"T_gota = {T_gota}   |   dentro da raíz = {a}")
    k_linha_aa = k_a / denominator
    return k_linha_aa

