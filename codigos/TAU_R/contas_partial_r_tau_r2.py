import numpy as np

def calcular_derivada_parcial2(f, M_H2O, sigma_s2, R, T_eq_em_k, rho_w, r_tau_r2,
                                rho_s2, m_s2, v_ion, Phi_s2, M_NaCl,
                                D_linha_w2, e_sat2, L_v2, k_linha2):
    # Y
    term1 = (2 * M_H2O * sigma_s2) / (R * T_eq_em_k * rho_w * r_tau_r2)
    term2 = (v_ion * Phi_s2 * m_s2 * (M_H2O / M_NaCl)) / ((4 * np.pi * rho_s2 * ((r_tau_r2**3) / 3)) - m_s2)
    Y = term1 - term2

    # ζ 
    zeta = (f - 1) - Y  

    # η 
    termo1 = (rho_s2 * R * T_eq_em_k) / (D_linha_w2 * M_H2O * e_sat2)
    termo2 = ((rho_s2 * L_v2) / (k_linha2 * T_eq_em_k)) * ((L_v2 * M_H2O) / (R * T_eq_em_k) - 1)
    eta = termo1 + termo2

    #  dr(tau_r/2)/dt 
    dr_dt = zeta / (r_tau_r2 * eta)

    return dr_dt
