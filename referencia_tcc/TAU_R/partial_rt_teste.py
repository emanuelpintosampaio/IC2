import math
import numpy as np

def calculate_tau_r_(r_i, r_eq, dr_dt, dr_dt_2):
    delta_r = r_i - r_eq
    if delta_r == 0:
        raise ValueError("r_i não pode ser igual a r_eq (divisão por zero).")
    
    # Cálculo dos coeficientes da equação quadrática
    a = (1/2)*((1/(r_i-r_eq)) * (dr_dt_2 ** 2) - 1/((r_i-r_eq)**2) * ((dr_dt)**2))
    b = (1/(r_i-r_eq))*dr_dt
    c = 1
    
    # Caso não quadrático
    if a == 0:
        if dr_dt != 0:
            raise ValueError("Erro inesperado: a é zero, mas dr_dt não é zero.")
        if 1 + (dr_dt_2) / (2 * delta_r) == 0:
            print("Infinitas soluções: qualquer tau_r satisfaz a equação.")
            return None
        else:
            raise ValueError("Não há solução quando dr_dt é zero.")
    
    # Resolução da equação quadrática
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("Sem soluções reais (discriminante negativo).")
    
    sqrt_discriminant = math.sqrt(discriminant)
    tau_r1 = (-b + sqrt_discriminant) / (2 * a)
    tau_r2 = (-b - sqrt_discriminant) / (2 * a)
    
    return (tau_r1, tau_r2)

def calcular_tau_r(f, M_H2O, sigma_s, R, T_gota_em_k, rho_w, r_i,
                            rho_s, m_s, v_ion, Phi_s, M_NaCl, D_linha_w,
                            e_sat, L_v, k_linha_a, r_eq):
    # Cálculo de Y
    term1 = (2 * M_H2O * sigma_s) / (R * T_gota_em_k * rho_w * r_i)
    term2 = (v_ion * Phi_s * m_s * (M_H2O / M_NaCl)) / (
        (4 * np.pi * rho_s * ((r_i**3) / 3)) - m_s)
    Y = term1 - term2

    # ζ
    zeta = (f - 1) - Y

# η
    termo1 = (rho_s * R * T_gota_em_k) / (D_linha_w * M_H2O * e_sat)
    termo2 = ((rho_s * L_v) / (k_linha_a * T_gota_em_k)) * (
    (L_v * M_H2O) / (R * T_gota_em_k) - 1)
    eta = termo1 + termo2

# Primeira derivada
    dr_dt = zeta / (r_i * eta)

# Derivada de ζ
    dzeta_dt = (zeta/(r_i*eta))*(
    (2*M_H2O*sigma_s) / (R * T_gota_em_k * rho_w * r_i**2) -  ((4*np.pi*rho_s*(r_i**2))*( v_ion * Phi_s * m_s)*(M_H2O/M_NaCl))/((4/3 * np.pi * r_i**3 * rho_s - m_s)**2)
    )

    # Derivada  η
    d_r_eta_dt = (zeta/(r_i*eta))*eta



    dr_dt_2 = (dzeta_dt * r_i * eta - zeta* d_r_eta_dt) / ((r_i * eta)**2)


    # Cálculo final usando a função unificada
    return calculate_tau_r_(r_i, r_eq, dr_dt, dr_dt_2)

