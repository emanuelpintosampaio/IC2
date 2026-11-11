import numpy as np
import math
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela
import matplotlib.pyplot as plt

# ANDREAS
T_a = 18           # Temperatura do ar em °C ANDREAS(18)
T_a_em_k = T_a + 273.15  # Temperatura do ar em K ANDREAS(291.15)
T_mar_em_k = 293.15  # Temperatura do mar em K ANDREAS(293.15)
T_gota = 19.8      # Temperatura da gota em °C ANDREAS(20.2)
T_gota_em_k = T_gota + 273.15  # Temperatura da gota em K ANDREAS(293.35)
P = 1000           # Pressão em mb
M_H2O = 18.016e-3  # Massa molecular da água em kg/mol
r_i = 100e-6       # Raio em metros
R = 8.31           # Constante universal dos gases
M_NaCl = 58.443e-3  # Massa molecular de sal na água em kg/mol
rho_w = 1025       # Densidade da água do mar kg/m^3
v_ion = 2          # Número de íons por molécula de NaCl dissociada
s = 34/1000        # Salinidade fracionária (34 psu)
f = 0.9            # Umidade relativa fracionária ANDREAS(0.8)
C_ar = 0.0154      # Concentração do gás carbônico no ar
S = 34             # Salinidade do mar
R_atm = 0.082      # Constante universal dos gases
g = 9.81           # Gravidade
v_ar = 1.32e-5     # Viscosidade cinemática do ar
rho_ar = 1.225     # Densidade do ar
H_s = 6            # Altura significativa da onda 
T0 = 273.15
P0 = 1013.25
c_ps = 4000        # Calor específico da spray


def func(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    return U_f - const / a

def dfunc(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    d_a = -0.158 * (2/3) * (2 * r_i / v_ar)**(2/3) * U_f**(-1/3)
    return 1 - (const * d_a) / (a**2)

U_f0 = 0.1
U_f_sol = metodo_newton.Newton(func, dfunc, U_f0, 1e-6, 50)

# Tempo de vida da gota 
tau_f = H_s / (2 * U_f_sol)

e_sat_val = e_sat.calcular_esat(T_a)
L_v_val = L_v.calcular_lv(T_gota)
D_linha_w_val = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)

rho_ww = m_s.calcular_rho_ww(T_gota)
m_s_val = m_s.calcular_ms(rho_ww, r_i, s)
v_a = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w = rho_s.calcular_massa_agua(r_i, rho_ww)
sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
Phi_s_val = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
rho_s_val = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

Delta_T = Delta_T_contas.calculate_delta_T(
    T_a_em_k, 
    alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O, D_linha_w_val, R, k_linha_a_val),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    f=f
)
rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k) 

rho_vr_val = rho_vr.calcular_rho_vr(
    M_H2O, 
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    R, 
    T_a_em_k
)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val, a=17.502, b=240.97)

# Tempo de decaimento exponencial da temperatura 
tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps)

# Temperatura de equilíbrio
T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

def zeta(r_i):
    term1 = (f - 1)
    term2 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r_i)
    denominator = (4 * np.pi * rho_s_val * (r_i**3 / 3)) - m_s_val
    term3 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / denominator
    return term1 - term2 + term3

def dzeta_dr(r_i):
    term1 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * (r_i**2))
    denominator = (4 * np.pi * rho_s_val * (r_i**3)) / 3 - m_s_val
    term2 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl) * 4 * np.pi * rho_s_val * (r_i**2)) / (denominator**2)
    return term1 - term2

r0 = (1.1) * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1 / 3)

# Raio de equilíbrio
r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)

# Tempo de decaimento da exponencial do raio
tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

# Sistema de EDOs
def f_system(t, variaveis_gota):
    m, r, T = variaveis_gota

    # dm/dt  
    H_novo  = H_estrela.calcular_H_estrela(T, S)
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol     = (4/3) * np.pi * r**3
    C_gota  = m / vol
    dm_dt   = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))

    # dr/dt  
    rho_ww_novo  = m_s.calcular_rho_ww(T - 273.15)
    m_w_novo     = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w_novo)
    Phi_s_novo   = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(
        T, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo     = L_v.calcular_lv(T - 273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(
        T - 273.15, T, P, r, R, T0, P0,
        M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo  = rho_vr.calcular_rho_vr(
        M_H2O,
        e_sat_esp.calcular_esat(T - 273.15),
        code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R,
                                   v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
        R, T_a_em_k
    )
    Y    = ((2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r))
            - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val)))
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = (rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k)
            * (L_v_novo * M_H2O / (R * T_a_em_k) - 1))
    dr_dt = ((f - 1) - Y) / (r * (den1 + den2))

    # dT/dt 
    dT_dt = (3 / (rho_s_val * c_ps * r**2)
             * (k_linha_a_novo * (T_a_em_k - T)
                + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo)))

    return np.array([dm_dt, dr_dt, dT_dt])

# MÉTODO RK3 sem controle por subpassos (ECONTROL)
def metodo_rk3(f_system, y0, t):
    num_pontos    = len(t)
    num_variaveis = len(y0)
    y = np.zeros((num_pontos, num_variaveis))
    y[0] = y0

    for n in range(num_pontos - 1):
        dt = t[n + 1] - t[n]
        
        # Estágios do RK3
        k1 = f_system(t[n], y[n])
        k2 = f_system(t[n] + dt/2, y[n] + (dt/2) * k1)
        k3 = f_system(t[n] + dt, y[n] - dt * k1 + 2 * dt * k2)
        
        # Atualização
        y[n + 1] = y[n] + (dt/6) * (k1 + 4*k2 + k3)

    return y

# MÉTODO RK3 com controle por subpassos (E1)
def metodo_rk3_subpassos(f_system, y0, t, n_sub):

    # Número total de pontos no vetor de tempo (passos globais da simulação)
    num_pontos = len(t)

    """
    Matriz que armazena a solução do sistema ao longo do tempo.
    Cada linha corresponde a um instante de tempo t[n]
    Cada coluna corresponde a uma variável do sistema:
    coluna 0 - m
    coluna 1 - r
    coluna 2 - T
    """
    y = np.zeros((num_pontos, 3))

    # Condição inicial do sistema
    y[0] = y0

    for n in range(num_pontos - 1):
        """
        Passo de tempo global (Delta_t):
        usado para atualizar as variáveis lentas (m e r)
        """
        Dt  = t[n+1] - t[n]

        """
        # Subpasso de tempo (delta_T):
        usado para atualizar a variável rápida (T)
        em n_sub etapas dentro de um único passo global
        """
        dt  = Dt / n_sub       

        # Sistema no instante atual
        m_n, r_n, T_n = y[n]

        # ETAPA 1 - SUBPASSOS PARA T
        T_durante_subpassos = T_n

        for _ in range(n_sub):
            """
            Função que retorna apenas dT/dt (índice [2]),
            mantendo m e r fixos (congelados)            
            """
            def dT_dt(T_val):
                return f_system(0, np.array([m_n, r_n, T_val]))[2]
        
            # Estágios do RK3 para temperatura
            k1_T = dT_dt(T_durante_subpassos)
            k2_T = dT_dt(T_durante_subpassos + dt/2 * k1_T)
            k3_T = dT_dt(T_durante_subpassos - dt * k1_T + 2 * dt * k2_T)
            T_durante_subpassos = T_durante_subpassos + (dt/6) * (k1_T + 4*k2_T + k3_T)
        
        T_novo = T_durante_subpassos   

        # ETAPA 2 - PASSO GLOBAL PARA m e r (T_novo)
        def f_lento(m_val, r_val):
            return f_system(0, np.array([m_val, r_val, T_novo]))[[0, 1]]

        k1_m, k1_r = f_lento(m_n, r_n)

        m2 = m_n + (Dt / 2) * k1_m
        r2 = r_n + (Dt / 2) * k1_r
        k2_m, k2_r = f_lento(m2, r2)

        m3 = m_n - Dt * k1_m + 2 * Dt * k2_m
        r3 = r_n - Dt * k1_r + 2 * Dt * k2_r
        k3_m, k3_r = f_lento(m3, r3)

        y[n + 1, 0] = m_n + (Dt / 6) * (k1_m + 4 * k2_m + k3_m)   # m
        y[n + 1, 1] = r_n + (Dt / 6) * (k1_r + 4 * k2_r + k3_r)   # r
        y[n + 1, 2] = T_novo                                      # T


    return y

# Número de subpassos (n_sub)


