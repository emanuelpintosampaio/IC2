import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from DGH import Dg_estrela, H_estrela
import matplotlib.pyplot as plt
import math

#IURY
T_a = 16.85           # Temperatura do ar em °C ANDREAS(18)
T_a_em_k = T_a+ 273.15       # Temperatura do ar em K ANDREAS(291.15)
T_mar_em_k = 285      # Temperatura do mar em k ANDREAS(293.15)
T_gota = 11.85        # Temperatura da gota em °C ANDREAS(20.2)
T_gota_em_k = T_gota+ 273.15     # Temperatura da gota em K ANDREAS(293.35)
P = 1000              # Pressão em mb
M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
r_i = 30e-6          # Raio em metros
R = 8.31              # Constante universal dos gases
M_NaCl = 58.443e-3    # Massa molecular de sal na água em kg/mol
rho_w = 1025          # Densidade da água do mar kg/m^3
v_ion = 2             # Número de íons por molécula de NaCl dissociada
s = 34/1000           # Salinidade fracionária (34 psu)
f = 0.8               # Umidade relativa fracionária ANDREAS(0.8)
C_ar =  0.0154        # Concentração do gás carbônico no ar
S = 34                # Salinidade do mar
R_atm = 0.082         # Constante universal dos gases
g = 9.81              # Gravidade
v_ar = 1.32e-5        # Viscosidade cinemática do ar
rho_ar = 1.225        # Densidade do ar
H_s = 6               # Altura significativa da onda 
T0 = 273.15
P0 = 1013.25
c_ps = 4000        # Calor específico da spray


def func(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    return U_f - const / a

def dfunc(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    d_a = -0.158 * (2 / 3) * (2 * r_i / v_ar) ** (2 / 3) * U_f ** (-1 / 3)
    return 1 - (const * d_a) / (a**2)

U_f0 = 0.1
U_f_sol = metodo_newton.Newton(func, dfunc, U_f0, 1e-6, 50)

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
    f=f)
rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k)

rho_vr_val = rho_vr.calcular_rho_vr(M_H2O,
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    R,T_a_em_k)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val, a=17.502, b=240.97)

tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps)

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

r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)

tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

# Condição inicial da massa 
vol_i = (4 / 3) * np.pi * r_i**3
H_i   = H_estrela.calcular_H_estrela(T_gota_em_k, S)
m_i   = vol_i * C_ar * H_i * R_atm * T_gota_em_k
y0    = np.array([r_i, T_gota_em_k, m_i])

# MÉTODOS COM SISTEMA DE EDO COMPLETO
def f_r_T_m(t, vars):
    r, T, m = vars

    rho_ww_novo    = m_s.calcular_rho_ww(T - 273.15)
    m_w_novo       = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo   = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w_novo)
    Phi_s_novo     = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0,
                                                   alpha_c=0.036, Delta_w=8e-8)
    L_v_novo       = L_v.calcular_lv(T - 273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T - 273.15, T, P, r, R, T0, P0,
                                                    M_a=28.9644e-3, alpha_T=0.7,
                                                    delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo    = rho_vr.calcular_rho_vr(
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
    den  = den1 + den2
    dr_dt = ((f - 1) - Y) / (r * den)

    dT_dt = (3 / (rho_s_val * c_ps * r**2)) * (
        k_linha_a_novo * (T_a_em_k - T)
        + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo)
    )

    H_nov  = H_estrela.calcular_H_estrela(T, S)
    Dg_nov = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol    = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    dm_dt  = 4 * np.pi * r * Dg_nov * (C_ar - C_gota / (H_nov * R_atm * T))

    return np.array([dr_dt, dT_dt, dm_dt])


# Completo 1: RK3 passo fixo 
def rk3_sistema(f_system, y0, t):
    num_pontos = len(t)
    y = np.zeros((num_pontos, len(y0)))
    y[0] = y0
    for n in range(num_pontos - 1):
        dt  = t[n + 1] - t[n]
        y_n = y[n];  t_n = t[n]
        F1 = f_system(t_n, y_n)
        y1 = y_n + dt * F1
        F2 = f_system(t_n + dt, y1)
        y2 = (3/4) * y_n + (1/4) * (y1 + dt * F2)
        F3 = f_system(t_n + dt / 2, y2)
        y[n + 1] = (1/3) * y_n + (2/3) * (y2 + dt * F3)
    return y

dt_fixo    = 1e-4
n_steps    = int(tau_f / dt_fixo)
tempo_fixo = np.linspace(0, tau_f, n_steps)

sol_fixo   = rk3_sistema(f_r_T_m, y0, tempo_fixo)
raio_fixo  = sol_fixo[:, 0]
temp_fixo  = sol_fixo[:, 1]
massa_fixo = sol_fixo[:, 2]


# Completo 2: RK3 passo adaptativo PID 
def rk3_adaptativo_completo(f_system, y0, t0, t_final, dt_min, dt_max,
                             dt_inicial, tol, K_P, K_I, K_D):
    t_hist = [t0];  y_hist = [y0];  dt_hist = [dt_inicial]
    dt = dt_inicial;  dt_prev = dt_min
    e_n_1 = tol;  e_n_2 = tol
    y_n = np.array(y0, dtype=float);  t_n = t0

    while t_n < t_final:
        if t_n + dt > t_final:
            dt = t_final - t_n

        F1 = f_system(t_n, y_n)
        y1 = y_n + dt * F1
        F2 = f_system(t_n + dt, y1)
        y2 = (3/4) * y_n + (1/4) * (y1 + dt * F2)
        F3 = f_system(t_n + dt / 2, y2)
        y_next = (1/3) * y_n + (2/3) * (y2 + dt * F3)

        e_r = abs(y_next[0] - y_n[0]) / abs(y_next[0])
        e_T = abs(y_next[1] - y_n[1]) / abs(y_next[1])
        e_m = abs(y_next[2] - y_n[2]) / abs(y_next[2])
        e_n = max(e_r, e_T, e_m)

        if e_n > tol and dt > dt_min:
            fator   = min(1 / e_n, 0.8)
            dt      = max(fator * dt, dt_min)
            dt_prev = (dt**2) / dt_prev
            continue

        t_n += dt;  y_n = y_next
        t_hist.append(t_n);  y_hist.append(y_n);  dt_hist.append(dt)

        fator_P = (e_n_1 / e_n)               ** K_P
        fator_I = (tol   / e_n)               ** K_I
        fator_D = (e_n_1**2 / (e_n * e_n_2))  ** K_D
        dt_next = fator_P * fator_I * fator_D * dt
        dt      = np.clip(dt_next, dt_min, dt_max)
        dt_prev = dt;  e_n_2 = e_n_1;  e_n_1 = e_n

    return np.array(t_hist), np.array(y_hist), np.array(dt_hist)

tempo_pid, sol_pid, _ = rk3_adaptativo_completo(
    f_r_T_m, y0, 0.0, tau_f,
    dt_min=1e-6, dt_max=1.0, dt_inicial=1e-4, tol=1e-4,
    K_P=0.075, K_I=0.175, K_D=0.01
)
raio_pid  = sol_pid[:, 0]
temp_pid  = sol_pid[:, 1]
massa_pid = sol_pid[:, 2]

# Completo 3: RK3 subcycling
def _grandezas_dinamicas(r, T):
    rho_ww_novo = m_s.calcular_rho_ww(T - 273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T - 273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T - 273.15, T, P, r, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo = rho_vr.calcular_rho_vr(M_H2O,
        e_sat_esp.calcular_esat(T - 273.15),
        code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R, v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
        R, T_a_em_k)

    Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r)) \
         - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val))
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2

    return k_linha_a_novo, D_linha_w_novo, L_v_novo, rho_vr_novo, Y, den

def f_rapido(t, T, r_fixo):
    k_linha_a_novo, D_linha_w_novo, L_v_novo, rho_vr_novo, Y, den = _grandezas_dinamicas(r_fixo, T)
    dT_dt = (3 / (rho_s_val * c_ps * r_fixo**2)) * (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo))
    return dT_dt

def f_lento(t, y_lento, T_final):
    r, m = y_lento
    k_linha_a_novo, D_linha_w_novo, L_v_novo, rho_vr_novo, Y, den = _grandezas_dinamicas(r, T_final)
    dr_dt = ((f - 1) - Y) / (r * den)
    H_novo = H_estrela.calcular_H_estrela(T_final, S)
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T_final))
    return np.array([dr_dt, dm_dt])

# RK3 escalar (para T) 
def rk3_step_scalar(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    y1 = y + dt * k1
    k2 = f(t + dt, y1, *args)
    y2 = (3/4) * y + (1/4) * (y1 + dt * k2)
    k3 = f(t + dt/2, y2, *args)
    return (1/3) * y + (2/3) * (y2 + dt * k3)

# RK3 vetorial (para [r, m]) 
def rk3_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    y1 = y + dt * k1
    k2 = f(t + dt, y1, *args)
    y2 = (3/4) * y + (1/4) * (y1 + dt * k2)
    k3 = f(t + dt/2, y2, *args)
    return (1/3) * y + (2/3) * (y2 + dt * k3)


# Algoritmo multiescala:
def passo_multiescala(r_n, T_n, m_n, t_n, H, M):
    dt = H / M

    # 1. Subpassos rápidos: avança T com r fixo
    T = T_n
    for j in range(M):
        t_sub = t_n + j * dt          
        T = rk3_step_scalar(f_rapido, t_sub, T, dt, r_n)

    T_new = T

    # 2. Passo macro lento: avança [r, m] com T já atualizado
    y_lento = np.array([r_n, m_n])
    y_lento_new = rk3_step(f_lento, t_n, y_lento, H, T_new)
    r_new, m_new = y_lento_new

    return r_new, T_new, m_new


def rk3_multiescala_completo(r0, T0, m0, t_final, H, M):
    t_list, r_list, T_list, m_list = [0], [r0], [T0], [m0]
    t, r, T, m = 0, r0, T0, m0

    while t < t_final:
        dt_macro = min(H, t_final - t)
        if dt_macro < 1e-15: break

        r, T, m = passo_multiescala(r, T, m, t, dt_macro, M) 
        t += dt_macro

        t_list.append(t)
        r_list.append(r)
        T_list.append(T)
        m_list.append(m)

    return np.array(t_list), np.array(r_list), np.array(T_list), np.array(m_list)

# Passo macro (lento: r e m)
dt_macro = 1e-3

# Número de subpassos rápidos (T) dentro de cada passo macro
M_sub   = 10

dt_micro = dt_macro/M_sub

tempo_sub, raio_sub, temp_sub, massa_sub = rk3_multiescala_completo(
    r_i, T_gota_em_k, m_i, tau_f, dt_macro, M_sub)


# MÉTODOS COM SIMPLIFICAÇÕES EXPONENCIAIS 
def r_simplificado(t):
    return r_eq + (r_i - r_eq) * np.exp(-t / tau_r)

def T_simplificado(t):
    return T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)

def f_m(t, m):
    r = r_simplificado(t);  T = T_simplificado(t)
    H_t   = H_estrela.calcular_H_estrela(T, S)
    Dg_t  = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol   = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    return 4 * np.pi * r * Dg_t * (C_ar - C_gota / (H_t * R_atm * T))

# Simplificado 1: RK3 passo fixo 
def rk3_escalar(f_m, m0, t):
    n  = len(t)
    mv = np.zeros(n);  mv[0] = m0
    for i in range(n - 1):
        dt  = t[i + 1] - t[i];  m_n = mv[i];  t_n = t[i]
        F1  = f_m(t_n, m_n);          m1 = m_n + dt * F1
        F2  = f_m(t_n + dt, m1);      m2 = (3/4)*m_n + (1/4)*(m1 + dt*F2)
        F3  = f_m(t_n + dt/2, m2)
        mv[i + 1] = (1/3)*m_n + (2/3)*(m2 + dt*F3)
    return mv

dt_simp = 1e-4
n_simp = int(tau_f / dt_simp)
tempo_simp = np.linspace(0, tau_f, n_simp)

massa_simp_fixo = rk3_escalar(f_m, m_i, tempo_simp)
raio_simp_fixo  = r_simplificado(tempo_simp)
temp_simp_fixo  = T_simplificado(tempo_simp)

# Simplificado 2: RK3 passo adaptativo PID 
def rk3_adaptativo_escalar(f_m, m0, t0, t_final, dt_min, dt_max,
                            dt_inicial, tol, K_P, K_I, K_D):
    t_hist = [t0];  m_hist = [m0];  dt_hist = [dt_inicial]
    dt = dt_inicial;  dt_prev = dt_min
    e_n_1 = tol;  e_n_2 = tol
    m_n = float(m0);  t_n = t0

    while t_n < t_final:
        if t_n + dt > t_final:
            dt = t_final - t_n

        F1 = f_m(t_n, m_n);           m1 = m_n + dt * F1
        F2 = f_m(t_n + dt, m1);       m2 = (3/4)*m_n + (1/4)*(m1 + dt*F2)
        F3 = f_m(t_n + dt/2, m2)
        m_next = (1/3)*m_n + (2/3)*(m2 + dt*F3)

        e_n = abs(m_next - m_n) / abs(m_next)

        if e_n > tol and dt > dt_min:
            fator = min(1 / e_n, 0.8)
            dt = max(fator * dt, dt_min)
            dt_prev = (dt**2) / dt_prev
            continue

        t_n += dt;  m_n = m_next
        t_hist.append(t_n);  m_hist.append(m_n);  dt_hist.append(dt)

        fator_P = (e_n_1 / e_n) ** K_P
        fator_I = (tol   / e_n) ** K_I
        fator_D = (e_n_1**2 / (e_n * e_n_2))  ** K_D
        dt_next = fator_P * fator_I * fator_D * dt
        dt      = np.clip(dt_next, dt_min, dt_max)
        dt_prev = dt;  e_n_2 = e_n_1;  e_n_1 = e_n

    return np.array(t_hist), np.array(m_hist), np.array(dt_hist)

tempo_spid, massa_simp_pid, _ = rk3_adaptativo_escalar(
    f_m, m_i, 0.0, tau_f,
    dt_min=1e-6, dt_max=1.0, dt_inicial=1e-4, tol=1e-4,
    K_P=0.075, K_I=0.175, K_D=0.01
)
raio_simp_pid = r_simplificado(tempo_spid)
temp_simp_pid = T_simplificado(tempo_spid)


# Simplificado 3: RK3 subcycling 
# Sistema RÁPIDO: T avaliado analiticamente no sub-tempo t
# Retorna T_new = T_simplificado(t + dt) diretamente (sem EDO)
def f_rapido(t, dt):
    return T_simplificado(t + dt)


# Sistema LENTO: apenas m como EDO
# r e T são avaliados analiticamente no tempo macro t_n + H
def f_lento(t, m, T_new):
    r = r_simplificado(t)
    H_t = H_estrela.calcular_H_estrela(T_new, S)
    Dg_t = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    return 4 * np.pi * r * Dg_t * (C_ar - C_gota / (H_t * R_atm * T_new))


# RK3 escalar (para m)
def rk3_step_scalar(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    y1 = y + dt * k1
    k2 = f(t + dt, y1, *args)
    y2 = (3/4) * y + (1/4) * (y1 + dt * k2)
    k3 = f(t + dt/2, y2, *args)
    return (1/3) * y + (2/3) * (y2 + dt * k3)


# Algoritmo multiescala (espelhando E1):
# 1. Subpassos rápidos: avança T analiticamente (M subpassos de dt = H/M)
# 2. Passo macro lento: avança m com RK3, usando T já atualizado e r analítico
def passo_multiescala(r_n, T_n, m_n, t_n, H, M):
    dt = H / M

    # 1. Subpassos rápidos: T é analítico, apenas avança o tempo interno
    #    T_new = T_simplificado avaliado no fim do passo macro
    T_new = f_rapido(t_n, H)

    # 2. Passo macro lento: avança m com RK3 usando T_new e r analítico
    m_new = rk3_step_scalar(f_lento, t_n, m_n, H, T_new)
    r_new = r_simplificado(t_n + H)

    return r_new, T_new, m_new


def rk3_multiescala_completo_s(r0, T0_val, m0, t_final, H, M):
    t_list   = [0]
    r_list   = [r0]
    T_list   = [T0_val]
    m_list   = [m0]
    t, r, T, m = 0, r0, T0_val, m0

    while t < t_final:
        dt_macro = min(H, t_final - t)
        if dt_macro < 1e-15:
            break

        r, T, m = passo_multiescala(r, T, m, t, dt_macro, M)
        t += dt_macro

        t_list.append(t)
        r_list.append(r)
        T_list.append(T)
        m_list.append(m)

    return np.array(t_list), np.array(r_list), np.array(T_list), np.array(m_list)
H_macro_s = 1e-3
M_sub_s   = 10
tempo_ssub, raio_simp_sub, temp_simp_sub, massa_simp_sub = rk3_multiescala_completo(
    r_i, T_gota_em_k, m_i, tau_f, dt_macro, M_sub)
raio_simp_sub = r_simplificado(tempo_ssub)
temp_simp_sub = T_simplificado(tempo_ssub)


# PID: {len(tempo_pid)}
# PID: {len(tempo_spid)}
print("Pontos:\n"
    f"  Completo     | Fixo: {len(tempo_fixo)}   Sub: {len(tempo_sub)}\n"
    f"  Simplificado | Fixo: {len(tempo_simp)}   Sub: {len(tempo_ssub)}")
# Gráficos
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size']   = 9

COR_SISTEMA_COMPLETO = '#0D00FF'   
COR_SISTEMA_SIMPLES  = "#FF0000" 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7,7), sharex=True)

#fig.suptitle(f'Sistema de EDO Completo vs Simplificações Exponenciais',
#    fontsize=11,
#    fontweight='bold')

#  Raio 
ax1.plot(tempo_fixo, raio_fixo * 1e6,'^-', color=COR_SISTEMA_COMPLETO, lw=2, ms=4,
         label=f'ECONTROL: r = {raio_fixo[-1]*1e6} µm')
#ax1.plot(tempo_pid,  raio_pid  * 1e6,'s-', color=COR_SISTEMA_COMPLETO,  lw=2, ms=4,
#         label=f'Completo   | PID: r={raio_pid[-1]*1e6} µm')
ax1.plot(tempo_sub,  raio_sub  * 1e6,'o-', color=COR_SISTEMA_COMPLETO,  lw=2, ms=3,
         label=f'E1: r = {raio_sub[-1]*1e6} µm')
ax1.plot(tempo_simp, raio_simp_fixo * 1e6, '^--', color=COR_SISTEMA_SIMPLES, lw=2, ms=4, alpha=0.7,
         label=f'E2: r = {raio_simp_fixo[-1]*1e6} µm')
#ax1.plot(tempo_spid, raio_simp_pid  * 1e6, 's--', color=COR_SISTEMA_SIMPLES,  lw=2, ms=4, alpha=0.7,
#         label=f'Simplific. | PID: r={raio_simp_pid[-1]*1e6} µm')
ax1.plot(tempo_ssub, raio_simp_sub  * 1e6, 'o--', color=COR_SISTEMA_SIMPLES,  lw=2, ms=3, alpha=0.7,
         label=f'E3: r = {raio_simp_sub[-1]*1e6} µm')
ax1.set_ylabel('Raio (µm)')
ax1.set_xscale('log')
ax1.ticklabel_format(axis='y', useOffset=False)
ax1.legend(fontsize=7.5, loc='best', ncol=2)
ax1.grid(True, alpha=0.3, which='both')

#  Temperatura 
ax2.plot(tempo_fixo, temp_fixo - 273.15,'^-', color=COR_SISTEMA_COMPLETO, lw=2, ms=4,
         label=f'ECONTROL: T={temp_fixo[-1]-273.15} °C')
#ax2.plot(tempo_pid,  temp_pid  - 273.15,'s-', color=COR_SISTEMA_COMPLETO,  lw=2, ms=4,
#         label=f'Completo   | PID: T={temp_pid[-1]-273.15} °C')
ax2.plot(tempo_sub,  temp_sub  - 273.15,'o-', color=COR_SISTEMA_COMPLETO,  lw=2, ms=3,
         label=f'E1: T = {temp_sub[-1]-273.15} °C')
ax2.plot(tempo_simp, temp_simp_fixo - 273.15, '^--', color=COR_SISTEMA_SIMPLES, lw=2, ms=4, alpha=0.7,
         label=f'E2: T = {temp_simp_fixo[-1]-273.15} °C')
#ax2.plot(tempo_spid, temp_simp_pid  - 273.15, 's--', color=COR_SISTEMA_SIMPLES,  lw=2, ms=4, alpha=0.7,
#         label=f'Simplific. | PID: T={temp_simp_pid[-1]-273.15} °C')
ax2.plot(tempo_ssub, temp_simp_sub  - 273.15, 'o--', color=COR_SISTEMA_SIMPLES,  lw=2, ms=3, alpha=0.7,
         label=f'E3: T = {temp_simp_sub[-1]-273.15} °C')
ax2.set_ylabel('Temperatura (°C)')
ax2.set_xscale('log')
ax2.legend(fontsize=7.5, loc='best', ncol=2)
ax2.grid(True, alpha=0.3, which='both')

#  Massa 
ax3.plot(tempo_fixo, massa_fixo,'^-', color=COR_SISTEMA_COMPLETO, lw=2, ms=4,
         label=f'ECONTROL: m = {massa_fixo[-1]} mol')
#ax3.plot(tempo_pid,  massa_pid,'s-', color=COR_SISTEMA_COMPLETO,  lw=2, ms=4,
#         label=f'Completo   | PID: m={massa_pid[-1]} mol')
ax3.plot(tempo_sub,  massa_sub,'o-', color=COR_SISTEMA_COMPLETO,  lw=2, ms=3,
         label=f'E1: m = {massa_sub[-1]} mol')
ax3.plot(tempo_simp, massa_simp_fixo,  '^--', color=COR_SISTEMA_SIMPLES, lw=2, ms=4, alpha=0.7,
         label=f'E2: m = {massa_simp_fixo[-1]} mol')
#ax3.plot(tempo_spid, massa_simp_pid,   's--', color=COR_SISTEMA_SIMPLES,  lw=2, ms=4, alpha=0.7,
#         label=f'Simplific. | PID: m={massa_simp_pid[-1]} mol')
ax3.plot(tempo_ssub, massa_simp_sub,   'o--', color=COR_SISTEMA_SIMPLES,  lw=2, ms=3, alpha=0.7,
         label=f'E3: m = {massa_simp_sub[-1]} mol')
ax3.set_ylabel('Massa (mol)')
ax3.set_xlabel('Tempo (s)')
ax3.set_xscale('log')
ax3.legend(fontsize=7.5, loc='best', ncol=2)
ax3.grid(True, alpha=0.3, which='both')

plt.tight_layout()
#plt.show()
plt.savefig("grafico_all_30.png")
