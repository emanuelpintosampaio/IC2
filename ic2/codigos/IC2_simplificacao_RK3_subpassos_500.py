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
r_i = 500e-6          # Raio em metros
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

# Simplificações analíticas para r(t) e T(t)
def r_simplificado(t):
    return r_eq + (r_i - r_eq) * np.exp(-t / tau_r)

def T_simplificado(t):
    return T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)


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


def rk3_multiescala_completo(r0, T0_val, m0, t_final, H, M):
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


# Condições iniciais para massa
vol_i = (4/3) * np.pi * r_i**3
H_i   = H_estrela.calcular_H_estrela(T_gota_em_k, S)
m_i   = vol_i * C_ar * H_i * R_atm * T_gota_em_k

# Passo macro (lento: r e m)
dt_macro = 1e-3

# Número de subpassos rápidos (T) dentro de cada passo macro
M_sub   = 10

dt_micro = dt_macro / M_sub

# Solução com RK3 multiescala
t_mr, raio_simp, temperatura_simp, massa_rk3_sub = rk3_multiescala_completo(
    r_i, T_gota_em_k, m_i, tau_f, dt_macro, M_sub)

print(f'raio inicial: {r_i}')
print(f'massa final: {massa_rk3_sub[-1]}')

#"""
# Gráficos
plt.rcParams['text.usetex'] = False
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

fig.suptitle(
    f'dt_macro = {dt_macro:.0e} s | M_sub = {M_sub} | dt_micro = {dt_micro:.0e} s | {len(t_mr)} pontos',
    fontsize=13)

# Raio
ax1.plot(t_mr, raio_simp * 1e6, 'o-', color='#FF0000', linewidth=2, markersize=4,
         label=f'r_final = {raio_simp[-1] * 1e6} µm')
ax1.set_ylabel('Raio da Gota (µm)', fontsize=12)
ax1.set_xscale('log')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Temperatura
ax2.plot(t_mr, temperatura_simp - 273.15, 'o-', color='#FF0000', linewidth=2,
         markersize=4, label=f'T_final = {temperatura_simp[-1] - 273.15} °C')
ax2.set_ylabel('Temperatura da Gota (°C)', fontsize=12)
ax2.set_xscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Massa
ax3.semilogx(t_mr, massa_rk3_sub, 'o-', color='#FF0000', linewidth=2,
             markersize=4, label=f'Massa final: {massa_rk3_sub[-1]} mol')
ax3.set_xlabel('Tempo (s)', fontsize=12)
ax3.set_ylabel('Massa (mol)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
 
plt.tight_layout()
#plt.show()
plt.savefig("grafico_subpassos_simplificacao_500.png")


