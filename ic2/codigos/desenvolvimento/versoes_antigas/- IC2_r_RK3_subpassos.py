import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import matplotlib.pyplot as plt
import math

#ANDREAS
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

# ----------------------------------------------------------------------
# Função derivada dr/dt com temperatura fixa (T_eq)
# ----------------------------------------------------------------------
def func_drdt_Tfixed(r):
    """
    Calcula dr/dt para a gota, mantendo a temperatura constante em T_eq_em_k.
    """
    T_celsius = T_eq - 273.15   # T_eq já é °C, mas cuidado: T_eq = Delta_T[0] + T_a
    # Na realidade T_eq está em °C, mas para cálculos que precisam de K usamos T_eq_em_k.
    # As funções abaixo usam T em K ou °C conforme necessário.

    # Propriedades dependentes da temperatura (fixa)
    rho_ww_novo = m_s.calcular_rho_ww(T_eq)          # T_eq em °C
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T_eq, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T_eq_em_k, R, P, r, M_H2O,
                                                  T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T_eq)                 # T_eq em °C
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T_eq, T_eq_em_k, P, r, R, T0, P0,
                                                   M_a=28.9644e-3, alpha_T=0.7,
                                                   delta_T=2.16e-7, c_pa=1.006e3)

    # Termos Y e denominador
    Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r)) \
        - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val))

    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2

    dr_dt = ((f - 1) - Y) / (r * den)
    return dr_dt

# ----------------------------------------------------------------------
# SSP‑RK3 para um passo (Shu & Osher, 1988)
# ----------------------------------------------------------------------
def ssp_rk3_passo(func, y_n, dt):
    """
    Avança a solução da EDO dy/dt = func(y) por um passo dt
    usando o método SSP‑RK3 de terceira ordem.
    """
    F_n   = func(y_n)
    y_1   = y_n + dt * F_n
    F_1   = func(y_1)
    y_2   = (3/4) * y_n + (1/4) * (y_1 + dt * F_1)
    F_2   = func(y_2)
    y_np1 = (1/3) * y_n + (2/3) * (y_2 + dt * F_2)
    return y_np1

# ----------------------------------------------------------------------
# Controle de subpassos baseado em tau_r
# ----------------------------------------------------------------------
def evolui_raio_com_subpassos(r0_m, tempo, tau_r):
    """
    Resolve dr/dt = func_drdt_Tfixed(r) usando SSP‑RK3 com subpassos.
    O número de subpassos em cada intervalo global é ajustado para que
    dt_sub ≤ tau_r.

    Parâmetros:
        r0_m : raio inicial (m)
        tempo : vetor de instantes de saída (global)
        tau_r : escala de tempo característica do raio (s)

    Retorna:
        y : array com o raio nos instantes dados (m)
    """
    n_global = len(tempo)
    y = np.zeros(n_global)
    y[0] = r0_m

    for i in range(n_global - 1):
        dt_global = tempo[i+1] - tempo[i]
        n_sub = math.ceil(dt_global / tau_r)
        dt_sub = dt_global / n_sub

        r_atual = y[i]
        for _ in range(n_sub):
            r_atual = ssp_rk3_passo(func_drdt_Tfixed, r_atual, dt_sub)
        y[i+1] = r_atual

    return y

# ----------------------------------------------------------------------
# Simulação
# ----------------------------------------------------------------------
dt_global_fixo = 100.0          # s
tempo_final = 1e3               # s
n_steps = int(tempo_final / dt_global_fixo)
tempo = np.linspace(0, tempo_final, n_steps)

raio_inicial = r_i
resultado_raio = evolui_raio_com_subpassos(raio_inicial, tempo, tau_r)

raio_final_um = resultado_raio[-1] * 1e6   # em µm

print(f"tau_r                         : {tau_r:.3e} s")
print(f"Passo global                  : {dt_global_fixo} s")
print(f"Número de passos globais      : {n_steps}")
print(f"Raio final                    : {raio_final_um} µm")

# ----------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------
plt.rcParams['text.usetex'] = False
fig, ax = plt.subplots(figsize=(8, 6))

ax.semilogx(tempo, resultado_raio * 1e6, 'o-', color='#0D00FF',
            linewidth=2.0, markersize=4,
            label=f'r_final = {raio_final_um:.2f} µm')
ax.set_xlabel('Tempo (s)', fontsize=12)
ax.set_ylabel('Raio da Gota (µm)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_title(f'Evolução do Raio com Subpassos\n'
             f'dt_global = {dt_global_fixo} s | τ_r = {tau_r:.3e} s',
             fontsize=13)

plt.tight_layout()
plt.show()