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

# =====================================================
# Constantes e parâmetros (mantidos do código original)
# =====================================================
T_a = 18
T_a_em_k = T_a + 273.15
T_mar_em_k = 293.15
T_gota = 19.8
T_gota_em_k = T_gota + 273.15
P = 1000
M_H2O = 18.016e-3
r_i = 100e-6
R = 8.31
M_NaCl = 58.443e-3
rho_w = 1025
v_ion = 2
s = 34 / 1000
f = 0.9
C_ar = 0.0154
S = 34
R_atm = 0.082
g = 9.81
v_ar = 1.32e-5
rho_ar = 1.225
H_s = 6
T0 = 273.15
P0 = 1013.25
c_ps = 4000

# =====================================================
# Funções auxiliares originais (para r, T, etc.)
# =====================================================
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

# Propriedades de referência (constantes para o problema)
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

# =====================================================
# Funções auxiliares para o novo sistema (massa + T)
# =====================================================
def calcular_raio(m_total, T_K):
    """
    Calcula o raio da gota a partir da massa total e temperatura.
    m_total = m_agua + m_sal, com m_sal constante.
    Volume de água = (m_total - m_sal) / densidade_agua(T)
    """
    m_w = m_total - m_s_val
    rho_ww_atual = m_s.calcular_rho_ww(T_K - 273.15)  # função espera °C
    r = (3 * m_w / (4 * np.pi * rho_ww_atual)) ** (1/3)
    return r

def func_dmdt(m, T_K):
    """
    Taxa de variação da massa total usando a equação de transferência de massa
    fornecida no comentário.
    """
    r = calcular_raio(m, T_K)
    # Calcular H* e Dg* (funções importadas)
    H_novo = H_estrela.calcular_H_estrela(T_K - 273.15, S)   # H_estrela espera T em °C?
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4/3) * np.pi * r**3
    C_gota = m / vol
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T_K))
    return dm_dt

def func_dTdt(m, T_K):
    """
    Taxa de variação da temperatura (mesma física do código original).
    """
    r = calcular_raio(m, T_K)
    T_C = T_K - 273.15
    # Recalcular propriedades dependentes de T e r
    rho_ww_novo    = m_s.calcular_rho_ww(T_C)
    m_w_novo       = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo   = sigma_s.calculate_sigma_s(T_C, m_s_val, m_w_novo)
    Phi_s_novo     = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T_K, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo       = L_v.calcular_lv(T_C)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T_C, T_K, P, r, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo    = rho_vr.calcular_rho_vr(
        M_H2O,
        e_sat_esp.calcular_esat(T_C),
        code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R, v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
        R, T_a_em_k
    )
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * (
        k_linha_a_novo * (T_a_em_k - T_K) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo)
    )
    return dT_dt

# =====================================================
# Método SSP-RK3 para uma única variável
# =====================================================
def ssp_rk3_passo(func, y_n, param_aux, dt, modo='m'):
    """
    Executa um passo SSP-RK3 para a equação dy/dt = func(y, param_aux).
    Se modo='m', param_aux = T (fixo) e func = func_dmdt.
    Se modo='T', param_aux = m (fixo) e func = func_dTdt.
    """
    if modo == 'm':
        # y = m, param_aux = T
        F_n   = func(y_n, param_aux)
        y_1   = y_n + dt * F_n
        F_1   = func(y_1, param_aux)
        y_2   = (3/4)*y_n + (1/4)*(y_1 + dt * F_1)
        F_2   = func(y_2, param_aux)
        y_np1 = (1/3)*y_n + (2/3)*(y_2 + dt * F_2)
    else:  # modo == 'T'
        # y = T, param_aux = m
        F_n   = func(param_aux, y_n)
        y_1   = y_n + dt * F_n
        F_1   = func(param_aux, y_1)
        y_2   = (3/4)*y_n + (1/4)*(y_1 + dt * F_1)
        F_2   = func(param_aux, y_2)
        y_np1 = (1/3)*y_n + (2/3)*(y_2 + dt * F_2)
    return y_np1

# =====================================================
# Splitting RK3 para (m, T) com subpassos adaptativos
# =====================================================
def rk3_splitting_mT(m0, T0_K, t_array, tau_T, tau_m):
    num_pontos = len(t_array)
    y = np.zeros((num_pontos, 2))  # [m, T]
    y[0] = [m0, T0_K]

    for n in range(num_pontos - 1):
        dt_global = t_array[n+1] - t_array[n]
        m_n, T_n = y[n]

        # Subpassos para T (processo rápido)
        n_sub_T = math.ceil(dt_global / tau_T)
        dt_sub_T = dt_global / n_sub_T
        T_atual = T_n
        for _ in range(n_sub_T):
            T_atual = ssp_rk3_passo(func_dTdt, T_atual, m_n, dt_sub_T, modo='T')
        T_novo = T_atual

        # Subpassos para m (processo rápido, usando T_novo)
        n_sub_m = math.ceil(dt_global / tau_m)
        dt_sub_m = dt_global / n_sub_m
        m_atual = m_n
        for _ in range(n_sub_m):
            m_atual = ssp_rk3_passo(func_dmdt, m_atual, T_novo, dt_sub_m, modo='m')
        m_novo = m_atual

        y[n+1] = [m_novo, T_novo]

    return y, n_sub_T, n_sub_m

# =====================================================
# Preparação e execução da simulação
# =====================================================
# Massa inicial
m0 = m_w + m_s_val   # massa total inicial

# Escala de tempo para massa (estimativa inicial)
dm0 = abs(func_dmdt(m0, T_gota_em_k))
tau_m0 = abs(m0 / dm0) if dm0 > 0 else 1e6

# Parâmetros da simulação
dt_global_fixo = 100
tempo_final = 1e3
n_steps = int(tempo_final / dt_global_fixo)
tempo = np.linspace(0, tempo_final, n_steps)

# Executar splitting
resultados, n_sub_T_usado, n_sub_m_usado = rk3_splitting_mT(
    m0, T_gota_em_k, tempo, tau_T, tau_m0
)

m_evolucao = resultados[:, 0]
T_evolucao = resultados[:, 1]

# Calcular raio a partir da massa e temperatura para cada instante
raio_evolucao = np.array([calcular_raio(m, T) for m, T in zip(m_evolucao, T_evolucao)])

# Impressão de resultados
print(f"Passo global             : {dt_global_fixo} s")
print(f"Subpassos de T (n_sub_T) : {n_sub_T_usado}  (τ_T ≈ {tau_T} s)")
print(f"Subpassos de m (n_sub_m) : {n_sub_m_usado}  (τ_m ≈ {tau_m0} s)")
print(f"Raio final               : {raio_evolucao[-1] * 1e6} µm")
print(f"Temperatura final        : {T_evolucao[-1] - 273.15} °C")
print(f"Massa final              : {m_evolucao[-1] * 1e3} mg")

# Gráficos
plt.rcParams['text.usetex'] = False
fig, axs = plt.subplots(2, 1, figsize=(10, 13), sharex=True)

fig.suptitle(
    f'Evolução da gota (massa → raio) e temperatura - RK3 - Subpassos Independentes\n'
    f'dt_global = {dt_global_fixo} s | n_sub_T = {n_sub_T_usado} | n_sub_m = {n_sub_m_usado} | {len(tempo)} pontos',
    fontsize=13
)

axs[0].semilogx(tempo, raio_evolucao * 1e6, 'o-', color='#0D00FF', linewidth=2.0,
                label=f'r_final = {raio_evolucao[-1]*1e6} µm')
axs[0].set_ylabel('Raio da Gota (µm)', fontsize=12)
axs[0].grid(True, alpha=0.3)
axs[0].legend(fontsize=10)

axs[1].semilogx(tempo, T_evolucao - 273.15, 'o-', color='#FF0000', linewidth=2.0,
                label=f'T_final = {T_evolucao[-1]-273.15} °C')
axs[1].set_xlabel('Tempo (s)', fontsize=12)
axs[1].set_ylabel('Temperatura da Gota (°C)', fontsize=12)
axs[1].grid(True, alpha=0.3)
axs[1].legend(fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()