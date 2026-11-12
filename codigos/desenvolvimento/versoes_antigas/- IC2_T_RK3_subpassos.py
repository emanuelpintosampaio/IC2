import numpy as np
import math
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros fixos (ANDREAS)
# ----------------------------------------------------------------------
T_a = 18                     # °C
T_a_em_k = T_a + 273.15      # K
T_mar_em_k = 293.15          # K
T_gota = 19.8                # °C
T_gota_em_k = T_gota + 273.15
P = 1000                     # mb
M_H2O = 18.016e-3            # kg/mol
r_i = 100e-6                 # m
R = 8.31                     # J/(mol·K)
M_NaCl = 58.443e-3           # kg/mol
rho_w = 1025                 # kg/m³
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
c_ps = 4000                  # J/(kg·K)

# ----------------------------------------------------------------------
# Cálculos iniciais (necessários para definir tau_T)
# ----------------------------------------------------------------------
e_sat_val = e_sat.calcular_esat(T_a)
L_v_val = L_v.calcular_lv(T_gota)
D_linha_w_val = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O, T0, P0,
                                             alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R, T0, P0,
                                              M_a=28.9644e-3, alpha_T=0.7,
                                              delta_T=2.16e-7, c_pa=1.006e3)

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
    exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion,
                                    Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    f=f
)
rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k)

rho_vr_val = rho_vr.calcular_rho_vr(
    M_H2O,
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion,
                              Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    R,
    T_a_em_k
)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val,
                                                            a=17.502, b=240.97)

tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val,
                                     partial_rho_vr_val, L_v_val, c_ps)

# ----------------------------------------------------------------------
# Função derivada dT/dt (com raio fixo = r_i)
# ----------------------------------------------------------------------
def func_dTdt(T):
    """
    Calcula dT/dt para a gota, assumindo raio constante igual a r_i.
    T em Kelvin.
    """
    T_celsius = T - 273.15
    rho_ww_novo = m_s.calcular_rho_ww(T_celsius)
    m_w_novo = rho_s.calcular_massa_agua(r_i, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T_celsius, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r_i, M_H2O, T0, P0,
                                                  alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T_celsius)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T_celsius, T, P, r_i, R, T0, P0,
                                                   M_a=28.9644e-3, alpha_T=0.7,
                                                   delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo = rho_vr.calcular_rho_vr(
        M_H2O,
        e_sat_esp.calcular_esat(T_celsius),
        code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R, v_ion,
                                  Phi_s_novo, m_s_val, r_i, rho_s_val, M_NaCl),
        R, T_a_em_k
    )

    dT_dt = 3 / (rho_s_val * c_ps * r_i**2) * (
        k_linha_a_novo * (T_a_em_k - T) +
        L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo)
    )
    return dT_dt

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
# Controle de subpassos baseado em tau_T
# ----------------------------------------------------------------------
def evolui_temperatura_com_subpassos(T0_K, tempo, tau_T):
    """
    Resolve dT/dt = func_dTdt(T) usando SSP‑RK3 com subpassos.
    O número de subpassos em cada intervalo global é ajustado para que
    dt_sub ≤ tau_T.

    Parâmetros:
        T0_K : temperatura inicial (K)
        tempo : vetor de instantes de saída (global)
        tau_T : escala de tempo característica da temperatura (s)

    Retorna:
        y : array com a temperatura nos instantes dados (K)
    """
    n_global = len(tempo)
    y = np.zeros(n_global)
    y[0] = T0_K

    for i in range(n_global - 1):
        dt_global = tempo[i+1] - tempo[i]
        # Número de subpassos para garantir dt_sub ≤ tau_T
        n_sub = math.ceil(dt_global / tau_T)
        dt_sub = dt_global / n_sub

        T_atual = y[i]
        for _ in range(n_sub):
            T_atual = ssp_rk3_passo(func_dTdt, T_atual, dt_sub)
        y[i+1] = T_atual

    return y

# ----------------------------------------------------------------------
# Simulação
# ----------------------------------------------------------------------
dt_global_fixo = 100.0          # s
tempo_final = 1e3               # s
n_steps = int(tempo_final / dt_global_fixo)
tempo = np.linspace(0, tempo_final, n_steps)

T_inicial_K = T_gota_em_k
resultado_T = evolui_temperatura_com_subpassos(T_inicial_K, tempo, tau_T)

T_final_celsius = resultado_T[-1] - 273.15

print(f"Temperatura final: {resultado_T[-1]-273.15} ")

# ----------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------
plt.rcParams['text.usetex'] = False
fig, ax = plt.subplots(figsize=(8, 6))

ax.semilogx(tempo, resultado_T - 273.15, 'o-', color='#FF0000',
            linewidth=2.0, markersize=4,
            label=f'T_final = {T_final_celsius:.2f} °C')
ax.set_xlabel('Tempo (s)', fontsize=12)
ax.set_ylabel('Temperatura da Gota (°C)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_title(f'Evolução da Temperatura – SSP‑RK3 com Subpassos\n'
             f'dt_global = {dt_global_fixo} s | τ_T = {tau_T:.3e} s',
             fontsize=13)

plt.tight_layout()
plt.show()