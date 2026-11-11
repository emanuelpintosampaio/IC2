import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parâmetros globais
# ---------------------------------------------------------------------------
T_a         = 18
T_a_em_k    = T_a + 273.15
T_mar_em_k  = 293.15
T_gota      = 19.8
T_gota_em_k = T_gota + 273.15
P           = 1000
M_H2O       = 18.016e-3
R           = 8.31
M_NaCl      = 58.443e-3
rho_w       = 1025
v_ion       = 2
s           = 34 / 1000
f           = 0.9
C_ar        = 0.0154
S           = 34
R_atm       = 0.082
g           = 9.81
v_ar        = 1.32e-5
rho_ar      = 1.225
H_s         = 6
T0          = 273.15
P0          = 1013.25
c_ps        = 4000
dt_fixo     = 1e-5
tempo_final = 1e3

# ---------------------------------------------------------------------------
# Lista de raios (1000 elementos)
# ---------------------------------------------------------------------------
raios_um = np.logspace(np.log10(10), np.log10(1000), 1000).tolist()
raios_m  = [r * 1e-6 for r in raios_um]

# ---------------------------------------------------------------------------
# Loop sobre os raios
# ---------------------------------------------------------------------------
all_results = []

print(f"Iniciando cálculo sequencial com {len(raios_m)} raios...")

for idx, r_i in enumerate(raios_m):
    # --- Velocidade terminal ---
    def func(U_f, r=r_i):
        const = (2 * r**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
        a = 1 + 0.158 * ((2 * r * U_f / v_ar) ** (2 / 3))
        return U_f - const / a

    def dfunc(U_f, r=r_i):
        const = (2 * r**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
        a = 1 + 0.158 * ((2 * r * U_f / v_ar) ** (2 / 3))
        d_a = -0.158 * (2 / 3) * (2 * r / v_ar) ** (2 / 3) * U_f ** (-1 / 3)
        return 1 - (const * d_a) / (a ** 2)

    U_f_sol = metodo_newton.Newton(func, dfunc, 0.1, 1e-6, 50)
    tau_f   = H_s / (2 * U_f_sol)

    # --- Propriedades termodinâmicas ---
    e_sat_val     = e_sat.calcular_esat(T_a)
    L_v_val       = L_v.calcular_lv(T_gota)
    D_linha_w_val = D_linha_w.calculate_Dw_prime(
        T_gota_em_k, R, P, r_i, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    k_linha_a_val = k_linha_a.calculate_K_linha_a(
        T_gota, T_gota_em_k, P, r_i, R, T0, P0,
        M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)

    rho_ww      = m_s.calcular_rho_ww(T_gota)
    m_s_val     = m_s.calcular_ms(rho_ww, r_i, s)
    v_a         = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
    m_w         = rho_s.calcular_massa_agua(r_i, rho_ww)
    sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
    Phi_s_val   = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w) 
    rho_s_val   = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

    exp_y_val = code_exp_y.calcular_exp_y(
        M_H2O, sigma_s_val, T_a_em_k, rho_w, R,
        v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl)

    Delta_T = Delta_T_contas.calculate_delta_T(
        T_a_em_k,
        alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
        beta=code_beta.calcular_beta(
            e_sat_val, T_a_em_k, L_v_val, M_H2O, D_linha_w_val, R, k_linha_a_val),
        b=240.97,
        exp_y=exp_y_val,
        f=f
    )

    rho_vr_val = rho_vr.calcular_rho_vr(
        M_H2O, e_sat_esp.calcular_esat(T_a), exp_y_val, R, T_a_em_k)

    partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(
        T_a_em_k, rho_vr_val, a=17.502, b=240.97)

    tau_T = contas_tau_T.calculate_tau_T(
        rho_s_val, r_i, k_linha_a_val, D_linha_w_val,
        partial_rho_vr_val, L_v_val, c_ps)

    T_eq      = Delta_T[0] + T_a
    T_eq_em_k = T_eq + 273.15

    # --- Raio de equilíbrio ---
    def zeta(r):
        term1       = f - 1
        term2       = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r)
        denominator = (4 * np.pi * rho_s_val * (r**3 / 3)) - m_s_val
        term3       = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / denominator
        return term1 - term2 + term3

    def dzeta_dr(r):
        term1       = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r**2)
        denominator = (4 * np.pi * rho_s_val * r**3) / 3 - m_s_val
        term2       = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)
                        * 4 * np.pi * rho_s_val * r**2) / denominator**2
        return term1 - term2

    r0   = 1.1 * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1 / 3)
    r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)

    tau_r = calcular_tau_r(
        f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i,
        rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl,
        D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

    # --- Série temporal ---
    n_steps               = int(tau_f / dt_fixo)
    if n_steps < 2: n_steps = 2
    
    tempo                 = np.linspace(0, tau_f, n_steps)
    temperatura_analitica = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-tempo / tau_T)
    temperatura_final     = temperatura_analitica[-1]

    # Print progresso (a cada 250)
    if idx % 250 == 0:
        print(f"Processando r_i = {raios_um[idx]:.1f} µm ({idx}/{len(raios_m)})")

    all_results.append({
        "r_um":    raios_um[idx],
        "tempo":   tempo,
        "temp_C":  temperatura_analitica - 273.15,
        "T_final": temperatura_final - 273.15,
        "tau_T":   tau_T,
        "T_eq":    T_eq,
    })

print("Cálculo sequencial concluído.")

# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------
# Ordenar resultados
all_results.sort(key=lambda x: x['r_um'])

# Plotar exemplos representativos
indices_exemplo = [0, len(all_results)//3, 2*len(all_results)//3, -1]
cores  = ['#2B00FF', '#FF0000', '#FFD700', '#00FF00BF']
estilos = ['-', '-', '-', '-']

plt.rcParams['text.usetex'] = False
fig, ax = plt.subplots(figsize=(9, 6))

for i, idx in enumerate(indices_exemplo):
    if idx >= len(all_results): continue
    res = all_results[idx]
    label = (f"r_i = {res['r_um']:.1f} µm | "
             f"T_final = {res['T_final']:.4f} °C")
    ax.plot(res['tempo'], res['temp_C'],
            linestyle=estilos[i], color=cores[i],
            linewidth=2.0, label=label)

ax.set_xlabel('Tempo (s)', fontsize=12)
ax.set_ylabel('Temperatura da Gota (°C)', fontsize=12)
ax.set_xscale('log')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.set_title(
    f'Evolução da Temperatura — 1000 raios (dt fixo = {dt_fixo})',
    fontsize=13)

plt.tight_layout()
plt.savefig('temperatura_multi_raios_seq.png', dpi=150)
plt.show()