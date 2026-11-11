import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "modulos"))
import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from DGH import Dg_estrela, H_estrela
import matplotlib.pyplot as plt
import time

# CONSTANTES

# Andreas
T_a = 18           # Temperatura do ar em °C ANDREAS(18)
T_a_em_k = T_a + 273.15  # Temperatura do ar em K ANDREAS(291.15)
T_mar_em_k = 293.15  # Temperatura do mar em K ANDREAS(293.15)
T_gota = 19.8      # Temperatura da gota em °C ANDREAS(20.2)
T_gota_em_k = T_gota + 273.15  # Temperatura da gota em K ANDREAS(293.35)
P = 1000           # Pressão em mb
M_H2O = 18.016e-3  # Massa molecular da água em kg/mol
r_i = 30e-6       # Raio em metros
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


# 
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
Phi_s_val = Phi_s.calcular_phi_s(m_s_val, M_H2O, m_w)
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

# DISCRETIZAÇÃO DO TEMPO
dt = 1e-3
n_steps = int(tau_f / dt) 
tempo = np.linspace(0, tau_f, n_steps) 
resultados = {}
tempo_computacional = {}

# Condição Inicial da massa
vol_i = (4 / 3) * np.pi * r_i**3
H_i = H_estrela.calcular_H_estrela(T_mar_em_k, S)
m_i = vol_i * C_ar * H_i * R_atm * T_mar_em_k
#print(f"m_i = {m_i}")

# Evoulução do Raio e da Temperatura por decaimento de exponenciais
def raio_variavel(t):
    return r_eq + (r_i - r_eq) * np.exp(-t / tau_r)

def temperatura_variavel(t):
    return T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)

def raio_constante(t):
    return r_i

def temperatura_constante(t):
    return T_mar_em_k

# Método de EULER EXPLÍCITO
# m_{n+1} = m_n + dt * dm/dt(t_n, m_n)

def simular_massa(raio, temperatura):
    massa = np.zeros(n_steps)
    massa[0] = m_i
    for n in range(n_steps - 1):
        dt_n = tempo[n + 1] - tempo[n]
        t = tempo[n]
        r = raio(t)
        T = temperatura(t)
        H = H_estrela.calcular_H_estrela(T, S)
        Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
        C_gota = massa[n] / ((4 / 3) * np.pi * r**3)
        derivada = 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))
        massa[n + 1] = massa[n] + dt_n * derivada
        #print(massa[n+1])
    return massa

simulacoes = {
    'RT': (raio_variavel, temperatura_variavel),               # E-RT: r e T variando
    'T': (raio_constante, temperatura_variavel),               # E-T: somente T variando
    'R': (raio_variavel, temperatura_constante),               # E-R: somente r variando
    'RT-Constantes': (raio_constante, temperatura_constante),  # E-CONTROL: r e T constantes
}

for metodo, (raio, temperatura) in simulacoes.items():
    inicio = time.time()
    resultados[metodo] = simular_massa(raio, temperatura)
    tempo_computacional[metodo] = time.time() - inicio

massa_rt = resultados['RT']
massa_T = resultados['T']
massa_R = resultados['R']
massa_const = resultados['RT-Constantes']

# RESULTADOS

print(f"EULER EXPLÍCITO | r_i = {r_i*1e6} | H_s = {H_s} | dt = {dt:.0e}")
print(f"\n{'SIMULAÇÃO':<15} {'m(τ_f) em mol':<25} {'TEMPO COMPUTACIONAL (s)':<25} {'N° PASSOS':<10}")
print("-" * 80)
    
for metodo in ['RT', 'T', 'R', 'RT-Constantes']:
    massa_final = resultados[metodo][-1]
    tempo_comp = tempo_computacional[metodo]
    print(f"{metodo:<15} {massa_final:<25.15e} {tempo_comp:<25.4f} {n_steps:<10}")

# GRÁFICO
plt.figure(figsize=(8, 4))
plt.title(f"$r_i = {r_i*1e6:.0f}\\mu m$ | $H_s = {H_s}\\,m$ ", fontsize=16)
plt.axhline(m_i, color='k', linestyle='--', linewidth=0.5,
            label=f'$m_i$ = {m_i:.3e} mol')
plt.plot(tempo, resultados['RT'], "#2B00FF",linestyle='-', linewidth=2.5, 
             label=f'E-RT: m($\\tau_f$) = {massa_rt[-1]:.3e} mol')
plt.plot(tempo, resultados['T'], '#FF0000',linestyle='-', linewidth=2.5, 
             label=f'E-T: m($\\tau_f$) = {massa_T[-1]:.3e} mol')
plt.plot(tempo, resultados['R'], '#FFD700',linestyle='-', linewidth=2.5, 
             label=f'E-R: m($\\tau_f$) = {massa_R[-1]:.3e} mol')
plt.plot(tempo, resultados['RT-Constantes'], "#00FF00BF",linestyle='-', linewidth=2.5, 
             label=f'E-CONTROL: m($\\tau_f$) = {massa_const[-1]:.3e} mol')
plt.xlabel('Tempo (s)', fontsize=14)
plt.ylabel('Massa (mol)', fontsize=14)
plt.xscale('log')
plt.legend(fontsize=16)  
plt.tight_layout()
plt.show()
