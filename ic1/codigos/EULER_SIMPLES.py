import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta,D_linha_w,e_sat,k_linha_a,L_v
from T_EQ.EXP_Y import code_exp_y,m_s,Phi_s,rho_s,sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr,rho_vr,contas_tau_T,e_sat_esp
from TAU_R.partial_rt_teste import calcular_tau_r
from rk4 import metodo_rk4
from GRAFICO.DGH import Dg_estrela, H_estrela  
import time

# Constantes  
T_a = 18           # Temperatura do ar em °C ANDREAS(18)
T_a_em_k = 291.15        # Temperatura do ar em K ANDREAS(291.15)
T_mar_em_k = 293.15      # Temperatura do mar em k ANDREAS(293.15)
T_gota = 19.8        # Temperatura da gota em °C ANDREAS(20.2)
T_gota_em_k = 293.35     # Temperatura da gota em K ANDREAS(293.35)
P = 1000              # Pressão em mb
M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
r_i = 100e-6          # Raio em metros
R = 8.31              # Constante universal dos gases
M_NaCl = 58.443e-3    # Massa molecular de sal na água em kg/mol
rho_w = 1025          # Densidade da água do mar kg/m^3
v_ion = 2             # Número de íons por molécula de NaCl dissociada
s = 34/1000           # Salinidade fracionária (34 psu)
f = 0.9               # Umidade relativa fracionária ANDREAS(0.8)
C_ar =  0.0154        # Concentração do gás carbônico no ar
S = 34                # Salinidade do mar
R_atm = 0.082         # Constante universal dos gases
g = 9.81              # Gravidade
v_ar = 1.32e-5        # Viscosidade cinemática do ar
rho_ar = 1.225        # Densidade do ar
H_s = 6               # Altura significativa da onda 
T0 = 273.15
P0 = 1013.25
c_ps = 4000           # Calor específico da spray

#-----------------------------------------------------------
# U_f PELO MÉTODO DE NEWTON Uf_n+1 = Uf_n - f(Uf_n)/f'(Uf_n)
#-----------------------------------------------------------

# f(Uf_n)
def func(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    return U_f - const / a

# f'(Uf_n)
def dfunc(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    d_a = -0.158 * (2/3) * (2 * r_i / v_ar)**(2/3) * U_f**(-1/3)
    return 1 - (const * d_a) / (a**2)

U_f0 = 0.1  # Chute inicial
U_f_sol = metodo_newton.Newton(func, dfunc, U_f0, 1e-6, 50) # Uf_n+1
tau_f = H_s / (2 * U_f_sol) # Tempo de vida da gota na atmosfesma

#--------------------------------------------
# PARÂMETROS PARA T_eq, r_eq, \tau_T e \tau_r
#--------------------------------------------

e_sat = e_sat.calcular_esat(T_a) # Pressão de vapor de saturação 
L_v = L_v.calcular_lv(T_gota) # Calor latente de vaporização da água
D_linha_w = D_linha_w.calculate_Dw_prime(T_gota_em_k, R,P, r_i, M_H2O , T0 , P0 , alpha_c = 0.036 , Delta_w = 8e-8) # Difusividade do vapor d'água ajustada para efeitos de não-continuidade
k_linha_a = k_linha_a.calculate_K_linha_a(T_gota,T_gota_em_k, P, r_i , R , T0 , P0, M_a = 28.9644e-3 , alpha_T = 0.7 , delta_T = 2.16e-7, c_pa = 1.006e3) # Condutividade térmica ajustada para efeitos de não-continuidade
rho_ww = m_s.calcular_rho_ww(T_gota) # Densidade da água pura em uma gota 
m_s = m_s.calcular_ms(rho_ww,r_i,s) # Massa de sal em uma gotícula de spray
v_a = rho_s.calcular_v_a(T_gota, m_s, M_NaCl, r_i) # Volume molar aparente de uma solução aquosa
m_w = rho_s.calcular_massa_agua(r_i, rho_ww) # Massa de água pura em uma gotícula de spray
sigma_s = sigma_s.calculate_sigma_s(T_gota, m_s,m_w) # Tensão superficial de uma superfície plana de água
Phi_s = Phi_s.calcular_phi_s(m_s, M_NaCl, m_w) # Coeficiente osmótico prático do NaCl dissolvido em água
rho_s = rho_s.calcular_rho_spray(rho_ww, m_s, m_w, v_a, M_NaCl) # Densidade da água do mar na gota
Delta_T = Delta_T_contas.calculate_delta_T(T_a_em_k, alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta= code_beta.calcular_beta(e_sat, T_a_em_k, L_v, M_H2O, D_linha_w, R, k_linha_a),
    b=240.97,exp_y=code_exp_y.calcular_exp_y(M_H2O , sigma_s , T_a_em_k , rho_w , R, v_ion , Phi_s , m_s , r_i , rho_s ,  M_NaCl),
    f=f)
rho_vr = rho_vr.calcular_rho_vr(M_H2O,e_sat_esp.calcular_esat(T_a),code_exp_y.calcular_exp_y(M_H2O , sigma_s , T_a_em_k , rho_w , R, v_ion , Phi_s , m_s , r_i , rho_s ,  M_NaCl),R,T_a_em_k) # É a densidade de vapor d’água na interface da gotícula
partial_rho_vr = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k,rho_vr,a=17.502,b=240.97)
tau_T = contas_tau_T.calculate_tau_T(rho_s , r_i , k_linha_a , D_linha_w ,partial_rho_vr ,L_v ,c_ps = 4000)
T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

#---------------------------------------------------------------
# r_eq PELO MÉTODO DE NEWTON req_n+1 = req_n - f(req_n)/f'(req_n)
#---------------------------------------------------------------

# f(req_n)
def zeta(r_i):
    term1 = (f - 1)
    term2 = (2 * M_H2O * sigma_s) / (R * T_eq_em_k * rho_w * r_i)
    denominator = (4 * np.pi * rho_s * (r_i**3 / 3)) - m_s
    term3 = (v_ion * Phi_s * m_s * (M_H2O / M_NaCl)) / denominator
    return term1 - term2 + term3

# f'(req_n)
def dzeta_dr(r_i):
    term1 = (2 * M_H2O * sigma_s) / (R * T_eq_em_k * rho_w * (r_i**2))
    denominator = (4 * np.pi * rho_s * (r_i**3)) / 3 - m_s
    term2 = (v_ion * Phi_s * m_s * (M_H2O / M_NaCl) * 4 * np.pi * rho_s * (r_i**2)) / (denominator**2)
    return term1 - term2

r0 = (1.1) * (3 * m_s / (4 * np.pi * rho_s)) ** (1 / 3) # chute inicial
r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)
tau_r = calcular_tau_r(f, M_H2O, sigma_s, R, T_a_em_k, rho_w, r_i, rho_s, m_s, v_ion, Phi_s, M_NaCl, D_linha_w, e_sat, L_v, k_linha_a, r_eq)[1]

#-----------------------
# DISCRETIZAÇÃO DO TEMPO
#-----------------------

# Lista de diferentes dt para testar
dts_test = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

print("\n" + "="*80)
print(f"SIMPLIFICAÇÕES - EULER")
print(f"r_i = {r_i*1e6} μm, H_s = {H_s} m")
print("="*80)
print(f"{'dt':<10} {'n_steps':<10} {'tempo_comp (s)':<25} {'massa_final (mol)':<25}")
print("-" * 80)

for dt_test in dts_test:
    n_steps_test = int(tau_f / dt_test) + 1
    tempo_test = np.linspace(0, tau_f, n_steps_test)
    
    # SIMULAÇÃO RT para dt atual
    inicio_rt = time.time()
    massa_rt_test = np.zeros(n_steps_test)
    massa_rt_test[0] = 1.2e-15
    
    for n in range(n_steps_test - 1):
        t = tempo_test[n + 1]
        r = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
        T = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
        H = H_estrela.calcular_H_estrela(T, S)
        Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
        C_gota = (3 * massa_rt_test[n]) / (4 * np.pi * (r**3))
        derivada = 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))
        massa_rt_test[n + 1] = massa_rt_test[n] + dt_test * derivada
    
    tempo_comp = time.time() - inicio_rt
    print(f"{dt_test:<10.0e} {n_steps_test:<10} {tempo_comp:<25.6f} {massa_rt_test[-1]:<25.6e}")

# Código original para dt = 1e-2 (mantido para comparação)
dt = 1e-2
n_steps = int(tau_f / dt)
tempo = np.linspace(0, tau_f, n_steps)
resultados = {}
tempo_computacional = {}

#------------------------------------
# ALGORITIMO MÉTODO DE EULER
#------------------------------------

# SIMULAÇÃO RT
inicio_rt = time.time()
massa_rt = np.zeros(n_steps)
massa_rt[0] = 1.2e-15
for n in range(n_steps - 1):
    t = tempo[n + 1]
    r = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
    T = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
    H = H_estrela.calcular_H_estrela(T, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    C_gota = (3*massa_rt[n]) / (4 * np.pi * (r**3))
    derivada = 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))
    massa_rt[n + 1] = massa_rt[n] + dt * derivada
resultados['RT'] = massa_rt
tempo_computacional['RT'] = time.time() - inicio_rt

# SIMULAÇÃO T
inicio_t = time.time()
massa_T = np.zeros(n_steps)
massa_T[0] = 1.2e-15
for n in range(n_steps - 1):
    t = tempo[n + 1]
    r = r_i
    T = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
    H = H_estrela.calcular_H_estrela(T, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    C_gota = (3*massa_T[n]) / (4 * np.pi * (r**3))
    derivada = 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))
    massa_T[n + 1] = massa_T[n] + dt * derivada
resultados['T'] = massa_T
tempo_computacional['T'] = time.time() - inicio_t

# SIMULAÇÃO R
inicio_r = time.time()
massa_R = np.zeros(n_steps)
massa_R[0] = 1.2e-15
for n in range(n_steps - 1):
    t = tempo[n + 1]
    r = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
    T = T_gota_em_k 
    H = H_estrela.calcular_H_estrela(T, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    C_gota = (3*massa_R[n]) / (4 * np.pi * (r**3))
    derivada = 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))
    massa_R[n + 1] = massa_R[n] + dt * derivada
resultados['R'] = massa_R
tempo_computacional['R'] = time.time() - inicio_r

# SIMULAÇÃO RT-Constantes
inicio_const = time.time()
massa_const = np.zeros(n_steps)
massa_const[0] = 1.2e-15
for n in range(n_steps - 1):
    t = tempo[n + 1]
    T = T_gota_em_k
    r = r_i
    H = H_estrela.calcular_H_estrela(T, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    C_gota = (3*massa_const[n]) / (4 * np.pi * (r**3))
    derivada = 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))
    massa_const[n + 1] = massa_const[n] + dt * derivada
resultados['RT-Constantes'] = massa_const
tempo_computacional['RT-Constantes'] = time.time() - inicio_const

#-----------
# RESULTADOS
#-----------

"""
print("\n" + "="*80)
print(f"EULER - r_i = {r_i*1e6} - H_s = {H_s}")
print("="*80)
print(f"\n{'SIMULAÇÃO':<15} {'MASSA FINAL (mol)':<25} {'TEMPO COMPUTACIONAL (s)':<25} {'N° PASSOS':<10}")
print("-" * 80)
    
for metodo in ['RT', 'T', 'R', 'RT-Constantes']:
    massa_final = resultados[metodo][-1]
    tempo_comp = tempo_computacional[metodo]
    print(f"{metodo:<15} {massa_final:<25.15e} {tempo_comp:<25.4f} {n_steps:<10}")
"""