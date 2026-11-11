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

# Constantes IURY 
# Constantes IURY 
T_a = 18           # Temperatura do ar em °C ANDREAS(18) IRUY(16.85)
T_a_em_k = 291.15        # Temperatura do ar em K ANDREAS(291.15) IURY(290)
T_mar_em_k = 293.15      # Temperatura do mar em k ANDREAS(293.15) IURY(285)
T_gota = 19.8        # Temperatura da gota em °C ANDREAS(19.8) IURY(11.85)
T_gota_em_k = 292.95     # Temperatura da gota em K ANDREAS(292.95) IURY(285)
P = 1000              # Pressão em mb
M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
r_i = 30e-6           # Raio inicial ANDREAS(100) IURY(30)
R = 8.31              # Constante universal dos gases
M_NaCl = 58.443e-3    # Massa molecular de sal na água em kg/mol
rho_w = 1025          # Densidade da água do mar kg/m^3
v_ion = 2             # Número de íons por molécula de NaCl dissociada
s = 34/1000           # Salinidade fracionária (34 psu)
f = 0.9               # Umidade relativa fracionária ANDREAS(0.9) IURY(0.8)
C_ar =  0.0154        # Concentração do gás carbônico no ar
S = 34                # Salinidade do mar
R_atm = 0.082         # Constante universal dos gases
g = 9.81              # Gravidade
v_ar = 1.32e-5        # Viscosidade cinemática do ar
rho_ar = 1.225        # Densidade do ar
H_s = 6               # Altura significativa da onda 
T0 = 273.15
P0 = 1013.25
c_ps = 4000

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

dt = 1e-2  
n_steps = int(tau_f / dt) 
tempo = np.linspace(0, tau_f, n_steps)  
resultados = {}
tempo_computacional = {}

#------------------------------------------
# ALGORITIMO MÉTODO DE RUNGE-KUTTA 4° ORDEM 
#------------------------------------------

def calcular_derivada(t, m, r_func, T_func):

    r = r_func(t)
    T = T_func(t)
    H = H_estrela.calcular_H_estrela(T, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    return 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))

# SIMULAÇÃO RT 
def r_RT(t):
    return r_eq + (r_i - r_eq) * np.exp(-t / tau_r)

def T_RT(t):
    return T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)

inicio_rt = time.time()
massa_rt = metodo_rk4(lambda t, m: calcular_derivada(t, m, r_RT, T_RT), 1.2e-15, tempo)
resultados['RT'] = massa_rt
tempo_computacional['RT'] = time.time() - inicio_rt

# SIMULAÇÃO T 
def r_T(t):
    return r_i  # Raio constante

def T_T(t):
    return T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)

inicio_t = time.time()
massa_T = metodo_rk4(lambda t, m: calcular_derivada(t, m, r_T, T_T), 1.2e-15, tempo)
resultados['T'] = massa_T
tempo_computacional['T'] = time.time() - inicio_t

# SIMULAÇÃO R 
def r_R(t):
    return r_eq + (r_i - r_eq) * np.exp(-t / tau_r)

def T_R(t):
    return T_gota_em_k  # Temperatura constante

inicio_r = time.time()
massa_R = metodo_rk4(lambda t, m: calcular_derivada(t, m, r_R, T_R), 1.2e-15, tempo)
resultados['R'] = massa_R
tempo_computacional['R'] = time.time() - inicio_r

# SIMULAÇÃO RT-CONSTANTES 
def r_const(t):
    return  r_i     # Raio constante

def T_const(t):
    return T_gota_em_k  # Temperatura constante

inicio_rt_constantes = time.time()
massa_const = metodo_rk4(lambda t, m: calcular_derivada(t, m, r_const, T_const), 1.2e-15, tempo)
resultados['RT-Constantes'] = massa_const
tempo_computacional['RT-Constantes'] = time.time() - inicio_rt_constantes

#-----------
# RESULTADOS
#-----------
print("="*80)
print(f"SIMPLIFICAÇÕES - RK4")
print(f"r_i = {r_i*1e6} μm, H_s = {H_s} m")
print("="*80)
print(f"{'dt':<10} {'n_steps':<10} {'tempo_comp (s)':<15} {'massa_final (mol)':<20}")
print("-" * 60)

# Teste de diferentes passos de tempo
dts_test = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for dt_test in dts_test:
    n_steps_test = int(tau_f / dt_test) 
    tempo_test = np.linspace(0, tau_f, n_steps_test)
    
    inicio = time.time()
    massa_test = metodo_rk4(lambda t, m: calcular_derivada(t, m, r_RT, T_RT), 1.2e-15, tempo_test)
    tempo_comp = time.time() - inicio
    
    print(f"{dt_test:<10.0e} {n_steps_test:<10} {tempo_comp:<15.6f} {massa_test[-1]:<20.6e}")
"""
print("\n" + "="*80)
print(f"RUNGE-KUTTA 4° ORDEM - r_i = {r_i*1e6} - H_s = {H_s}")
print("="*80)
print(f"\n{'SIMULAÇÃO':<15} {'MASSA FINAL (mol)':<25} {'TEMPO COMPUTACIONAL (s)':<25} {'N° PASSOS':<10}")
print("-" * 80)
    
for metodo in ['RT', 'T', 'R', 'RT-Constantes']:
    massa_final = resultados[metodo][-1]
    tempo_comp = tempo_computacional[metodo]
    print(f"{metodo:<15} {massa_final:<25.15e} {tempo_comp:<25.4f} {n_steps:<10}")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.rcParams['text.usetex'] = True
# GRÁFICOS COMPLETOS - DESATIVAR LaTeX PARA EVITAR ERROS
plt.rcParams['text.usetex'] = False

# Calcular evolução do raio e temperatura para cada simulação
raio_RT = r_eq + (r_i - r_eq) * np.exp(-tempo / tau_r)
temperatura_RT = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-tempo / tau_T)

raio_T = np.full_like(tempo, r_i)  # Raio constante
temperatura_T = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-tempo / tau_T)

raio_R = r_eq + (r_i - r_eq) * np.exp(-tempo / tau_r)
temperatura_R = np.full_like(tempo, T_gota_em_k)  # Temperatura constante

raio_const = np.full_like(tempo, r_i)  # Raio constante
temperatura_const = np.full_like(tempo, T_gota_em_k)  # Temperatura constante



# Criar figura com subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Remover o quarto subplot
fig.delaxes(ax4)


# Gráfico 1: Massa vs Tempo
ax1.plot(tempo, resultados['RT'], "#2B00FF", linestyle='-', linewidth=2.5, label='RT')
ax1.set_xlabel('Tempo (s)', fontsize=12)
ax1.set_ylabel('Massa de CO₂ (mol)', fontsize=12)
ax1.set_xscale('log')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Evolução da Massa de CO₂', fontsize=14)

# Gráfico 2: Raio vs Tempo
ax2.plot(tempo, raio_RT*1e6, "#FF0000", linestyle='-', linewidth=2.5, label='RT')
ax2.set_xlabel('Tempo (s)', fontsize=12)
ax2.set_ylabel('Raio (μm)', fontsize=12)
ax2.set_xscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Evolução do Raio da Gota', fontsize=14)

# Gráfico 3: Temperatura vs Tempo
ax3.plot(tempo, temperatura_RT-273.15, "#00AA00", linestyle='-', linewidth=2.5, label='RT')
ax3.set_xlabel('Tempo (s)', fontsize=12)
ax3.set_ylabel('Temperatura (°C)', fontsize=12)
ax3.set_xscale('log')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_title('Evolução da Temperatura', fontsize=14)

plt.suptitle(f'Runge-Kutta 4ª Ordem - SIMPLES \nr_i = {r_i*1e6:.1f} μm  H_s = {H_s} m', fontsize=16)
plt.tight_layout()
plt.show()
"""
"""
plt.figure(figsize=(8, 4))
plt.title(f"Runge–Kutta ordem 4\n$r_i = {r_i*1e6:.1f}\\,\\mu m$\\,\\,\\, $H_s = {H_s}\\,m$", fontsize=16)
plt.plot(tempo, resultados['RT'], "#2B00FF",linestyle='-', linewidth=2.5, 
             label=f'E-RT - Massa: {massa_rt[-1]:.3e} mol | Tempo: {tempo_computacional["RT"]:.2f}s')
plt.plot(tempo, resultados['T'], '#FF0000',linestyle='-', linewidth=2.5, 
             label=f'E-T - Massa: {massa_T[-1]:.3e} mol | Tempo: {tempo_computacional["T"]:.2f}s')
plt.plot(tempo, resultados['R'], '#FFD700',linestyle='-', linewidth=2.5, 
             label=f'E-R - Massa: {massa_R[-1]:.3e} mol | Tempo: {tempo_computacional["R"]:.2f}s')
plt.plot(tempo, resultados['RT-Constantes'], "#00FF00BF",linestyle='-', linewidth=2.5, 
             label=f'E-CONTROL - Massa: {massa_const[-1]:.3e} mol | Tempo: {tempo_computacional["RT-Constantes"]:.2f}s')  
plt.xlabel('Tempo (s)', fontsize=14)
plt.ylabel('Massa (mol)', fontsize=14)
plt.xscale('log')
plt.legend(fontsize=10)  
plt.tight_layout()
plt.show()
"""