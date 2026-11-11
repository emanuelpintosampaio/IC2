import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta,D_linha_w,e_sat,k_linha_a,L_v
from T_EQ.EXP_Y import code_exp_y,m_s,Phi_s,rho_s,sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr,rho_vr,contas_tau_T,e_sat_esp
from TAU_R.partial_rt_teste import calcular_tau_r
from rk4 import raio_simplificado,temperatura_simplificada,metodo_rk4
from GRAFICO.DGH import Dg_estrela, H_estrela  
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Constantes IURY 
T_a = 18           # Temperatura do ar em °C ANDREAS(18) IRUY(16.85)
T_a_em_k = 291.15        # Temperatura do ar em K ANDREAS(291.15) IURY(290)
T_mar_em_k = 293.15      # Temperatura do mar em k ANDREAS(293.15) IURY(285)
T_gota = 19.8        # Temperatura da gota em °C ANDREAS(19.8) IURY(11.85)
T_gota_em_k = 292.95     # Temperatura da gota em K ANDREAS(292.95) IURY(285)
P = 1000              # Pressão em mb
M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
r_i = 100e-6           # Raio inicial ANDREAS(100) IURY(30)
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

print(f"\nU_f = {U_f_sol}")
print(rf"\tau_f = {tau_f}")

#--------------------------------------------
# PARÂMETROS PARA T_eq, r_eq, \tau_T e \tau_r
#--------------------------------------------

e_sat_val = e_sat.calcular_esat(T_a) # Pressão de vapor de saturação 
L_v_val = L_v.calcular_lv(T_gota) # Calor latente de vaporização da água
D_linha_w_val = D_linha_w.calculate_Dw_prime(T_gota_em_k, R,P, r_i, M_H2O , T0 , P0 , alpha_c = 0.036 , Delta_w = 8e-8) # Difusividade do vapor d'água ajustada para efeitos de não-continuidade
k_linha_a_val = k_linha_a.calculate_K_linha_a(T_gota,T_gota_em_k, P, r_i , R , T0 , P0, M_a = 28.9644e-3 , alpha_T = 0.7 , delta_T = 2.16e-7, c_pa = 1.006e3) # Condutividade térmica ajustada para efeitos de não-continuidade
rho_ww = m_s.calcular_rho_ww(T_gota) # Densidade da água pura em uma gota 
m_s_val = m_s.calcular_ms(rho_ww,r_i,s) # Massa de sal em uma gotícula de spray
v_a = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i) # Volume molar aparente de uma solução aquosa
m_w_val = rho_s.calcular_massa_agua(r_i, rho_ww) # Massa de água pura em uma gotícula de spray
sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val,m_w_val) # Tensão superficial de uma superfície plana de água
Phi_s_val = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_val) # Coeficiente osmótico prático do NaCl dissolvido em água
rho_s_val = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w_val, v_a, M_NaCl) # Densidade da água do mar na gota
Delta_T = Delta_T_contas.calculate_delta_T(T_a_em_k, alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta= code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O, D_linha_w_val, R, k_linha_a_val),
    b=240.97,exp_y=code_exp_y.calcular_exp_y(M_H2O , sigma_s_val , T_a_em_k , rho_w , R, v_ion , Phi_s_val , m_s_val , r_i , rho_s_val ,  M_NaCl),
    f=f)
rho_vr_val = rho_vr.calcular_rho_vr(M_H2O,e_sat_esp.calcular_esat(T_a),code_exp_y.calcular_exp_y(M_H2O , sigma_s_val , T_a_em_k , rho_w , R, v_ion , Phi_s_val , m_s_val , r_i , rho_s_val ,  M_NaCl),R,T_a_em_k) # É a densidade de vapor d'água na interface da gotícula
partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k,rho_vr_val,a=17.502,b=240.97)
tau_T = contas_tau_T.calculate_tau_T(rho_s_val , r_i , k_linha_a_val , D_linha_w_val ,partial_rho_vr_val ,L_v_val ,c_ps)
T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

print(rf"\tau_T = {tau_T}")
print(f"T_eq = {T_eq:.2f} °C ({T_eq_em_k:.2f} K)")

#---------------------------------------------------------------
# r_eq PELO MÉTODO DE NEWTON req_n+1 = req_n - f(req_n)/f'(req_n)
#---------------------------------------------------------------

# f(req_n)
def zeta(r):
    term1 = (f - 1)
    term2 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r)
    denominator = (4 * np.pi * rho_s_val * (r**3 / 3)) - m_s_val
    term3 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / denominator
    return term1 - term2 + term3

# f'(req_n)
def dzeta_dr(r):
    term1 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * (r**2))
    denominator = (4 * np.pi * rho_s_val * (r**3)) / 3 - m_s_val
    term2 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl) * 4 * np.pi * rho_s_val * (r**2)) / (denominator**2)
    return term1 - term2

r0 = (1.1) * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1 / 3) # chute inicial
r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)
tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

print(f"r_eq = {r_eq}")
print(rf"\tau_r = {tau_r}")

#-----------------------
# DISCRETIZAÇÃO DO TEMPO
#-----------------------

# Implementação do método de Euler com preditor-corretor
def euler_preditor_corretor_sistema(f, y0, t):
    """
    Método de Euler com preditor-corretor para sistemas de EDOs
    """
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        
        # Passo preditor (Euler explícito)
        phi_n = f(t[i], y[i])
        y_pred = y[i] + dt * phi_n
        
        # Calcula phi_{n+1} no ponto predito
        phi_n1 = f(t[i + 1], y_pred)
        
        # Média de phi (eq. 43)
        phi_hat = 0.5 * (phi_n + phi_n1)
        
        # Passo corretor (eq. 44)
        y[i + 1] = y[i] + dt * phi_hat
    
    return y

# Escolha adaptativa do passo de tempo conforme descrito no texto
def calcular_dt_adaptativo(r0, t_n):

    # Passo de tempo inicial baseado no raio inicial
    dt_inicial = 10 ** np.floor(np.log10(2e5 * r0**2))
    
    # Passos subsequentes são logarítmicos
    if t_n > 0:
        return 0.1 * t_n
    else:
        return dt_inicial

# Vetor de tempo adaptativo
def criar_tempo_adaptativo(tau_f, r0, n_pontos_min=1000):
    """
    Cria vetor de tempo com passos adaptativos
    """
    tempo = [0]
    t_atual = 0
    
    while t_atual < tau_f and len(tempo) < n_pontos_min * 10:  # Limite máximo de pontos
        dt = calcular_dt_adaptativo(r0, t_atual)
        t_proximo = t_atual + dt
        
        if t_proximo > tau_f:
            t_proximo = tau_f
            
        tempo.append(t_proximo)
        t_atual = t_proximo
    
    return np.array(tempo)

# Criar vetor de tempo adaptativo
tempo_adaptativo = criar_tempo_adaptativo(tau_f, r_i)
print(f"Número de pontos temporais: {len(tempo_adaptativo)}")
print(f"Tempo máximo: {tempo_adaptativo[-1]:.2f}s")

# rho_v fixa
rho_v = f * M_H2O * e_sat_val / (R * T_a_em_k)
e_sat_a = e_sat_val

# SISTEMA DE EDO 
def f_system(t, variaveis_gota):
    m, r, T = variaveis_gota
    # Atualizações 
    m_w = (4 / 3) * np.pi * r**3 * rho_s_val - m_s_val
    sigma_novo = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w)
    Phi_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
    Y = (2 * M_H2O * sigma_novo / (R * T_a_em_k * rho_w * r)) - (v_ion * Phi_novo * m_s_val * (M_H2O / M_NaCl) / m_w)
    exp_y = (2 * M_H2O * sigma_novo / (R * T * rho_w * r)) - (v_ion * Phi_novo * m_s_val * (M_H2O / M_NaCl) / m_w)
    e_sat_novo = e_sat.calcular_esat(T - 273.15)
    rho_vr_novo = M_H2O * e_sat_novo * np.exp(exp_y) / (R * T)
    L_v_novo = L_v.calcular_lv(T - 273.15)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0, 0.036, 8e-8)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T - 273.15, T, P, r, R, T0, P0, 28.9644e-3, 0.7, 2.16e-7, 1.006e3)
    # dm/dt 
    H_novo = H_estrela.calcular_H_estrela(T, S)
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))
    # dr/dt 
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_a)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2
    dr_dt = ((f - 1) - Y) / r / den
    # dT/dt
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v - rho_vr_novo))
    return np.array([dm_dt, dr_dt, dT_dt])

# SIMULAÇÃO DO SISTEMA COMPLETO COM EULER PREDITOR-CORRETOR
inicio_completo = time.time()
variaveis_iniciais = np.array([1.2e-15, r_i, T_gota_em_k])
variaveis_finais = euler_preditor_corretor_sistema(f_system, variaveis_iniciais, tempo_adaptativo)
resultados = variaveis_finais[:, 0]  # massa m(t)
tempo_computacional = time.time() - inicio_completo

# Interpolação para obter valores nos mesmos pontos de tempo para comparação
from scipy import interpolate

# Criar funções de interpolação
massa_interp = interpolate.interp1d(tempo_adaptativo, variaveis_finais[:, 0], kind='linear', 
                                   bounds_error=False, fill_value=(variaveis_finais[0, 0], variaveis_finais[-1, 0]))
raio_interp = interpolate.interp1d(tempo_adaptativo, variaveis_finais[:, 1], kind='linear',
                                  bounds_error=False, fill_value=(variaveis_finais[0, 1], variaveis_finais[-1, 1]))
temp_interp = interpolate.interp1d(tempo_adaptativo, variaveis_finais[:, 2], kind='linear',
                                  bounds_error=False, fill_value=(variaveis_finais[0, 2], variaveis_finais[-1, 2]))

# Criar vetor de tempo uniforme para plotagem (usando tempo adaptativo como referência)
tempo_plot = np.linspace(0, min(tau_f, tempo_adaptativo[-1]), 1000)

# Obter valores interpolados
massa_plot = massa_interp(tempo_plot)
raio_plot = raio_interp(tempo_plot)
temp_plot = temp_interp(tempo_plot)

# SIMULAÇÃO DAS SOLUÇÕES SIMPLIFICADAS
raio_simplificado_vals = raio_simplificado(tempo_plot, r_i, r_eq, tau_r)
temperatura_simplificada_vals = temperatura_simplificada(tempo_plot, T_gota_em_k, T_eq_em_k, tau_T)

# SIMULAÇÃO RT 
def f_rt(t, m):
    r = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
    T = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
    H = H_estrela.calcular_H_estrela(T, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    return 4 * np.pi * r * Dg * (C_ar - C_gota / (H * R_atm * T))

massa_rt = metodo_rk4(f_rt, 1.2e-15, tempo_plot)

# Calcular massa usando as soluções simplificadas para raio e temperatura
massa_simplificada = np.zeros_like(tempo_plot)
massa_simplificada[0] = 1.2e-15  # Massa inicial

for i, t in enumerate(tempo_plot):
    if i < len(tempo_plot) - 1:  # Parar no penúltimo ponto
        dt_step = tempo_plot[i+1] - tempo_plot[i]
        
        r_novo = raio_simplificado_vals[i]
        T_novo = temperatura_simplificada_vals[i]
        
        H_novo = H_estrela.calcular_H_estrela(T_novo, S)
        Dg_novo = Dg_estrela.calcular_Dg_estrela(r_novo, T_a_em_k, R_atm)
        vol = (4 / 3) * np.pi * r_novo**3
        C_gota = massa_simplificada[i] / vol
        dm_dt = 4 * np.pi * r_novo * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T_novo))        
        massa_simplificada[i+1] = massa_simplificada[i] + dm_dt * dt_step

#-----------
# RESULTADOS
#-----------

print("\n" + "="*80)
print(f"MÉTODO EULER PREDITOR-CORRETOR - r_i = {r_i*1e6} - H_s = {H_s}")
print("="*80)
print(f"\n{'SIMULAÇÃO':<15} {'MASSA FINAL (mol)':<25} {'TEMPO COMPUTACIONAL (s)':<25} {'N° PONTOS':<10}")
print("-" * 80)
    
massa_final = variaveis_finais[-1, 0]
print(f"{'completo':<15} {massa_final:<25.15e} {tempo_computacional:<25.4f} {len(tempo_adaptativo):<10}")

# GRÁFICOS COMPLETOS - DESATIVAR LaTeX PARA EVITAR ERROS
plt.rcParams['text.usetex'] = False

# Criar figura com subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

fig.delaxes(ax4)

# Gráfico 1: Massa vs Tempo
ax1.plot(tempo_plot, massa_plot, "#2B00FF", linestyle='-', linewidth=2.5, 
         label=f'Sistema Completo - {massa_plot[-1]:.3e} mol')
ax1.plot(tempo_plot, massa_simplificada, "#2B00FF", linestyle='--', linewidth=2, 
         label=f'Solução Simplificada -  {massa_simplificada[-1]:.3e} mol', alpha=0.7)
ax1.set_xlabel('Tempo (s)', fontsize=12)
ax1.set_ylabel('Massa de CO₂ (mol)', fontsize=12)
ax1.set_xscale('log')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Evolução da Massa de CO₂', fontsize=14)

# Gráfico 2: Raio vs Tempo
ax2.plot(tempo_plot, raio_plot*1e6, "#FF0000", linestyle='-', linewidth=2.5, 
         label=f'Sistema Completo - {raio_plot[-1]*1e6:.3f} μm')
ax2.plot(tempo_plot, raio_simplificado_vals*1e6, "#FF0000", linestyle='--', linewidth=2, 
         label=f'Solução Simplificada - {raio_simplificado_vals[-1]*1e6:.3f} μm', alpha=0.7)
ax2.set_xlabel('Tempo (s)', fontsize=12)
ax2.set_ylabel('Raio (μm)', fontsize=12)
ax2.set_xscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Evolução do Raio da Gota', fontsize=14)

# Gráfico 3: Temperatura vs Tempo
ax3.plot(tempo_plot, temp_plot-273.15, "#00AA00", linestyle='-', linewidth=2.5, 
         label=f'Sistema Completo - {temp_plot[-1]-273.15:.2f} °C ')
ax3.plot(tempo_plot, temperatura_simplificada_vals-273.15, "#00AA00", linestyle='--', linewidth=2, 
         label=f'Solução Simplificada - {temperatura_simplificada_vals[-1]-273.15:.2f} °C', alpha=0.7)
ax3.set_xlabel('Tempo (s)', fontsize=12)
ax3.set_ylabel('Temperatura (°C)', fontsize=12)
ax3.set_xscale('log')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_title('Evolução da Temperatura', fontsize=14)

plt.suptitle(f'Método Euler Preditor-Corretor - Sistema Completo\nr_i = {r_i*1e6:.1f} μm  H_s = {H_s} m  ', 
             fontsize=16)
plt.tight_layout()
plt.show()

print(f"\nRESUMO DOS RESULTADOS:")
print(f"• Massa final de CO₂: {variaveis_finais[-1, 0]:.3e} mol")
print(f"• Raio final: {variaveis_finais[-1, 1]*1e6:.3f} μm")
print(f"• Temperatura final: {variaveis_finais[-1, 2]-273.15:.2f} °C")
print(f"• Tempo de vida da gota (τ_f): {tau_f:.2f} s")
print(f"• Tempo computacional: {tempo_computacional:.2f} s")
print(f"• Número de pontos temporais: {len(tempo_adaptativo)}")
print(f"• Raio de equilíbrio: {r_eq*1e6:.3f} μm")
print(f"• Temperatura de equilíbrio: {T_eq:.2f} °C")