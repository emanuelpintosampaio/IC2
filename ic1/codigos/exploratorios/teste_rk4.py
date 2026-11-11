import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import time

# =============================================================================
# CONSTANTES E PARÂMETROS INICIAIS
# =============================================================================
T_a = 16.85           
T_a_em_k = 290        
T_mar_em_k = 285      
T_gota = 11.85        
T_gota_em_k = 285     
P = 1000              
M_H2O = 18.016e-3     
r_i = 30e-6          
R = 8.31              
M_NaCl = 58.443e-3    
rho_w = 1025          
v_ion = 2             
s = 34/1000           
f = 0.8               
C_ar =  0.0154        
S = 34                
R_atm = 0.082         
g = 9.81              
v_ar = 1.32e-5        
rho_ar = 1.225        
H_s = 6               
T0 = 273.15
P0 = 1013.25

# =============================================================================
# CÁLCULOS PRELIMINARES (U_f, tau_f, tau_T, tau_r)
# =============================================================================

# --- U_f PELO MÉTODO DE NEWTON ---
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
tau_f = H_s / (2 * U_f_sol)

# --- PARÂMETROS FÍSICO-QUÍMICOS ---
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

rho_vr_val = rho_vr.calcular_rho_vr(
    M_H2O, 
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    R, 
    T_a_em_k
)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val, a=17.502, b=240.97)
tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps=4000)
T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

# --- r_eq PELO MÉTODO DE NEWTON ---
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

print(f"\n--- Parâmetros Temporais ---")
print(f"tau_T = {tau_T:.3e} s")
print(f"tau_f = {tau_f:.3e} s")
print(f"tau_r = {tau_r:.3e} s")

# =============================================================================
# FUNÇÃO DE EULER COM DIAGNÓSTICO (DEBUG)
# =============================================================================
def euler_explicito_debug(massa_anterior, r, T, dt, step_n):
    """
    Método de Euler explícito com verificação de estabilidade.
    Equação: dm/dt = A - B*m
    Discretização: m_nova = m_ant + dt*(A - B*m_ant)
                  m_nova = m_ant*(1 - B*dt) + A*dt
    Estabilidade: O termo (1 - B*dt) deve estar entre -1 e 1.
    """
    H = H_estrela.calcular_H_estrela(T, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    
    # Coeficientes
    A = 4 * np.pi * r * Dg * C_ar
    B = 3 * Dg / (H * R_atm * T * r**2)
    
    # Fator de estabilidade e cálculo da nova massa
    fator_estabilidade = 1 - (B * dt)
    dm_dt = A - B * massa_anterior
    massa_nova = massa_anterior + dt * dm_dt

    # LOGICA DE PRINT:
    # Mostra a cada 20 passos OU se houver problema (massa negativa ou oscilação forte)
    if step_n % 20 == 0 or massa_nova < 0 or fator_estabilidade < -0.9:
        print(f"--- Passo {step_n} (t = {step_n*dt:.4f}s) ---")
        print(f"Raio atual (r):    {r:.4e} m")
        print(f"Massa Anterior:    {massa_anterior:.4e}")
        print(f"Coeficiente B:     {B:.4e}")
        print(f"Estabilidade (1-B*dt): {fator_estabilidade:.4f}")
        print(f"Massa Nova:        {massa_nova:.4e}")
        
        if fator_estabilidade < 0:
            print(">>> ALERTA: Oscilação iniciada (Fator < 0)")
        if fator_estabilidade < -1:
            print(">>> PERIGO: INSTABILIDADE NUMÉRICA (Fator < -1) -> Solução vai explodir")
        if massa_nova < 0:
            print(">>> ERRO CRÍTICO: Massa negativa atingida")
        
        print("-" * 40)

    return massa_nova

# =============================================================================
# SIMULAÇÃO DE DEBUG
# =============================================================================
# Usando o dt problemático para ver onde quebra
dt_debug = 1e-2  

print("\n" + "="*80)
print(f"INVESTIGAÇÃO DE ERRO - EULER EXPLÍCITO (dt = {dt_debug})")
print("="*80)

n_steps_debug = int(tau_f / dt_debug)
tempo_debug = np.linspace(0, tau_f, n_steps_debug)

massa_debug = np.zeros(n_steps_debug)
massa_debug[0] = 1.2e-15

# Loop temporal
for n in range(n_steps_debug - 1):
    t = tempo_debug[n + 1]
    
    # Atualização das variáveis físicas (Caso RT)
    r = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
    T = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
    
    # Chamada da função com debug
    massa_debug[n + 1] = euler_explicito_debug(massa_debug[n], r, T, dt_debug, n)
    
    # Trava de segurança para não poluir o terminal se explodir para infinito
    if np.abs(massa_debug[n+1]) > 1e-10: 
        print(f"\n!!! Simulação interrompida no passo {n} pois o valor explodiu ({massa_debug[n+1]:.2e}) !!!")
        break
    
    # Trava se ficar negativo para parar logo após o erro
    if massa_debug[n+1] < 0:
        print(f"\n!!! Simulação interrompida no passo {n} pois a massa ficou negativa !!!")
        break