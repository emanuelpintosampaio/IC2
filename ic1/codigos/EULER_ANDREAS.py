import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import time

#-----------------------------------------------------------
# 1. CONSTANTES E CONFIGURAÇÕES INICIAIS
#-----------------------------------------------------------
T_a = 18           
T_a_em_k = 291.15        
T_mar_em_k = 293.15      
T_gota = 19.8        
T_gota_em_k = 293.35     
P = 1000              
M_H2O = 18.016e-3     
r_i = 100e-6          
R = 8.31              
M_NaCl = 58.443e-3    
rho_w = 1025          
v_ion = 2             
s = 34/1000           
f = 0.9               
C_ar =  0.0154        
S = 34                
R_atm = 0.082         
g = 9.81              
v_ar = 1.32e-5        
rho_ar = 1.225        
H_s = 6               
T0 = 273.15
P0 = 1013.25
c_ps = 4000           

#-----------------------------------------------------------
# 2. CÁLCULOS DE VELOCIDADE TERMINAL E TEMPO DE QUEDA
#-----------------------------------------------------------
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

"""
print("="*80)
print("CONFIGURAÇÕES INICIAIS DA SIMULAÇÃO")
print("="*80)
print(f"Temperatura do ar: {T_a} °C ({T_a_em_k} K)")
print(f"Temperatura da gota inicial: {T_gota} °C ({T_gota_em_k} K)")
print(f"Raio inicial da gota: {r_i*1e6:.2f} μm")
print(f"Altura significativa: {H_s} m")
print(f"Umidade relativa: {f*100}%")
print(f"Velocidade terminal calculada: {U_f_sol:.6f} m/s")
print(f"Tempo de vida da gota (tau_f): {tau_f:.4f} s")
"""

#--------------------------------------------
# 3. CÁLCULO DOS PARÂMETROS FISICOS E EQUILÍBRIO
#--------------------------------------------

e_sat_val = e_sat.calcular_esat(T_a)
L_v_val = L_v.calcular_lv(T_gota)
D_linha_w_val = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
rho_ww_val = m_s.calcular_rho_ww(T_gota)
m_s_val = m_s.calcular_ms(rho_ww_val, r_i, s)
v_a_val = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w_val = rho_s.calcular_massa_agua(r_i, rho_ww_val)
sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w_val)
Phi_s_val = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_val)
rho_s_val = rho_s.calcular_rho_spray(rho_ww_val, m_s_val, m_w_val, v_a_val, M_NaCl)

Delta_T = Delta_T_contas.calculate_delta_T(
    T_a_em_k, 
    alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O, D_linha_w_val, R, k_linha_a_val),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    f=f
)

rho_vr_val = rho_vr.calcular_rho_vr(M_H2O, e_sat_esp.calcular_esat(T_a), code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl), R, T_a_em_k)
partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val, a=17.502, b=240.97)
tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps=4000)

T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

# Cálculo do r_eq
def zeta(r_i_val):
    term1 = (f - 1)
    term2 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r_i_val)
    denominator = (4 * np.pi * rho_s_val * (r_i_val**3 / 3)) - m_s_val
    term3 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / denominator
    return term1 - term2 + term3

def dzeta_dr(r_i_val):
    term1 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * (r_i_val**2))
    denominator = (4 * np.pi * rho_s_val * (r_i_val**3)) / 3 - m_s_val
    term2 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl) * 4 * np.pi * rho_s_val * (r_i_val**2)) / (denominator**2)
    return term1 - term2

r0_newton = (1.1) * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1 / 3)
r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0_newton, 1e-6, 100)
tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

"""
print("\n" + "="*80)
print("PARÂMETROS DE EQUILÍBRIO CALCULADOS")
print("="*80)
print(f"Temperatura de equilíbrio (T_eq): {T_eq:.2f} °C ({T_eq_em_k:.2f} K)")
print(f"Raio de equilíbrio (r_eq): {r_eq*1e6:.4f} μm")
print(f"Tempo característico térmico (tau_T): {tau_T:.6f} s")
print(f"Tempo característico do raio (tau_r): {tau_r:.6f} s")
print(f"Massa de sal na gota: {m_s_val:.2e} kg")
print(f"Densidade do spray: {rho_s_val:.2f} kg/m³")
"""

#---------------------------------------------------------------
# 4. FUNÇÃO AUXILIAR: CÁLCULO DO FLUXO DE GÁS
#---------------------------------------------------------------
def calcular_fluxo(massa_atual, r_atual, T_atual):
    H = H_estrela.calcular_H_estrela(T_atual, S)
    Dg = Dg_estrela.calcular_Dg_estrela(r_atual, T_a_em_k, R_atm)
    
    C_gota = (3 * massa_atual) / (4 * np.pi * (r_atual**3))
    derivada = 4 * np.pi * r_atual * Dg * (C_ar - C_gota / (H * R_atm * T_atual))
    return derivada

#---------------------------------------------------------------
# 5. SIMULAÇÃO COM DT FIXO PARA ANÁLISE
#---------------------------------------------------------------
def simular_com_dt_fixo(dt_fixo):
    """Executa simulação com passo de tempo fixo"""
    inicio_sim = time.time()
    
    # Condições Iniciais
    t = 0.0
    massa = 1.2e-15
    historico_t = [t]
    historico_massa = [massa]
    historico_r = [r_i]
    historico_T = [T_gota_em_k]

    passo = 0
    while t < tau_f:
        passo += 1
        
        # Calcular Estado Atual (n)
        r_n = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
        T_n = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
        
        # Derivada no passo n
        phi_n = calcular_fluxo(massa, r_n, T_n)
        
        # Preditor
        massa_predita = massa + dt_fixo * phi_n
        
        # Estado Futuro
        t_next = t + dt_fixo
        r_next = r_eq + (r_i - r_eq) * np.exp(-t_next / tau_r)
        T_next = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t_next / tau_T)
        
        # Corretor
        converged = False
        iter_corretor = 0
        
        while not converged and iter_corretor < 20:
            phi_next = calcular_fluxo(massa_predita, r_next, T_next)
            phi_hat = 0.5 * (phi_n + phi_next)
            massa_corrigida = massa + dt_fixo * phi_hat
            
            if abs(massa_predita) > 1e-30:
                erro_rel = abs(massa_corrigida - massa_predita) / abs(massa_predita)
            else:
                erro_rel = 0.0
                
            if erro_rel < 0.0003:
                converged = True
                massa = massa_corrigida
            else:
                massa_predita = massa_corrigida
                iter_corretor += 1
        
        t = t_next
        
        # Armazenar resultados
        historico_t.append(t)
        historico_massa.append(massa)
        historico_r.append(r_next)
        historico_T.append(T_next)
        
        # Verificar se ultrapassou tau_f
        if t >= tau_f:
            break

    tempo_comp = time.time() - inicio_sim
    return passo, tempo_comp, massa, historico_t[-1]

#---------------------------------------------------------------
# 6. IMPLEMENTAÇÃO DO MODELO DE ANDREAS (1989) - DT VARIÁVEL
#---------------------------------------------------------------

inicio_sim = time.time()

# Listas para armazenar o histórico
historico_t = []
historico_massa = []
historico_r = []
historico_T = []
historico_dt = []  # Para armazenar os dt usados

# Condições Iniciais
t = 0.0
massa = 1.2e-15
historico_t.append(t)
historico_massa.append(massa)
historico_r.append(r_i)
historico_T.append(T_gota_em_k)

# Passo de Tempo Inicial (Andreas Eq. 47)
dt = 10 ** int(np.log10(2e5 * (r_i**2)))
historico_dt.append(dt)

print(f"dt inicial do Andreas: {dt} s")

passo = 0
while t < tau_f:
    passo += 1
    
    # A. Calcular Estado Atual (n)
    r_n = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
    T_n = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
    
    # Derivada no passo n (Phi_n)
    phi_n = calcular_fluxo(massa, r_n, T_n)
    
    # B. Preditor (Euler Explícito)
    massa_predita = massa + dt * phi_n
    
    # C. Calcular Estado Futuro Estimado (n+1)
    t_next = t + dt
    r_next = r_eq + (r_i - r_eq) * np.exp(-t_next / tau_r)
    T_next = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t_next / tau_T)
    
    # D. Loop Corretor
    converged = False
    iter_corretor = 0
    
    while not converged and iter_corretor < 20:
        phi_next = calcular_fluxo(massa_predita, r_next, T_next)
        phi_hat = 0.5 * (phi_n + phi_next)
        massa_corrigida = massa + dt * phi_hat
        
        if abs(massa_predita) > 1e-30:
            erro_rel = abs(massa_corrigida - massa_predita) / abs(massa_predita)
        else:
            erro_rel = 0.0
            
        if erro_rel < 0.0003:
            converged = True
            massa = massa_corrigida
        else:
            massa_predita = massa_corrigida
            iter_corretor += 1
    
    # E. Atualização do Tempo e Passo
    t = t_next
    
    # Armazenar resultados
    historico_t.append(t)
    historico_massa.append(massa)
    historico_r.append(r_next)
    historico_T.append(T_next)
    
    # Calcular novo dt (Andreas Eq 49)
    if t > 0:
        new_dt = 0.1 * t
        dt = new_dt
        historico_dt.append(dt)
        
        if t + dt > tau_f:
            dt = tau_f - t

tempo_total = time.time() - inicio_sim

#---------------------------------------------------------------
# 7. ANÁLISE DE CONVERGÊNCIA COM DIFERENTES DT FIXOS
#---------------------------------------------------------------
print("\n" + "="*80)
print("dt FIXOS")
print("="*80)

dts_testar = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
resultados = []

print(f"\n{'dt':<12} {'n° passos':<10} {'tempo comp (s)':<15} {'massa final (mol)':<20} {'tempo final (s)':<15}")
print("-" * 80)

for dt in dts_testar:
    n_steps, tempo_comp, massa_final, tempo_final = simular_com_dt_fixo(dt)
    resultados.append((dt, n_steps, tempo_comp, massa_final, tempo_final))
    print(f"{dt:<12.1e} {n_steps:<10} {tempo_comp:<15.6f} {massa_final:<20.6e} {tempo_final:<15.4f}")


#---------------------------------------------------------------
# 8. RESULTADOS FINAIS - MÉTODO ANDREAS
#---------------------------------------------------------------
#---------------------------------------------------------------
# 8. RESULTADOS FINAIS - MÉTODO ANDREAS (DT VARIÁVEL)
#---------------------------------------------------------------
#---------------------------------------------------------------
# 8. RESULTADOS FINAIS - MÉTODO ANDREAS (DT VARIÁVEL)
#---------------------------------------------------------------
print("\n" + "="*80)
print("dt VARIÁVEL - ANDREAS")
print("="*80)

# Mostrar pontos amostrais representativos
pontos_amostra_indices = [0, len(historico_t)//4, len(historico_t)//2, 3*len(historico_t)//4, len(historico_t)-1]
pontos_amostra_passos = [0, passo//4, passo//2, 3*passo//4, passo]

print(f"\n{'dt (s)':<12} {'n° passos':<10} {'massa (mol)':<20} {'t (s)':<18}")
print("-" * 65)

for i, n_passos in zip(pontos_amostra_indices, pontos_amostra_passos):
    if i < len(historico_t):
        print(f"{historico_dt[i]:<12.2e} {n_passos:<10} {historico_massa[i]:<20.6e} {historico_t[i]:<18.4f}")
print(f"Tempo computacional: {tempo_total:.4f} s")

print("\n" + "="*80)
print("Comparação")
print("="*80)
print(f"{'Método':<15} {'n° passos':<10} {'tempo computacional (s)':<15} {'massa final (mol)':<20} {'t (s)':<15}")
print("-" * 80)

# Resultados do método Andreas
print(f"{'Andreas':<15} {passo:<10} {tempo_total:<15.4f} {historico_massa[-1]:<20.6e} {historico_t[-1]:<15.4f}")

# Resultados do melhor passo fixo (para comparação)
if resultados:
    melhor_dt_fixo = resultados[-1]
    print(f"{'Fixo '+str(melhor_dt_fixo[0]):<15} {melhor_dt_fixo[1]:<10} {melhor_dt_fixo[2]:<15.4f} {melhor_dt_fixo[3]:<20.6e} {melhor_dt_fixo[4]:<15.4f}")

