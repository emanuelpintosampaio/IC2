import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela
import time

# Constantes (mantidas as mesmas)
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

#-----------------------------------------------------------
# CÁLCULOS INICIAIS 
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

#--------------------------------------------
# PARÂMETROS PARA T_eq, r_eq, tau_T e tau_r
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
Delta_T = Delta_T_contas.calculate_delta_T(T_a_em_k, alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O, D_linha_w_val, R, k_linha_a_val),
    b=240.97, exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    f=f)
rho_vr_val = rho_vr.calcular_rho_vr(M_H2O, e_sat_esp.calcular_esat(T_a), code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl), R, T_a_em_k)
partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val, a=17.502, b=240.97)
tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps=4000)
T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

#---------------------------------------------------------------
# r_eq PELO MÉTODO DE NEWTON
#---------------------------------------------------------------

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
tau_r_val = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]


#---------------------------------------------------------------
# MÉTODO PREDITOR-CORRETOR CORRIGIDO
#---------------------------------------------------------------

def calcular_derivada_massa(r, T, massa):
    """Calcula a derivada da massa de CO2 - CORRIGIDA"""
    try:
        H = H_estrela.calcular_H_estrela(T, S)
        Dg = Dg_estrela.calcular_Dg_estrela(r, T, R_atm)  # Corrigido: usar T da gota, não T_a
        
        # Verificar valores físicos
        if r <= 0 or T <= 0 or massa < 0:
            return 0.0
            
        C_gota = (3 * massa) / (4 * np.pi * (r**3))
        
        # Termo de driving force limitado fisicamente
        driving_force = C_ar - C_gota / (H * R_atm * T)
        
        # Limitar a driving force para evitar instabilidades
        max_drive = 10 * C_ar  # Limite físico razoável
        driving_force = np.clip(driving_force, -max_drive, max_drive)
        
        derivada = 4 * np.pi * r * Dg * driving_force
        
        # Verificar se a derivada é fisicamente razoável
        max_derivada = 1e-10  # Limite conservador
        if abs(derivada) > max_derivada:
            derivada = np.sign(derivada) * max_derivada
            
        return derivada
        
    except (ValueError, ZeroDivisionError) as e:
        print(f"Erro no cálculo da derivada: {e}")
        return 0.0

def metodo_preditor_corretor_corrigido(tau_f, r_i, r_eq, T_gota_em_k, T_eq_em_k, tau_r, tau_T):
    """Método preditor-corretor corrigido com controle de passo"""
    
    # Condições iniciais
    t = 0.0
    massa = 1.2e-15
    
    # Listas para armazenar resultados
    tempo_list = [t]
    massa_list = [massa]
    r_list = [r_i]
    T_list = [T_gota_em_k]
    
    # Passo de tempo inicial CORRETO conforme Andreas
    # Andreas: Δt₁ = 10^Int[log(2×10⁵ r₀²)]
    dt_initial = 10 ** int(np.log10(2e5 * r_i**2))
    dt = max(dt_initial, 1e-6)  # Evitar passo muito pequeno
    print(f"Passo inicial: {dt}")
    
    iteracao = 0
    max_iteracoes = 10000
    
    # Parâmetros de controle
    massa_min = 1e-20  # Massa mínima fisicamente possível
    convergencia_count = 0
    
    while t < tau_f and iteracao < max_iteracoes and massa > massa_min:
        iteracao += 1
        
        # Calcular raio e temperatura atuais
        r_current = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
        T_current = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t / tau_T)
        
        # PREDITOR: Calcular derivada no ponto atual
        phi_n = calcular_derivada_massa(r_current, T_current, massa)
        
        # Predição
        massa_pred = massa + dt * phi_n
        massa_pred = max(massa_pred, massa_min)  # Evitar massa negativa
        
        # Calcular raio e temperatura no próximo passo
        t_next = t + dt
        r_next = r_eq + (r_i - r_eq) * np.exp(-t_next / tau_r)
        T_next = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t_next / tau_T)
        
        # CORRETOR: Calcular derivada no ponto predito
        phi_n1 = calcular_derivada_massa(r_next, T_next, massa_pred)
        
        # Média das derivadas
        phi_avg = 0.5 * (phi_n + phi_n1)
        
        # Correção
        massa_cor = massa + dt * phi_avg
        massa_cor = max(massa_cor, massa_min)
        
        # Verificar convergência
        if massa > 0:
            erro_relativo = abs(massa_cor - massa_pred) / massa
            if erro_relativo < 1e-4:
                convergencia_count += 1
            else:
                convergencia_count = 0
        else:
            convergencia_count = 0
        
        # Aceitar o passo se convergente ou reduzir dt
        if convergencia_count >= 2 or dt < 1e-8:
            # Aceitar passo
            massa = massa_cor
            t = t_next
            
            # Armazenar resultados
            tempo_list.append(t)
            massa_list.append(massa)
            r_list.append(r_next)
            T_list.append(T_next)
            
            # Atualizar passo de tempo (conforme Andreas)
            novo_dt = 0.1 * t
            dt = max(min(novo_dt, tau_f - t), 1e-6)
            
            convergencia_count = 0
        else:
            # Reduzir passo e tentar novamente
            dt *= 0.5
            if dt < 1e-10:
                break
        
        # Log a cada 100 iterações
        if iteracao % 100 == 0:
            print(f"Iteração {iteracao}: t = {t:.4f}, massa = {massa:.2e}, dt = {dt:.2e}")
    
    print(f"Simulação finalizada: {iteracao} iterações")
    return np.array(tempo_list), np.array(massa_list), np.array(r_list), np.array(T_list)

# Executar a simulação corrigida
print("\n" + "="*80)
print("MÉTODO PREDITOR-CORRETOR CORRIGIDO")
print("="*80)

inicio = time.time()
tempo_corrigido, massa_corrigida, raio_corrigido, temp_corrigida = metodo_preditor_corretor_corrigido(
    tau_f, r_i, r_eq, T_gota_em_k, T_eq_em_k, tau_r_val, tau_T
)
tempo_comp_corrigido = time.time() - inicio

print(f"Tempo computacional: {tempo_comp_corrigido:.4f} s")
print(f"Número de passos: {len(tempo_corrigido)}")
print(f"Massa final: {massa_corrigida[-1]:.6e} mol")
print(f"Tempo final: {tempo_corrigido[-1]:.4f} s")
print(f"Variação total de massa: {massa_corrigida[-1] - massa_corrigida[0]:.6e} mol")
