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

# =============================================================================
# CONSTANTES E PARÂMETROS
# =============================================================================

# Constantes físicas
CONST = {
    'T_a': 18,                    # Temperatura do ar em °C
    'T_a_em_k': 291.15,           # Temperatura do ar em K
    'T_mar_em_k': 293.15,         # Temperatura do mar em K
    'T_gota': 19.8,               # Temperatura da gota em °C
    'T_gota_em_k': 292.95,        # Temperatura da gota em K
    'P': 1000,                    # Pressão em mb
    'M_H2O': 18.016e-3,           # Massa molecular da água em kg/mol
    'r_i': 100e-6,                # Raio inicial em metros
    'R': 8.31,                    # Constante universal dos gases
    'M_NaCl': 58.443e-3,          # Massa molecular de sal
    'rho_w': 1025,                # Densidade da água do mar kg/m³
    'v_ion': 2,                   # Número de íons por molécula de NaCl
    's': 34/1000,                 # Salinidade fracionária
    'f': 0.9,                     # Umidade relativa fracionária
    'C_ar': 0.0154,               # Concentração de CO₂ no ar
    'S': 34,                      # Salinidade do mar
    'R_atm': 0.082,               # Constante universal dos gases (atm)
    'g': 9.81,                    # Gravidade
    'v_ar': 1.32e-5,              # Viscosidade cinemática do ar
    'rho_ar': 1.225,              # Densidade do ar
    'H_s': 6,                     # Altura significativa da onda
    'T0': 273.15,
    'P0': 1013.25,
    'c_ps': 4000
}

# =============================================================================
# FUNÇÕES AUXILIARES OTIMIZADAS
# =============================================================================

def calcular_parametros_fixos():
    """Calcula todos os parâmetros fixos uma única vez"""
    params = {}
    
    # Velocidade terminal
    def func(U_f):
        const = (2 * CONST['r_i']**2 * CONST['g']) / (9 * CONST['v_ar']) * ((CONST['rho_w'] / CONST['rho_ar']) - 1)
        a = 1 + 0.158 * ((2 * CONST['r_i'] * U_f / CONST['v_ar']) ** (2/3))
        return U_f - const / a

    def dfunc(U_f):
        const = (2 * CONST['r_i']**2 * CONST['g']) / (9 * CONST['v_ar']) * ((CONST['rho_w'] / CONST['rho_ar']) - 1)
        a = 1 + 0.158 * ((2 * CONST['r_i'] * U_f / CONST['v_ar']) ** (2/3))
        d_a = -0.158 * (2/3) * (2 * CONST['r_i'] / CONST['v_ar'])**(2/3) * U_f**(-1/3)
        return 1 - (const * d_a) / (a**2)

    U_f0 = 0.1
    params['U_f_sol'] = metodo_newton.Newton(func, dfunc, U_f0, 1e-6, 50)
    params['tau_f'] = CONST['H_s'] / (2 * params['U_f_sol'])
    
    # Parâmetros termodinâmicos
    params['e_sat_val'] = e_sat.calcular_esat(CONST['T_a'])
    params['L_v_val'] = L_v.calcular_lv(CONST['T_gota'])
    params['D_linha_w_val'] = D_linha_w.calculate_Dw_prime(
        CONST['T_gota_em_k'], CONST['R'], CONST['P'], CONST['r_i'], 
        CONST['M_H2O'], CONST['T0'], CONST['P0'], alpha_c=0.036, Delta_w=8e-8
    )
    params['k_linha_a_val'] = k_linha_a.calculate_K_linha_a(
        CONST['T_gota'], CONST['T_gota_em_k'], CONST['P'], CONST['r_i'], 
        CONST['R'], CONST['T0'], CONST['P0'], M_a=28.9644e-3, 
        alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3
    )
    
    # Parâmetros da solução salina
    params['rho_ww'] = m_s.calcular_rho_ww(CONST['T_gota'])
    params['m_s_val'] = m_s.calcular_ms(params['rho_ww'], CONST['r_i'], CONST['s'])
    params['v_a'] = rho_s.calcular_v_a(CONST['T_gota'], params['m_s_val'], CONST['M_NaCl'], CONST['r_i'])
    params['m_w_val'] = rho_s.calcular_massa_agua(CONST['r_i'], params['rho_ww'])
    params['sigma_s_val'] = sigma_s.calculate_sigma_s(CONST['T_gota'], params['m_s_val'], params['m_w_val'])
    params['Phi_s_val'] = Phi_s.calcular_phi_s(params['m_s_val'], CONST['M_NaCl'], params['m_w_val'])
    params['rho_s_val'] = rho_s.calcular_rho_spray(params['rho_ww'], params['m_s_val'], params['m_w_val'], params['v_a'], CONST['M_NaCl'])
    
    # Temperatura de equilíbrio
    alpha_val = alpha.calcular_alpha(CONST['T_a_em_k'], a=17.502, b=240.97)
    beta_val = code_beta.calcular_beta(
        params['e_sat_val'], CONST['T_a_em_k'], params['L_v_val'], 
        CONST['M_H2O'], params['D_linha_w_val'], CONST['R'], params['k_linha_a_val']
    )
    exp_y_val = code_exp_y.calcular_exp_y(
        CONST['M_H2O'], params['sigma_s_val'], CONST['T_a_em_k'], CONST['rho_w'], 
        CONST['R'], CONST['v_ion'], params['Phi_s_val'], params['m_s_val'], 
        CONST['r_i'], params['rho_s_val'], CONST['M_NaCl']
    )
    
    Delta_T = Delta_T_contas.calculate_delta_T(
        CONST['T_a_em_k'], alpha=alpha_val, beta=beta_val,
        b=240.97, exp_y=exp_y_val, f=CONST['f']
    )
    
    params['T_eq'] = Delta_T[0] + CONST['T_a']
    params['T_eq_em_k'] = params['T_eq'] + 273.15
    
    # Tempos característicos
    rho_vr_val = rho_vr.calcular_rho_vr(
        CONST['M_H2O'], e_sat_esp.calcular_esat(CONST['T_a']), exp_y_val,
        CONST['R'], CONST['T_a_em_k']
    )
    partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(
        CONST['T_a_em_k'], rho_vr_val, a=17.502, b=240.97
    )
    params['tau_T'] = contas_tau_T.calculate_tau_T(
        params['rho_s_val'], CONST['r_i'], params['k_linha_a_val'], 
        params['D_linha_w_val'], partial_rho_vr_val, params['L_v_val'], CONST['c_ps']
    )
    
    # Raio de equilíbrio
    def zeta(r):
        term1 = (CONST['f'] - 1)
        term2 = (2 * CONST['M_H2O'] * params['sigma_s_val']) / (CONST['R'] * params['T_eq_em_k'] * CONST['rho_w'] * r)
        denominator = (4 * np.pi * params['rho_s_val'] * (r**3 / 3)) - params['m_s_val']
        term3 = (CONST['v_ion'] * params['Phi_s_val'] * params['m_s_val'] * (CONST['M_H2O'] / CONST['M_NaCl'])) / denominator
        return term1 - term2 + term3

    def dzeta_dr(r):
        term1 = (2 * CONST['M_H2O'] * params['sigma_s_val']) / (CONST['R'] * params['T_eq_em_k'] * CONST['rho_w'] * (r**2))
        denominator = (4 * np.pi * params['rho_s_val'] * (r**3)) / 3 - params['m_s_val']
        term2 = (CONST['v_ion'] * params['Phi_s_val'] * params['m_s_val'] * (CONST['M_H2O'] / CONST['M_NaCl']) * 4 * np.pi * params['rho_s_val'] * (r**2)) / (denominator**2)
        return term1 - term2

    r0 = (1.1) * (3 * params['m_s_val'] / (4 * np.pi * params['rho_s_val'])) ** (1 / 3)
    params['r_eq'] = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)
    
    params['tau_r'] = calcular_tau_r(
        CONST['f'], CONST['M_H2O'], params['sigma_s_val'], CONST['R'], CONST['T_a_em_k'],
        CONST['rho_w'], CONST['r_i'], params['rho_s_val'], params['m_s_val'], CONST['v_ion'],
        params['Phi_s_val'], CONST['M_NaCl'], params['D_linha_w_val'], params['e_sat_val'],
        params['L_v_val'], params['k_linha_a_val'], params['r_eq']
    )[1]
    
    return params

# =============================================================================
# MÉTODO RK4 OTIMIZADO PARA SISTEMAS
# =============================================================================

def metodo_rk4_vetor_otimizado(f, y0, t):
    """Método RK4 vetorial otimizado"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return y

def criar_sistema_otimizado(params):
    """Cria o sistema de EDOs otimizado com parâmetros pré-calculados"""
    
    # Parâmetros fixos
    rho_v = CONST['f'] * CONST['M_H2O'] * params['e_sat_val'] / (CONST['R'] * CONST['T_a_em_k'])
    e_sat_a = params['e_sat_val']
    
    def f_system(t, variaveis_gota):
        m, r, T = variaveis_gota
        
        # Evitar divisões repetidas
        inv_R = 1.0 / CONST['R']
        inv_rho_w = 1.0 / CONST['rho_w']
        four_pi_3 = 4 * np.pi / 3
        
        # Cálculos otimizados
        m_w = four_pi_3 * r**3 * params['rho_s_val'] - params['m_s_val']
        T_celsius = T - 273.15
        
        # Calcular apenas se necessário (evitar chamadas desnecessárias)
        sigma_novo = sigma_s.calculate_sigma_s(T_celsius, params['m_s_val'], m_w)
        Phi_novo = Phi_s.calcular_phi_s(params['m_s_val'], CONST['M_NaCl'], m_w)
        
        # Termos Y e exp_y otimizados
        term_Y1 = 2 * CONST['M_H2O'] * sigma_novo * inv_R * inv_rho_w / (CONST['T_a_em_k'] * r)
        term_Y2 = CONST['v_ion'] * Phi_novo * params['m_s_val'] * (CONST['M_H2O'] / CONST['M_NaCl']) / m_w
        Y = term_Y1 - term_Y2
        
        term_exp1 = 2 * CONST['M_H2O'] * sigma_novo * inv_R * inv_rho_w / (T * r)
        exp_y = term_exp1 - term_Y2
        
        # Outros parâmetros
        e_sat_novo = e_sat.calcular_esat(T_celsius)
        rho_vr_novo = CONST['M_H2O'] * e_sat_novo * np.exp(exp_y) * inv_R / T
        L_v_novo = L_v.calcular_lv(T_celsius)
        
        # Coeficientes de transporte - cache se possível
        D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, CONST['R'], CONST['P'], r, CONST['M_H2O'], CONST['T0'], CONST['P0'], 0.036, 8e-8)
        k_linha_a_novo = k_linha_a.calculate_K_linha_a(T_celsius, T, CONST['P'], r, CONST['R'], CONST['T0'], CONST['P0'], 28.9644e-3, 0.7, 2.16e-7, 1.006e3)
        
        # dm/dt
        H_novo = H_estrela.calcular_H_estrela(T, CONST['S'])
        Dg_novo = Dg_estrela.calcular_Dg_estrela(r, CONST['T_a_em_k'], CONST['R_atm'])
        vol = four_pi_3 * r**3
        C_gota = m / vol
        dm_dt = 4 * np.pi * r * Dg_novo * (CONST['C_ar'] - C_gota / (H_novo * CONST['R_atm'] * T))
        
        # dr/dt
        den1 = params['rho_s_val'] * CONST['R'] * CONST['T_a_em_k'] / (D_linha_w_novo * CONST['M_H2O'] * e_sat_a)
        den2 = params['rho_s_val'] * L_v_novo / (k_linha_a_novo * CONST['T_a_em_k']) * (L_v_novo * CONST['M_H2O'] / (CONST['R'] * CONST['T_a_em_k']) - 1)
        dr_dt = ((CONST['f'] - 1) - Y) / (r * (den1 + den2))
        
        # dT/dt
        dT_dt = 3 / (params['rho_s_val'] * CONST['c_ps'] * r**2) * (k_linha_a_novo * (CONST['T_a_em_k'] - T) + L_v_novo * D_linha_w_novo * (rho_v - rho_vr_novo))
        
        return np.array([dm_dt, dr_dt, dT_dt])
    
    return f_system

# =============================================================================
# SIMULAÇÃO PRINCIPAL OTIMIZADA
# =============================================================================

def executar_simulacao_otimizada():
    """Executa a simulação completa otimizada"""
    
    print("Calculando parâmetros fixos...")
    inicio_calculos = time.time()
    params = calcular_parametros_fixos()
    tempo_calculos = time.time() - inicio_calculos
    
    print(f"Parâmetros calculados em {tempo_calculos:.2f}s")
    print(f"U_f = {params['U_f_sol']:.4f} m/s")
    print(f"τ_f = {params['tau_f']:.2f} s")
    print(f"τ_T = {params['tau_T']:.2f} s")
    print(f"T_eq = {params['T_eq']:.2f} °C ({params['T_eq_em_k']:.2f} K)")
    print(f"r_eq = {params['r_eq']*1e6:.3f} μm")
    print(f"τ_r = {params['tau_r']:.2f} s")
    
    # =========================================================================
    # DISCRETIZAÇÃO OTIMIZADA
    # =========================================================================
    
    dt = 1e-4  # Aumentado de 1e-4 para 1e-3 (10x maior)
    n_steps = int(params['tau_f'] / dt) + 1
    tempo = np.linspace(0, 1e3, n_steps)
    
    print(f"\nDiscretização: dt = {dt:.0e}, n_steps = {n_steps}, tempo_total = {params['tau_f']:.2f}s")
    
    # =========================================================================
    # SISTEMA COMPLETO
    # =========================================================================
    
    print("\nExecutando sistema completo...")
    sistema = criar_sistema_otimizado(params)
    
    inicio_completo = time.time()
    variaveis_iniciais = np.array([1.2e-15, CONST['r_i'], CONST['T_gota_em_k']])
    variaveis_finais = metodo_rk4_vetor_otimizado(sistema, variaveis_iniciais, tempo)
    tempo_completo = time.time() - inicio_completo
    
    # =========================================================================
    # SOLUÇÕES SIMPLIFICADAS
    # =========================================================================
    
    print("Calculando soluções simplificadas...")
    
    # Raio e temperatura simplificados
    raio_simplificado_vals = raio_simplificado(tempo, CONST['r_i'], params['r_eq'], params['tau_r'])
    temperatura_simplificada_vals = temperatura_simplificada(tempo, CONST['T_gota_em_k'], params['T_eq_em_k'], params['tau_T'])
    
    # Massa simplificada usando RK4
    def f_rt(t, m):
        r = params['r_eq'] + (CONST['r_i'] - params['r_eq']) * np.exp(-t / params['tau_r'])
        T = params['T_eq_em_k'] + (CONST['T_gota_em_k'] - params['T_eq_em_k']) * np.exp(-t / params['tau_T'])
        H = H_estrela.calcular_H_estrela(T, CONST['S'])
        Dg = Dg_estrela.calcular_Dg_estrela(r, CONST['T_a_em_k'], CONST['R_atm'])
        vol = (4 / 3) * np.pi * r**3
        C_gota = m / vol
        return 4 * np.pi * r * Dg * (CONST['C_ar'] - C_gota / (H * CONST['R_atm'] * T))
    
    inicio_simplificado = time.time()
    massa_simplificada = metodo_rk4(f_rt, 1.2e-15, tempo)
    tempo_simplificado = time.time() - inicio_simplificado
    
    # =========================================================================
    # RESULTADOS
    # =========================================================================
    
    print("\n" + "="*80)
    print(f"RUNGE-KUTTA 4° ORDEM - SISTEMA COMPLETO OTIMIZADO")
    print(f"r_i = {CONST['r_i']*1e6:.1f} μm, H_s = {CONST['H_s']} m")
    print("="*80)
    
    print(f"\n{'PARÂMETRO':<25} {'VALOR':<25} {'UNIDADE':<15}")
    print("-" * 70)
    print(f"{'Massa final CO₂':<25} {variaveis_finais[-1, 0]:<25.3e} {'mol':<15}")
    print(f"{'Raio final':<25} {variaveis_finais[-1, 1]*1e6:<25.3f} {'μm':<15}")
    print(f"{'Temperatura final':<25} {variaveis_finais[-1, 2]-273.15:<25.2f} {'°C':<15}")
    print(f"{'Tempo computacional':<25} {tempo_completo:<25.2f} {'s':<15}")
    print(f"{'Número de passos':<25} {n_steps:<25} {'':<15}")
    
    # =========================================================================
    # GRÁFICOS
    # =========================================================================
    
    plt.rcParams['text.usetex'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.delaxes(ax4)
    
    # Massa vs Tempo
    ax1.plot(tempo, variaveis_finais[:, 0], "#2B00FF", linestyle='-', linewidth=2.5, 
             label=f'Sistema Completo - {variaveis_finais[-1, 0]:.3e} mol')
    ax1.plot(tempo, massa_simplificada, "#2B00FF", linestyle='--', linewidth=2, 
             label=f'Solução Simplificada - {massa_simplificada[-1]:.3e} mol', alpha=0.7)
    ax1.set_xlabel('Tempo (s)', fontsize=12)
    ax1.set_ylabel('Massa de CO₂ (mol)', fontsize=12)
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Evolução da Massa de CO₂', fontsize=14)
    
    # Raio vs Tempo
    ax2.plot(tempo, variaveis_finais[:, 1]*1e6, "#FF0000", linestyle='-', linewidth=2.5, 
             label=f'Sistema Completo - {variaveis_finais[-1, 1]*1e6:.3f} μm')
    ax2.plot(tempo, raio_simplificado_vals*1e6, "#FF0000", linestyle='--', linewidth=2, 
             label=f'Solução Simplificada - {raio_simplificado_vals[-1]*1e6:.3f} μm', alpha=0.7)
    ax2.set_xlabel('Tempo (s)', fontsize=12)
    ax2.set_ylabel('Raio (μm)', fontsize=12)
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Evolução do Raio da Gota', fontsize=14)
    
    # Temperatura vs Tempo
    ax3.plot(tempo, variaveis_finais[:, 2]-273.15, "#00AA00", linestyle='-', linewidth=2.5, 
             label=f'Sistema Completo - {variaveis_finais[-1, 2]-273.15:.2f} °C')
    ax3.plot(tempo, temperatura_simplificada_vals-273.15, "#00AA00", linestyle='--', linewidth=2, 
             label=f'Solução Simplificada - {temperatura_simplificada_vals[-1]-273.15:.2f} °C', alpha=0.7)
    ax3.set_xlabel('Tempo (s)', fontsize=12)
    ax3.set_ylabel('Temperatura (°C)', fontsize=12)
    ax3.set_xscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Evolução da Temperatura', fontsize=14)
    
    plt.suptitle(f'Runge-Kutta 4ª Ordem - Sistema Completo Otimizado\n'
                f'r_i = {CONST["r_i"]*1e6:.1f} μm, H_s = {CONST["H_s"]} m, dt = {dt:.0e}, n_steps = {n_steps}', 
                fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return {
        'params': params,
        'variaveis_finais': variaveis_finais,
        'massa_simplificada': massa_simplificada,
        'tempo_completo': tempo_completo,
        'tempo_simplificado': tempo_simplificado,
        'n_steps': n_steps
    }

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    resultados = executar_simulacao_otimizada()
    
    print(f"\nRESUMO DA OTIMIZAÇÃO:")
    print(f"• Tempo total sistema completo: {resultados['tempo_completo']:.2f}s")
    print(f"• Tempo total simplificado: {resultados['tempo_simplificado']:.2f}s") 
    print(f"• Número de passos reduzido: {resultados['n_steps']} (dt = 1e-3)")
    print(f"• Eficiência: {resultados['tempo_completo']/resultados['n_steps']*1e6:.2f} μs/passo")