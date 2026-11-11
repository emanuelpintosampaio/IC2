import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================================
# CONSTANTES DO PROBLEMA
# ============================================================================
T_a = 18.0           # Temperatura do ar em °C
T_a_K = 291.15       # Temperatura do ar em K
T_mar_K = 293.15     # Temperatura do mar em K
T_gota = 19.8        # Temperatura da gota em °C
T_gota_K = 292.95    # Temperatura da gota em K
P = 1000.0           # Pressão em mb
M_H2O = 18.016e-3    # Massa molecular da água em kg/mol
r_i = 100e-6         # Raio inicial em metros
R = 8.31             # Constante universal dos gases
M_NaCl = 58.443e-3   # Massa molecular de sal na água em kg/mol
rho_w = 1025.0       # Densidade da água do mar kg/m^3
v_ion = 2            # Número de íons por molécula de NaCl dissociada
s = 34/1000          # Salinidade fracionária (34 psu)
f = 0.9              # Umidade relativa fracionária
C_ar = 0.0154        # Concentração do gás carbônico no ar
S = 34.0             # Salinidade do mar
g = 9.81             # Gravidade
v_ar = 1.32e-5       # Viscosidade cinemática do ar
rho_ar = 1.225       # Densidade do ar
H_s = 6.0            # Altura significativa da onda
c_ps = 4000.0        # Calor específico
R_atm = 0.082        # Constante universal dos gases (atm)

# Parâmetros calculados (você deve substituir pelos seus cálculos reais)
U_f = 1.5            # Velocidade terminal (calculado por Newton)
tau_f = H_s / (2 * U_f)  # Tempo de vida da gota
r_eq = 95e-6         # Raio de equilíbrio (calculado por Newton)
T_eq_K = 292.5       # Temperatura de equilíbrio em K
tau_r = 10.0         # Tempo característico do raio
tau_T = 5.0          # Tempo característico da temperatura

# Valores iniciais de parâmetros (substituir pelos seus cálculos)
e_sat_val = 2064.0   # Pressão de vapor de saturação
L_v_val = 2.45e6     # Calor latente
D_w_prime = 2.5e-5   # Difusividade ajustada
k_a_prime = 0.025    # Condutividade térmica ajustada
rho_s_val = 1025.0   # Densidade da água do mar na gota
m_s_val = 4.3e-12    # Massa de sal

# ============================================================================
# FUNÇÕES AUXILIARES (suas funções existentes - versão simplificada)
# ============================================================================

def calcular_H_estrela(T, S):
    """Constante de Henry para CO2 (mol/(L·atm))"""
    T_C = T - 273.15
    return 0.034 * np.exp(-0.024 * T_C)

def calcular_Dg_estrela(r, T_a, R_atm):
    """Difusividade do CO2 no ar (m²/s)"""
    return 1.6e-5 * (T_a / 273.15)**1.75

def calcular_esat(T_C):
    """Pressão de vapor de saturação (Pa)"""
    return 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))

# ============================================================================
# MÉTODO DE RUNGE-KUTTA DE 4ª ORDEM PARA SISTEMAS
# ============================================================================

def rk4_sistema(f, y0, t_span, N):
    """
    Resolve sistema de EDOs usando Runge-Kutta de 4ª ordem
    
    Parâmetros:
    -----------
    f : função
        Função que calcula dy/dt = f(t, y)
        Deve retornar array com as derivadas
    y0 : array
        Condições iniciais [y1(0), y2(0), ..., yn(0)]
    t_span : tuple
        (t_inicial, t_final)
    N : int
        Número de passos
        
    Retorna:
    --------
    t : array
        Vetor de tempos
    y : array
        Matriz solução (cada linha é um instante, cada coluna uma variável)
    """
    t0, tf = t_span
    dt = (tf - t0) / N
    
    # Inicialização
    n_eq = len(y0)  # número de equações
    t = np.linspace(t0, tf, N+1)
    y = np.zeros((N+1, n_eq))
    y[0] = y0
    
    # Loop principal do RK4
    for n in range(N):
        tn = t[n]
        yn = y[n]
        
        # Cálculo dos coeficientes k1, k2, k3, k4
        k1 = dt * f(tn, yn)
        k2 = dt * f(tn + dt/2, yn + k1/2)
        k3 = dt * f(tn + dt/2, yn + k2/2)
        k4 = dt * f(tn + dt, yn + k3)
        
        # Atualização
        y[n+1] = yn + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y


# ============================================================================
# SISTEMA DE EQUAÇÕES DIFERENCIAIS
# ============================================================================

def sistema_completo(t, variaveis):
    """
    Sistema de EDOs para massa, raio e temperatura da gota
    
    variaveis = [m, r, T]
    retorna [dm/dt, dr/dt, dT/dt]
    """
    m, r, T = variaveis
    
    # Atualização de parâmetros que dependem de r e T
    m_w = (4/3) * np.pi * r**3 * rho_s_val - m_s_val
    
    # Evitar divisão por zero ou valores negativos
    if m_w <= 0:
        m_w = 1e-20
    
    # Parâmetros atualizados (versão simplificada - use suas funções completas)
    sigma_novo = 0.073  # Tensão superficial aproximada
    Phi_novo = 0.93     # Coeficiente osmótico aproximado
    
    Y = (2 * M_H2O * sigma_novo / (R * T_a_K * rho_w * r)) - \
        (v_ion * Phi_novo * m_s_val * (M_H2O / M_NaCl) / m_w)
    
    exp_y = (2 * M_H2O * sigma_novo / (R * T * rho_w * r)) - \
            (v_ion * Phi_novo * m_s_val * (M_H2O / M_NaCl) / m_w)
    
    e_sat_novo = calcular_esat(T - 273.15)
    rho_vr_novo = M_H2O * e_sat_novo * np.exp(exp_y) / (R * T)
    
    L_v_novo = L_v_val
    D_w_prime_novo = D_w_prime
    k_a_prime_novo = k_a_prime
    
    # dm/dt - Absorção de CO2
    H_novo = calcular_H_estrela(T, S)
    Dg_novo = calcular_Dg_estrela(r, T_a_K, R_atm)
    vol = (4/3) * np.pi * r**3
    C_gota = m / vol if vol > 0 else 0
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))
    
    # dr/dt - Evaporação/condensação
    rho_v = f * M_H2O * e_sat_val / (R * T_a_K)
    den1 = rho_s_val * R * T_a_K / (D_w_prime_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_a_prime_novo * T_a_K) * \
           (L_v_novo * M_H2O / (R * T_a_K) - 1)
    den = den1 + den2
    dr_dt = ((f - 1) - Y) / r / den if den != 0 else 0
    
    # dT/dt - Transferência de calor
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * \
            (k_a_prime_novo * (T_a_K - T) + L_v_novo * D_w_prime_novo * (rho_v - rho_vr_novo))
    
    return np.array([dm_dt, dr_dt, dT_dt])


# ============================================================================
# SOLUÇÕES ANALÍTICAS SIMPLIFICADAS
# ============================================================================

def solucao_simplificada(t, r_i, r_eq, tau_r, T_i, T_eq, tau_T, m_i):
    """
    Soluções exponenciais simplificadas para comparação
    """
    r_simp = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
    T_simp = T_eq + (T_i - T_eq) * np.exp(-t / tau_T)
    
    # Massa calculada usando r e T simplificados
    m_simp = np.zeros_like(t)
    m_simp[0] = m_i
    
    for i in range(len(t)-1):
        r_atual = r_simp[i]
        T_atual = T_simp[i]
        H = calcular_H_estrela(T_atual, S)
        Dg = calcular_Dg_estrela(r_atual, T_a_K, R_atm)
        vol = (4/3) * np.pi * r_atual**3
        C_gota = m_simp[i] / vol if vol > 0 else 0
        dm_dt = 4 * np.pi * r_atual * Dg * (C_ar - C_gota / (H * R_atm * T_atual))
        
        dt = t[i+1] - t[i]
        m_simp[i+1] = m_simp[i] + dm_dt * dt
    
    return m_simp, r_simp, T_simp


# ============================================================================
# SIMULAÇÃO PRINCIPAL
# ============================================================================

# Configuração da simulação
t_inicial = 0.0
t_final = min(tau_f, 1000.0)  # Limitar tempo máximo
N_passos = 10000

# Condições iniciais: [massa, raio, temperatura]
m_inicial = 1.2e-15  # kg (massa inicial de CO2)
condicoes_iniciais = np.array([m_inicial, r_i, T_gota_K])

print("="*80)
print("SIMULAÇÃO RUNGE-KUTTA 4ª ORDEM - SISTEMA COMPLETO")
print("="*80)
print(f"\nParâmetros:")
print(f"  Raio inicial: {r_i*1e6:.1f} μm")
print(f"  Temperatura inicial: {T_gota:.1f} °C")
print(f"  Altura significativa: {H_s} m")
print(f"  Tempo de vida estimado: {tau_f:.2f} s")
print(f"  Número de passos: {N_passos}")
print(f"  dt = {t_final/N_passos:.6f} s")

# Resolver o sistema completo
import time
inicio = time.time()
t, solucao = rk4_sistema(sistema_completo, condicoes_iniciais, (t_inicial, t_final), N_passos)
tempo_comp = time.time() - inicio

# Extrair variáveis
m_completo = solucao[:, 0]
r_completo = solucao[:, 1]
T_completo = solucao[:, 2]

# Calcular solução simplificada
m_simp, r_simp, T_simp = solucao_simplificada(t, r_i, r_eq, tau_r, T_gota_K, T_eq_K, m_inicial)

print(f"\nResultados finais (t = {t[-1]:.2f} s):")
print(f"  Massa de CO₂: {m_completo[-1]:.3e} mol")
print(f"  Raio: {r_completo[-1]*1e6:.3f} μm")
print(f"  Temperatura: {T_completo[-1]-273.15:.2f} °C")
print(f"\nTempo computacional: {tempo_comp:.4f} s")

# ============================================================================
# GRÁFICOS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Massa vs Tempo
ax1 = axes[0, 0]
ax1.plot(t, m_completo, 'b-', linewidth=2.5, 
         label=f'Sistema Completo: {m_completo[-1]:.3e} mol')
ax1.plot(t, m_simp, 'b--', linewidth=2, alpha=0.7,
         label=f'Solução Simplificada: {m_simp[-1]:.3e} mol')
ax1.set_xlabel('Tempo (s)', fontsize=12)
ax1.set_ylabel('Massa de CO₂ (mol)', fontsize=12)
ax1.set_xscale('log')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Evolução da Massa de CO₂', fontsize=14)

# Gráfico 2: Raio vs Tempo
ax2 = axes[0, 1]
ax2.plot(t, r_completo*1e6, 'r-', linewidth=2.5,
         label=f'Sistema Completo: {r_completo[-1]*1e6:.3f} μm')
ax2.plot(t, r_simp*1e6, 'r--', linewidth=2, alpha=0.7,
         label=f'Solução Simplificada: {r_simp[-1]*1e6:.3f} μm')
ax2.set_xlabel('Tempo (s)', fontsize=12)
ax2.set_ylabel('Raio (μm)', fontsize=12)
ax2.set_xscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Evolução do Raio da Gota', fontsize=14)

# Gráfico 3: Temperatura vs Tempo
ax3 = axes[1, 0]
ax3.plot(t, T_completo-273.15, 'g-', linewidth=2.5,
         label=f'Sistema Completo: {T_completo[-1]-273.15:.2f} °C')
ax3.plot(t, T_simp-273.15, 'g--', linewidth=2, alpha=0.7,
         label=f'Solução Simplificada: {T_simp[-1]-273.15:.2f} °C')
ax3.set_xlabel('Tempo (s)', fontsize=12)
ax3.set_ylabel('Temperatura (°C)', fontsize=12)
ax3.set_xscale('log')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_title('Evolução da Temperatura', fontsize=14)

# Remover o quarto subplot
fig.delaxes(axes[1, 1])

plt.suptitle(f'Runge-Kutta 4ª Ordem - Sistema Completo\nr_i = {r_i*1e6:.1f} μm | H_s = {H_s} m | {N_passos} passos', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("SIMULAÇÃO CONCLUÍDA")
print("="*80)