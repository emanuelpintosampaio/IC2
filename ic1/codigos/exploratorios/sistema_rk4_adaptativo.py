"""
Sistema de EDOs para dinâmica de gotas com spray marítimo
Implementação usando RK4 com controle adaptativo de passo (step doubling)

Baseado no código original 6_-RK4_SISTEMA.py com implementação do método adaptativo
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Importar o módulo RK4 adaptativo
from rk4_adaptativo import integrar_rk4_adaptativo, interpolate_solution

# ==============================================================================
# SEÇÃO 1: FUNÇÕES AUXILIARES (simulando os módulos externos)
# ==============================================================================

class FuncoesAuxiliares:
    """Classe com funções auxiliares para simular os módulos externos"""
    
    @staticmethod
    def calcular_esat(T_celsius):
        """Pressão de vapor de saturação (mb) - Fórmula de Magnus"""
        return 6.1094 * np.exp(17.625 * T_celsius / (T_celsius + 243.04))
    
    @staticmethod
    def calcular_lv(T_celsius):
        """Calor latente de vaporização (J/kg)"""
        return (2.501 - 0.00237 * T_celsius) * 1e6
    
    @staticmethod
    def calculate_Dw_prime(T_K, R, P, r, M_H2O, T0, P0, alpha_c, Delta_w):
        """Difusividade do vapor d'água ajustada"""
        T_celsius = T_K - 273.15
        D0 = 2.11e-5 * (T_K / T0)**1.94 * (P0 / P)
        lambda_v = 2.0 * Delta_w  # Comprimento médio livre
        correction = 1 + (lambda_v / r) * (1.33 + 0.71 * (r / lambda_v))
        return D0 / correction
    
    @staticmethod
    def calculate_K_linha_a(T_celsius, T_K, P, r, R, T0, P0, M_a, alpha_T, delta_T, c_pa):
        """Condutividade térmica ajustada"""
        k0 = 2.4e-2 * (T_K / T0)**0.8
        lambda_a = 2.0 * delta_T  # Comprimento médio livre
        correction = 1 + (lambda_a / r) * (1.67 + 0.67 * (r / lambda_a))
        return k0 / correction
    
    @staticmethod
    def calcular_rho_ww(T_celsius):
        """Densidade da água pura (kg/m³)"""
        return 1000.0 - 0.0178 * abs(T_celsius - 4.0)
    
    @staticmethod
    def calcular_ms(rho_ww, r, s):
        """Massa de sal na gotícula (kg)"""
        V = (4/3) * np.pi * r**3
        m_gota = rho_ww * V
        return m_gota * s
    
    @staticmethod
    def calculate_sigma_s(T_celsius, m_s, m_w):
        """Tensão superficial (N/m)"""
        sigma_w = 0.0761 - 0.000155 * T_celsius  # Água pura
        if m_w > 0:
            salinidade_efetiva = m_s / (m_s + m_w)
            return sigma_w * (1 + 0.1 * salinidade_efetiva)
        return sigma_w
    
    @staticmethod
    def calcular_phi_s(m_s, M_NaCl, m_w):
        """Coeficiente osmótico do NaCl"""
        if m_w > 0:
            molalidade = (m_s / M_NaCl) / m_w
            # Aproximação simplificada
            return 0.93 + 0.05 * molalidade
        return 1.0
    
    @staticmethod
    def calcular_rho_spray(rho_ww, m_s, m_w, v_a, M_NaCl):
        """Densidade da água do mar na gota (kg/m³)"""
        if m_w > 0:
            return (m_s + m_w) / ((m_w / rho_ww) + (m_s / M_NaCl) * v_a)
        return rho_ww
    
    @staticmethod
    def calcular_H_estrela(T_K, S):
        """Constante de Henry para CO2 (mol/(L·atm))"""
        T_celsius = T_K - 273.15
        # Fórmula empírica simplificada
        H0 = 3.4e-2  # a 25°C
        return H0 * np.exp(2400 * (1/T_K - 1/298.15))
    
    @staticmethod
    def calcular_Dg_estrela(r, T_K, R_atm):
        """Difusividade do CO2 no ar (m²/s)"""
        # Difusividade base a 25°C e 1 atm
        D0 = 1.6e-5
        return D0 * (T_K / 298.15)**1.75

# Criar instância global
aux = FuncoesAuxiliares()

# ==============================================================================
# SEÇÃO 2: MÉTODO DE NEWTON PARA RESOLVER EQUAÇÕES NÃO-LINEARES
# ==============================================================================

def metodo_newton(f, df, x0, tol=1e-6, max_iter=50):
    """
    Método de Newton para encontrar raízes de f(x) = 0
    
    Parâmetros:
    -----------
    f : callable
        Função cuja raiz se deseja encontrar
    df : callable
        Derivada da função f
    x0 : float
        Chute inicial
    tol : float
        Tolerância para convergência
    max_iter : int
        Número máximo de iterações
        
    Retorna:
    --------
    x : float
        Aproximação da raiz
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-12:
            raise ValueError("Derivada muito próxima de zero")
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new
        
        x = x_new
    
    raise ValueError(f"Método de Newton não convergiu em {max_iter} iterações")

# ==============================================================================
# SEÇÃO 3: CONSTANTES E PARÂMETROS DO SISTEMA
# ==============================================================================

# Constantes físicas
T_a = 18.0              # Temperatura do ar (°C)
T_a_em_k = 291.15       # Temperatura do ar (K)
T_mar_em_k = 293.15     # Temperatura do mar (K)
T_gota = 19.8           # Temperatura da gota (°C)
T_gota_em_k = 292.95    # Temperatura da gota (K)
P = 1000.0              # Pressão (mb)
M_H2O = 18.016e-3       # Massa molecular H2O (kg/mol)
r_i = 100e-6            # Raio inicial (m)
R = 8.31                # Constante universal dos gases (J/(mol·K))
M_NaCl = 58.443e-3      # Massa molecular NaCl (kg/mol)
rho_w = 1025.0          # Densidade água do mar (kg/m³)
v_ion = 2               # Número de íons por molécula NaCl
s = 34/1000             # Salinidade fracionária
f = 0.9                 # Umidade relativa fracionária
C_ar = 0.0154           # Concentração CO2 no ar
S = 34.0                # Salinidade (psu)
R_atm = 0.082           # Constante gases (L·atm/(mol·K))
g = 9.81                # Gravidade (m/s²)
v_ar = 1.32e-5          # Viscosidade cinemática do ar (m²/s)
rho_ar = 1.225          # Densidade do ar (kg/m³)
H_s = 6.0               # Altura significativa da onda (m)
T0 = 273.15             # Temperatura de referência (K)
P0 = 1013.25            # Pressão de referência (mb)
c_ps = 4000.0           # Calor específico da solução (J/(kg·K))

print("="*80)
print("SISTEMA DE EQUAÇÕES DIFERENCIAIS ORDINÁRIAS - DINÂMICA DE GOTAS")
print("Método: Runge-Kutta 4ª ordem com controle adaptativo de passo")
print("="*80)

# ==============================================================================
# SEÇÃO 4: CÁLCULO DA VELOCIDADE DE QUEDA E TEMPO DE VIDA
# ==============================================================================

print("\n[1/4] Calculando velocidade de queda da gota...")

def func_Uf(U_f):
    """Equação para velocidade de queda"""
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    return U_f - const / a

def dfunc_Uf(U_f):
    """Derivada da equação para velocidade de queda"""
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    d_a = -0.158 * (2/3) * (2 * r_i / v_ar)**(2/3) * U_f**(-1/3)
    return 1 - (const * d_a) / (a**2)

U_f0 = 0.1
U_f_sol = metodo_newton(func_Uf, dfunc_Uf, U_f0)
tau_f = H_s / (2 * U_f_sol)

print(f"   U_f = {U_f_sol:.6f} m/s")
print(f"   τ_f = {tau_f:.3f} s (tempo de vida da gota)")

# ==============================================================================
# SEÇÃO 5: CÁLCULO DOS PARÂMETROS DE EQUILÍBRIO
# ==============================================================================

print("\n[2/4] Calculando parâmetros de equilíbrio...")

# Propriedades iniciais
e_sat_val = aux.calcular_esat(T_a)
L_v_val = aux.calcular_lv(T_gota)
D_linha_w_val = aux.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O, T0, P0, 0.036, 8e-8)
k_linha_a_val = aux.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R, T0, P0, 28.9644e-3, 0.7, 2.16e-7, 1.006e3)

rho_ww = aux.calcular_rho_ww(T_gota)
m_s_val = aux.calcular_ms(rho_ww, r_i, s)
m_w_val = rho_ww * (4/3) * np.pi * r_i**3 - m_s_val
sigma_s_val = aux.calculate_sigma_s(T_gota, m_s_val, m_w_val)
Phi_s_val = aux.calcular_phi_s(m_s_val, M_NaCl, m_w_val)
v_a = 0.0  # Simplificação
rho_s_val = aux.calcular_rho_spray(rho_ww, m_s_val, m_w_val, v_a, M_NaCl)

# Temperatura de equilíbrio (simplificada)
T_eq = T_a + 1.0  # Aproximação
T_eq_em_k = T_eq + 273.15

# Raio de equilíbrio via Newton
def zeta(r):
    term1 = (f - 1)
    term2 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r)
    m_w_temp = (4/3) * np.pi * rho_s_val * r**3 - m_s_val
    if m_w_temp <= 0:
        return 1e10
    term3 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / m_w_temp
    return term1 - term2 + term3

def dzeta_dr(r):
    term1 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * (r**2))
    denominator = (4 * np.pi * rho_s_val * (r**3)) / 3 - m_s_val
    if abs(denominator) < 1e-20:
        return -1e10
    term2 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl) * 4 * np.pi * rho_s_val * (r**2)) / (denominator**2)
    return term1 - term2

r0 = 1.1 * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1/3)
r_eq = metodo_newton(zeta, dzeta_dr, r0)

# Escalas de tempo características (estimativas simplificadas)
tau_T = rho_s_val * r_i**2 * c_ps / (3 * k_linha_a_val)
tau_r = rho_s_val * r_i**2 / (D_linha_w_val * rho_w)

print(f"   T_eq = {T_eq:.2f} °C ({T_eq_em_k:.2f} K)")
print(f"   r_eq = {r_eq*1e6:.3f} μm")
print(f"   τ_T = {tau_T:.3e} s (escala de tempo térmica)")
print(f"   τ_r = {tau_r:.3e} s (escala de tempo do raio)")

# ==============================================================================
# SEÇÃO 6: DEFINIÇÃO DO SISTEMA DE EDOs
# ==============================================================================

print("\n[3/4] Definindo sistema de EDOs...")

# Constantes calculadas uma vez
rho_v = f * M_H2O * e_sat_val / (R * T_a_em_k)

def f_system(t, variaveis_gota):
    """
    Sistema de EDOs para a dinâmica da gota
    
    Variáveis: [m, r, T]
    - m: massa de CO2 na gota (mol)
    - r: raio da gota (m)
    - T: temperatura da gota (K)
    """
    m, r, T = variaveis_gota
    
    # Proteções numéricas
    r = max(r, 1e-9)
    T = max(T, 200)
    
    # Atualizar propriedades dependentes do estado
    T_celsius = T - 273.15
    m_w = (4/3) * np.pi * r**3 * rho_s_val - m_s_val
    
    if m_w <= 0:
        m_w = 1e-20
    
    sigma_novo = aux.calculate_sigma_s(T_celsius, m_s_val, m_w)
    Phi_novo = aux.calcular_phi_s(m_s_val, M_NaCl, m_w)
    
    # Termo de Kelvin e Raoult
    exp_y = (2 * M_H2O * sigma_novo / (R * T * rho_w * r)) - \
            (v_ion * Phi_novo * m_s_val * (M_H2O / M_NaCl) / m_w)
    
    Y = (2 * M_H2O * sigma_novo / (R * T_a_em_k * rho_w * r)) - \
        (v_ion * Phi_novo * m_s_val * (M_H2O / M_NaCl) / m_w)
    
    # Propriedades de transporte atualizadas
    e_sat_novo = aux.calcular_esat(T_celsius)
    rho_vr_novo = M_H2O * e_sat_novo * np.exp(exp_y) / (R * T)
    L_v_novo = aux.calcular_lv(T_celsius)
    D_linha_w_novo = aux.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0, 0.036, 8e-8)
    k_linha_a_novo = aux.calculate_K_linha_a(T_celsius, T, P, r, R, T0, P0, 
                                              28.9644e-3, 0.7, 2.16e-7, 1.006e3)
    
    # dm/dt - Taxa de absorção de CO2
    H_novo = aux.calcular_H_estrela(T, S)
    Dg_novo = aux.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4/3) * np.pi * r**3
    C_gota = m / vol if vol > 0 else 0
    
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))
    
    # dr/dt - Taxa de mudança do raio
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * \
           (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2
    
    dr_dt = ((f - 1) - Y) / (r * den) if abs(den) > 1e-20 else 0
    
    # dT/dt - Taxa de mudança da temperatura
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * \
            (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v - rho_vr_novo))
    
    return np.array([dm_dt, dr_dt, dT_dt])

print("   Sistema de 3 EDOs definido:")
print("   - dm/dt: absorção de CO2")
print("   - dr/dt: evaporação/condensação")
print("   - dT/dt: transferência de calor")

# ==============================================================================
# SEÇÃO 7: INTEGRAÇÃO NUMÉRICA COM RK4 ADAPTATIVO
# ==============================================================================

print("\n[4/4] Integrando sistema com RK4 adaptativo...")

# Condições iniciais
m0 = 1.2e-15  # Massa inicial de CO2 (mol)
variaveis_iniciais = np.array([m0, r_i, T_gota_em_k])

# Parâmetros de integração
t_inicial = 0.0
t_final = min(tau_f, 1e3)  # Limitar tempo de integração
h_inicial = 1e-4
tolerancia = 1e-6

print(f"\n   Condições iniciais:")
print(f"   - m0 = {m0:.3e} mol")
print(f"   - r0 = {r_i*1e6:.1f} μm")
print(f"   - T0 = {T_gota_em_k:.2f} K ({T_gota:.2f} °C)")
print(f"\n   Parâmetros de integração:")
print(f"   - Intervalo: [0, {t_final:.2f}] s")
print(f"   - Passo inicial: {h_inicial:.3e} s")
print(f"   - Tolerância: {tolerancia:.3e}")

# Executar integração
inicio_completo = time.time()

resultado = integrar_rk4_adaptativo(
    f_system, 
    variaveis_iniciais, 
    (t_inicial, t_final),
    h_inicial=h_inicial,
    tol=tolerancia,
    h_min=1e-10,
    h_max=1.0,
    max_passos=1000000
)

tempo_computacional = time.time() - inicio_completo

# Extrair resultados
tempo = resultado['t']
variaveis_finais = resultado['y']
h_history = resultado['h_history']
erro_history = resultado['erro_history']
n_passos = resultado['n_passos']

# ==============================================================================
# SEÇÃO 8: SOLUÇÕES ANALÍTICAS APROXIMADAS
# ==============================================================================

print("\nCalculando soluções analíticas aproximadas...")

def raio_simplificado(t, r0, r_eq, tau_r):
    """Solução aproximada para o raio"""
    return r_eq + (r0 - r_eq) * np.exp(-t / tau_r)

def temperatura_simplificada(t, T0, T_eq, tau_T):
    """Solução aproximada para a temperatura"""
    return T_eq + (T0 - T_eq) * np.exp(-t / tau_T)

raio_analítico = raio_simplificado(tempo, r_i, r_eq, tau_r)
temp_analítica = temperatura_simplificada(tempo, T_gota_em_k, T_eq_em_k, tau_T)

# ==============================================================================
# SEÇÃO 9: VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================

print("\nGerando gráficos...")

plt.rcParams['text.usetex'] = False
fig = plt.figure(figsize=(16, 12))

# Layout dos subplots
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Subplot 1: Massa de CO2
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tempo, variaveis_finais[:, 0]*1e15, '#2B00FF', linewidth=2.5, 
         label=f'RK4 Adaptativo')
ax1.set_xlabel('Tempo (s)', fontsize=11)
ax1.set_ylabel('Massa de CO₂ (×10⁻¹⁵ mol)', fontsize=11)
ax1.set_xscale('log')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Evolução da Massa de CO₂', fontsize=12, fontweight='bold')

# Subplot 2: Raio
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(tempo, variaveis_finais[:, 1]*1e6, '#FF0000', linewidth=2.5, 
         label=f'Numérico: {variaveis_finais[-1, 1]*1e6:.3f} μm')
ax2.plot(tempo, raio_analítico*1e6, '#FF0000', linewidth=2, linestyle='--',
         label=f'Analítico: {raio_analítico[-1]*1e6:.3f} μm', alpha=0.7)
ax2.set_xlabel('Tempo (s)', fontsize=11)
ax2.set_ylabel('Raio (μm)', fontsize=11)
ax2.set_xscale('log')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title('Evolução do Raio', fontsize=12, fontweight='bold')

# Subplot 3: Temperatura
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(tempo, variaveis_finais[:, 2]-273.15, '#00AA00', linewidth=2.5,
         label=f'Numérico: {variaveis_finais[-1, 2]-273.15:.2f} °C')
ax3.plot(tempo, temp_analítica-273.15, '#00AA00', linewidth=2, linestyle='--',
         label=f'Analítico: {temp_analítica[-1]-273.15:.2f} °C', alpha=0.7)
ax3.set_xlabel('Tempo (s)', fontsize=11)
ax3.set_ylabel('Temperatura (°C)', fontsize=11)
ax3.set_xscale('log')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_title('Evolução da Temperatura', fontsize=12, fontweight='bold')

# Subplot 4: Tamanho do passo h(t)
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(tempo[:-1], h_history, '#FF8C00', linewidth=1.5)
ax4.set_xlabel('Tempo (s)', fontsize=11)
ax4.set_ylabel('Tamanho do passo h (s)', fontsize=11)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
ax4.set_title('Controle Adaptativo - Tamanho do Passo', fontsize=12, fontweight='bold')

# Subplot 5: Erro estimado
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(tempo[:-1], erro_history, '#8B008B', linewidth=1.5)
ax5.axhline(y=tolerancia, color='r', linestyle='--', linewidth=2, label=f'Tolerância = {tolerancia:.0e}')
ax5.set_xlabel('Tempo (s)', fontsize=11)
ax5.set_ylabel('Erro estimado', fontsize=11)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_title('Estimativa do Erro Local', fontsize=12, fontweight='bold')

# Subplot 6: Distribuição dos passos
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(np.log10(h_history), bins=50, color='#1E90FF', edgecolor='black', alpha=0.7)
ax6.set_xlabel('log₁₀(h)', fontsize=11)
ax6.set_ylabel('Frequência', fontsize=11)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_title('Distribuição dos Tamanhos de Passo', fontsize=12, fontweight='bold')

# Subplot 7: Trajetória no espaço de fase (r vs T)
ax7 = fig.add_subplot(gs[2, 0])
scatter = ax7.scatter(variaveis_finais[:, 1]*1e6, variaveis_finais[:, 2]-273.15, 
                     c=tempo, cmap='viridis', s=10, alpha=0.6)
ax7.plot(r_eq*1e6, T_eq, 'r*', markersize=20, label='Equilíbrio')
ax7.set_xlabel('Raio (μm)', fontsize=11)
ax7.set_ylabel('Temperatura (°C)', fontsize=11)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Tempo (s)', fontsize=9)
ax7.set_title('Trajetória no Espaço de Fase', fontsize=12, fontweight='bold')

# Subplot 8: Comparação r(t) - Erro relativo
ax8 = fig.add_subplot(gs[2, 1])
erro_r = np.abs((variaveis_finais[:, 1] - raio_analítico) / raio_analítico) * 100
ax8.plot(tempo, erro_r, '#DC143C', linewidth=2)
ax8.set_xlabel('Tempo (s)', fontsize=11)
ax8.set_ylabel('Erro relativo (%)', fontsize=11)
ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.grid(True, alpha=0.3)
ax8.set_title('Erro Relativo: Raio Numérico vs Analítico', fontsize=12, fontweight='bold')

# Subplot 9: Estatísticas da integração
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

stats_text = f"""
ESTATÍSTICAS DA INTEGRAÇÃO
{'='*40}

Método: RK4 com Step Doubling
Número de passos: {n_passos:,}
Tempo computacional: {tempo_computacional:.3f} s

Tamanho do passo:
  • Inicial: {h_inicial:.3e} s
  • Mínimo: {np.min(h_history):.3e} s
  • Médio: {np.mean(h_history):.3e} s
  • Máximo: {np.max(h_history):.3e} s

Erro estimado:
  • Tolerância: {tolerancia:.3e}
  • Médio: {np.mean(erro_history):.3e}
  • Máximo: {np.max(erro_history):.3e}

Resultados finais (t = {tempo[-1]:.2f} s):
  • Massa CO₂: {variaveis_finais[-1, 0]:.3e} mol
  • Raio: {variaveis_finais[-1, 1]*1e6:.3f} μm
  • Temperatura: {variaveis_finais[-1, 2]-273.15:.2f} °C

Valores de equilíbrio:
  • r_eq: {r_eq*1e6:.3f} μm
  • T_eq: {T_eq:.2f} °C
"""

ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Título principal
fig.suptitle(f'Sistema de EDOs - Dinâmica de Gotas com Spray Marítimo\n' + 
             f'RK4 com Controle Adaptativo de Passo (Step Doubling) - r₀ = {r_i*1e6:.0f} μm, H_s = {H_s} m',
             fontsize=14, fontweight='bold')

plt.savefig('/home/claude/resultados_rk4_adaptativo.png', dpi=150, bbox_inches='tight')
print("   Gráfico salvo: resultados_rk4_adaptativo.png")

# ==============================================================================
# SEÇÃO 10: SUMÁRIO DOS RESULTADOS
# ==============================================================================

print("\n" + "="*80)
print("RESUMO DOS RESULTADOS")
print("="*80)
print(f"\nParâmetros do Sistema:")
print(f"  • Raio inicial: {r_i*1e6:.1f} μm")
print(f"  • Temperatura inicial: {T_gota:.2f} °C")
print(f"  • Altura significativa da onda: {H_s} m")
print(f"  • Tempo de vida da gota: {tau_f:.3f} s")
print(f"\nEstatísticas da Integração:")
print(f"  • Método: RK4 com step doubling")
print(f"  • Número de passos: {n_passos:,}")
print(f"  • Tempo computacional: {tempo_computacional:.3f} s")
print(f"  • Passo médio: {np.mean(h_history):.3e} s")
print(f"  • Erro médio: {np.mean(erro_history):.3e}")
print(f"\nResultados Finais (t = {tempo[-1]:.2f} s):")
print(f"  • Massa de CO₂: {variaveis_finais[-1, 0]:.6e} mol")
print(f"  • Raio: {variaveis_finais[-1, 1]*1e6:.3f} μm")
print(f"  • Temperatura: {variaveis_finais[-1, 2]-273.15:.2f} °C")
print(f"\nValores de Equilíbrio:")
print(f"  • r_eq: {r_eq*1e6:.3f} μm")
print(f"  • T_eq: {T_eq:.2f} °C")
print(f"\nEficiência:")
print(f"  • Passos/segundo de simulação: {n_passos/tempo[-1]:.1f}")
print(f"  • Custo computacional: {tempo_computacional/tempo[-1]:.3f} s/s_simulado")
print("="*80)

plt.show()
