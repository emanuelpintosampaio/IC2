"""
Teste de validação do método RK4 adaptativo
Compara com soluções analíticas conhecidas
"""

import numpy as np
import matplotlib.pyplot as plt
from rk4_adaptativo import integrar_rk4_adaptativo
import time

print("="*80)
print("VALIDAÇÃO DO MÉTODO RK4 ADAPTATIVO")
print("="*80)

# ==============================================================================
# TESTE 1: Decaimento Exponencial
# ==============================================================================

print("\n[TESTE 1] Decaimento Exponencial: dy/dt = -λy, y(0) = 1")
print("-" * 80)

lambda_decay = 2.0
y0_test1 = 1.0

def f_exponential(t, y):
    """dy/dt = -λy"""
    return np.array([-lambda_decay * y[0]])

def solucao_analitica_exp(t):
    """Solução: y(t) = exp(-λt)"""
    return np.exp(-lambda_decay * t)

# Integração
t_span = (0.0, 5.0)
resultado1 = integrar_rk4_adaptativo(
    f_exponential,
    [y0_test1],
    t_span,
    h_inicial=0.1,
    tol=1e-8
)

# Comparação
t_analitico = resultado1['t']
y_analitica = solucao_analitica_exp(t_analitico)
y_numerica = resultado1['y'][:, 0]
erro_abs = np.abs(y_numerica - y_analitica)
erro_rel = erro_abs / np.abs(y_analitica)

print(f"Resultado:")
print(f"  • Número de passos: {resultado1['n_passos']}")
print(f"  • Erro máximo absoluto: {np.max(erro_abs):.3e}")
print(f"  • Erro máximo relativo: {np.max(erro_rel):.3e}")
print(f"  • Valor final numérico: {y_numerica[-1]:.10f}")
print(f"  • Valor final analítico: {y_analitica[-1]:.10f}")

# ==============================================================================
# TESTE 2: Oscilador Harmônico
# ==============================================================================

print("\n[TESTE 2] Oscilador Harmônico: d²y/dt² = -ω²y")
print("-" * 80)

omega = 2*np.pi  # Frequência angular
y0_test2 = [1.0, 0.0]  # [posição, velocidade]

def f_harmonic(t, y):
    """Sistema: dy1/dt = y2, dy2/dt = -ω²y1"""
    return np.array([y[1], -omega**2 * y[0]])

def solucao_analitica_harm(t):
    """Solução: y(t) = cos(ωt), v(t) = -ω·sin(ωt)"""
    return np.cos(omega * t), -omega * np.sin(omega * t)

# Integração
t_span = (0.0, 10.0)  # Várias oscilações
resultado2 = integrar_rk4_adaptativo(
    f_harmonic,
    y0_test2,
    t_span,
    h_inicial=0.05,
    tol=1e-8
)

# Comparação
t_analitico2 = resultado2['t']
y_pos_num = resultado2['y'][:, 0]
y_vel_num = resultado2['y'][:, 1]
y_pos_ana, y_vel_ana = solucao_analitica_harm(t_analitico2)

erro_pos = np.abs(y_pos_num - y_pos_ana)
erro_vel = np.abs(y_vel_num - y_vel_ana)

print(f"Resultado:")
print(f"  • Número de passos: {resultado2['n_passos']}")
print(f"  • Erro máximo posição: {np.max(erro_pos):.3e}")
print(f"  • Erro máximo velocidade: {np.max(erro_vel):.3e}")

# Energia (conservação)
E_numerica = 0.5 * (y_vel_num**2 + (omega * y_pos_num)**2)
E_inicial = 0.5 * omega**2
erro_energia = np.abs(E_numerica - E_inicial) / E_inicial

print(f"  • Erro relativo na conservação de energia: {np.max(erro_energia):.3e}")

# ==============================================================================
# TESTE 3: Equação de Van der Pol (Sistema Stiff leve)
# ==============================================================================

print("\n[TESTE 3] Van der Pol: d²y/dt² = μ(1-y²)dy/dt - y")
print("-" * 80)

mu = 1.0  # Parâmetro de não-linearidade
y0_test3 = [2.0, 0.0]

def f_vanderpol(t, y):
    """Sistema de Van der Pol"""
    return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])

# Integração
t_span = (0.0, 20.0)
inicio = time.time()
resultado3 = integrar_rk4_adaptativo(
    f_vanderpol,
    y0_test3,
    t_span,
    h_inicial=0.1,
    tol=1e-6
)
tempo_comp = time.time() - inicio

print(f"Resultado:")
print(f"  • Número de passos: {resultado3['n_passos']}")
print(f"  • Tempo computacional: {tempo_comp:.3f} s")
print(f"  • Passo mínimo: {np.min(resultado3['h_history']):.3e} s")
print(f"  • Passo máximo: {np.max(resultado3['h_history']):.3e} s")
print(f"  • Razão passo_max/passo_min: {np.max(resultado3['h_history'])/np.min(resultado3['h_history']):.1f}")

# ==============================================================================
# VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================

print("\n" + "="*80)
print("Gerando gráficos de validação...")
print("="*80)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Teste 1: Decaimento Exponencial
axes[0, 0].plot(resultado1['t'], y_numerica, 'b-', linewidth=2, label='Numérico')
axes[0, 0].plot(t_analitico, y_analitica, 'r--', linewidth=2, label='Analítico')
axes[0, 0].set_xlabel('Tempo (s)')
axes[0, 0].set_ylabel('y(t)')
axes[0, 0].set_title('Teste 1: Decaimento Exponencial')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].semilogy(resultado1['t'], erro_abs, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Tempo (s)')
axes[0, 1].set_ylabel('Erro Absoluto')
axes[0, 1].set_title('Erro vs Tempo')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].loglog(resultado1['t'][:-1], resultado1['h_history'], 'orange', linewidth=2)
axes[0, 2].set_xlabel('Tempo (s)')
axes[0, 2].set_ylabel('Tamanho do Passo h')
axes[0, 2].set_title('Controle Adaptativo')
axes[0, 2].grid(True, alpha=0.3)

# Teste 2: Oscilador Harmônico
axes[1, 0].plot(resultado2['t'], y_pos_num, 'b-', linewidth=2, label='Posição Numérica')
axes[1, 0].plot(t_analitico2, y_pos_ana, 'r--', linewidth=1, label='Posição Analítica')
axes[1, 0].set_xlabel('Tempo (s)')
axes[1, 0].set_ylabel('Posição y(t)')
axes[1, 0].set_title('Teste 2: Oscilador Harmônico')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(y_pos_num, y_vel_num, 'b-', linewidth=2, label='Numérico')
axes[1, 1].plot(y_pos_ana, y_vel_ana, 'r--', linewidth=1, label='Analítico')
axes[1, 1].set_xlabel('Posição y')
axes[1, 1].set_ylabel('Velocidade dy/dt')
axes[1, 1].set_title('Espaço de Fase')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axis('equal')

axes[1, 2].semilogy(resultado2['t'], erro_energia, 'm-', linewidth=2)
axes[1, 2].set_xlabel('Tempo (s)')
axes[1, 2].set_ylabel('Erro Relativo na Energia')
axes[1, 2].set_title('Conservação de Energia')
axes[1, 2].grid(True, alpha=0.3)

# Teste 3: Van der Pol
axes[2, 0].plot(resultado3['t'], resultado3['y'][:, 0], 'b-', linewidth=2)
axes[2, 0].set_xlabel('Tempo (s)')
axes[2, 0].set_ylabel('y(t)')
axes[2, 0].set_title('Teste 3: Van der Pol')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(resultado3['y'][:, 0], resultado3['y'][:, 1], 'b-', linewidth=1.5)
axes[2, 1].set_xlabel('y')
axes[2, 1].set_ylabel('dy/dt')
axes[2, 1].set_title('Ciclo Limite (Espaço de Fase)')
axes[2, 1].grid(True, alpha=0.3)

axes[2, 2].loglog(resultado3['t'][:-1], resultado3['h_history'], 'orange', linewidth=2)
axes[2, 2].set_xlabel('Tempo (s)')
axes[2, 2].set_ylabel('Tamanho do Passo h')
axes[2, 2].set_title('Adaptação do Passo')
axes[2, 2].grid(True, alpha=0.3)

plt.suptitle('Validação do Método RK4 Adaptativo com Step Doubling', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/validacao_rk4_adaptativo.png', dpi=150, bbox_inches='tight')
print("Gráfico salvo: validacao_rk4_adaptativo.png")

# ==============================================================================
# RESUMO FINAL
# ==============================================================================

print("\n" + "="*80)
print("RESUMO DA VALIDAÇÃO")
print("="*80)

print("\n✓ TESTE 1 - Decaimento Exponencial:")
print(f"    Erro máximo relativo: {np.max(erro_rel):.3e}")
if np.max(erro_rel) < 1e-6:
    print("    Status: PASSOU ✓")
else:
    print("    Status: FALHOU ✗")

print("\n✓ TESTE 2 - Oscilador Harmônico:")
print(f"    Erro máximo posição: {np.max(erro_pos):.3e}")
print(f"    Erro conservação energia: {np.max(erro_energia):.3e}")
if np.max(erro_pos) < 1e-5 and np.max(erro_energia) < 1e-4:
    print("    Status: PASSOU ✓")
else:
    print("    Status: FALHOU ✗")

print("\n✓ TESTE 3 - Van der Pol:")
print(f"    Passos adaptativos: {resultado3['n_passos']}")
print(f"    Razão passos: {np.max(resultado3['h_history'])/np.min(resultado3['h_history']):.1f}×")
if resultado3['n_passos'] < 10000:  # Eficiência
    print("    Status: PASSOU ✓")
else:
    print("    Status: ALERTA - Muitos passos")

print("\n" + "="*80)
print("VALIDAÇÃO CONCLUÍDA!")
print("="*80)

plt.show()
