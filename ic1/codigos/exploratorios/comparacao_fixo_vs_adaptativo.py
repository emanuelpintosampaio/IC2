"""
Comparação Direta: RK4 Passo Fixo vs RK4 Adaptativo
Demonstra ganhos de eficiência e precisão
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from rk4_adaptativo import integrar_rk4_adaptativo

print("="*80)
print("COMPARAÇÃO: RK4 PASSO FIXO vs RK4 ADAPTATIVO")
print("="*80)

# ==============================================================================
# Sistema de Teste: Problema de Múltiplas Escalas
# ==============================================================================

print("\nSistema de teste: Reação química com múltiplas escalas de tempo")
print("-" * 80)

# Constantes
k_rapido = 100.0   # Taxa de reação rápida (τ₁ ~ 0.01 s)
k_lento = 0.1      # Taxa de reação lenta (τ₂ ~ 10 s)

def sistema_multiescala(t, y):
    """
    Sistema com duas escalas de tempo:
    dA/dt = -k_rapido·A + k_lento·B    (rápido)
    dB/dt = +k_rapido·A - k_lento·B    (lento)
    """
    A, B = y
    dA_dt = -k_rapido * A + k_lento * B
    dB_dt = +k_rapido * A - k_lento * B
    return np.array([dA_dt, dB_dt])

# Condições iniciais
y0 = np.array([1.0, 0.0])
t_inicial = 0.0
t_final = 50.0

print(f"Condições iniciais: A(0) = {y0[0]}, B(0) = {y0[1]}")
print(f"Intervalo de integração: [{t_inicial}, {t_final}] s")
print(f"Escala de tempo rápida: τ₁ ~ {1/k_rapido:.4f} s")
print(f"Escala de tempo lenta: τ₂ ~ {1/k_lento:.1f} s")

# ==============================================================================
# MÉTODO 1: RK4 COM PASSO FIXO
# ==============================================================================

print("\n" + "="*80)
print("MÉTODO 1: RK4 COM PASSO FIXO")
print("="*80)

def rk4_passo_fixo(f, y0, t_span, dt):
    """RK4 clássico com passo fixo"""
    t_inicial, t_final = t_span
    t = np.arange(t_inicial, t_final + dt, dt)
    n_steps = len(t)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    
    for i in range(n_steps - 1):
        h = t[i+1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

# Testar diferentes passos fixos
passos_fixos = [1e-3, 1e-2, 1e-1]
resultados_fixo = {}

print("\nTestando diferentes tamanhos de passo fixo:")
print(f"{'Passo dt':<15} {'N° Passos':<15} {'Tempo (s)':<15} {'A_final':<15} {'B_final':<15}")
print("-" * 80)

for dt in passos_fixos:
    inicio = time.time()
    t_fixo, y_fixo = rk4_passo_fixo(sistema_multiescala, y0, (t_inicial, t_final), dt)
    tempo_comp = time.time() - inicio
    
    resultados_fixo[dt] = {
        't': t_fixo,
        'y': y_fixo,
        'n_passos': len(t_fixo),
        'tempo': tempo_comp
    }
    
    print(f"{dt:<15.0e} {len(t_fixo):<15,} {tempo_comp:<15.4f} {y_fixo[-1,0]:<15.8f} {y_fixo[-1,1]:<15.8f}")

# ==============================================================================
# MÉTODO 2: RK4 ADAPTATIVO
# ==============================================================================

print("\n" + "="*80)
print("MÉTODO 2: RK4 ADAPTATIVO (STEP DOUBLING)")
print("="*80)

# Testar diferentes tolerâncias
tolerancias = [1e-4, 1e-6, 1e-8]
resultados_adaptativo = {}

print("\nTestando diferentes tolerâncias:")
print(f"{'Tolerância':<15} {'N° Passos':<15} {'Tempo (s)':<15} {'A_final':<15} {'B_final':<15}")
print("-" * 80)

for tol in tolerancias:
    inicio = time.time()
    resultado = integrar_rk4_adaptativo(
        sistema_multiescala,
        y0,
        (t_inicial, t_final),
        h_inicial=1e-2,
        tol=tol,
        h_min=1e-10,
        h_max=1.0
    )
    tempo_comp = time.time() - inicio
    
    resultados_adaptativo[tol] = {
        'resultado': resultado,
        'tempo': tempo_comp
    }
    
    y_final = resultado['y'][-1]
    print(f"{tol:<15.0e} {resultado['n_passos']:<15,} {tempo_comp:<15.4f} {y_final[0]:<15.8f} {y_final[1]:<15.8f}")

# ==============================================================================
# COMPARAÇÃO DE EFICIÊNCIA
# ==============================================================================

print("\n" + "="*80)
print("ANÁLISE COMPARATIVA")
print("="*80)

# Escolher caso base para comparação
dt_ref = 1e-3  # Passo fixo de referência (mais preciso)
tol_ref = 1e-6  # Tolerância de referência

y_referencia = resultados_fixo[dt_ref]['y'][-1]
y_adaptativo = resultados_adaptativo[tol_ref]['resultado']['y'][-1]

print(f"\nResultado de Referência (dt = {dt_ref:.0e}):")
print(f"  A_final = {y_referencia[0]:.10f}")
print(f"  B_final = {y_referencia[1]:.10f}")

print(f"\nResultado Adaptativo (tol = {tol_ref:.0e}):")
print(f"  A_final = {y_adaptativo[0]:.10f}")
print(f"  B_final = {y_adaptativo[1]:.10f}")

print(f"\nErro Relativo:")
erro_A = abs(y_adaptativo[0] - y_referencia[0]) / abs(y_referencia[0])
erro_B = abs(y_adaptativo[1] - y_referencia[1]) / abs(y_referencia[1])
print(f"  Componente A: {erro_A:.3e}")
print(f"  Componente B: {erro_B:.3e}")

print(f"\nEficiência:")
n_passos_fixo = resultados_fixo[dt_ref]['n_passos']
n_passos_adapt = resultados_adaptativo[tol_ref]['resultado']['n_passos']
tempo_fixo = resultados_fixo[dt_ref]['tempo']
tempo_adapt = resultados_adaptativo[tol_ref]['tempo']

print(f"  Passo Fixo: {n_passos_fixo:,} passos em {tempo_fixo:.3f} s")
print(f"  Adaptativo: {n_passos_adapt:,} passos em {tempo_adapt:.3f} s")
print(f"  Redução de passos: {n_passos_fixo/n_passos_adapt:.1f}×")
print(f"  Speedup: {tempo_fixo/tempo_adapt:.1f}×")

# ==============================================================================
# VISUALIZAÇÃO
# ==============================================================================

print("\nGerando gráficos comparativos...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Comparação das soluções A(t)
ax1 = fig.add_subplot(gs[0, :2])
for dt in passos_fixos:
    res = resultados_fixo[dt]
    ax1.plot(res['t'], res['y'][:, 0], label=f'Fixo dt={dt:.0e}', 
             linewidth=2, alpha=0.7)

res_adapt = resultados_adaptativo[1e-6]['resultado']
ax1.plot(res_adapt['t'], res_adapt['y'][:, 0], 'k--', 
         label=f'Adaptativo tol=1e-6', linewidth=2.5)

ax1.set_xlabel('Tempo (s)', fontsize=11)
ax1.set_ylabel('Concentração A(t)', fontsize=11)
ax1.set_title('Comparação das Soluções - Componente A', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Comparação B(t)
ax2 = fig.add_subplot(gs[0, 2])
for dt in passos_fixos:
    res = resultados_fixo[dt]
    ax2.plot(res['t'], res['y'][:, 1], label=f'dt={dt:.0e}', linewidth=2, alpha=0.7)

res_adapt = resultados_adaptativo[1e-6]['resultado']
ax2.plot(res_adapt['t'], res_adapt['y'][:, 1], 'k--', 
         label=f'tol=1e-6', linewidth=2.5)

ax2.set_xlabel('Tempo (s)', fontsize=11)
ax2.set_ylabel('Concentração B(t)', fontsize=11)
ax2.set_title('Componente B', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Espaço de fase
ax3 = fig.add_subplot(gs[1, 0])
for dt in passos_fixos:
    res = resultados_fixo[dt]
    ax3.plot(res['y'][:, 0], res['y'][:, 1], label=f'dt={dt:.0e}', 
             linewidth=1.5, alpha=0.7)

res_adapt = resultados_adaptativo[1e-6]['resultado']
ax3.plot(res_adapt['y'][:, 0], res_adapt['y'][:, 1], 'k--', 
         label='Adaptativo', linewidth=2.5)

ax3.set_xlabel('A', fontsize=11)
ax3.set_ylabel('B', fontsize=11)
ax3.set_title('Espaço de Fase', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Controle do passo adaptativo
ax4 = fig.add_subplot(gs[1, 1])
for tol in tolerancias:
    res = resultados_adaptativo[tol]['resultado']
    ax4.loglog(res['t'][:-1], res['h_history'], 
               label=f'tol={tol:.0e}', linewidth=2, alpha=0.7)

ax4.set_xlabel('Tempo (s)', fontsize=11)
ax4.set_ylabel('Tamanho do Passo h (s)', fontsize=11)
ax4.set_title('Controle Adaptativo do Passo', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Erro estimado
ax5 = fig.add_subplot(gs[1, 2])
for tol in tolerancias:
    res = resultados_adaptativo[tol]['resultado']
    ax5.loglog(res['t'][:-1], res['erro_history'], 
               label=f'tol={tol:.0e}', linewidth=2, alpha=0.7)
    ax5.axhline(y=tol, linestyle='--', linewidth=1, alpha=0.5)

ax5.set_xlabel('Tempo (s)', fontsize=11)
ax5.set_ylabel('Erro Estimado', fontsize=11)
ax5.set_title('Controle do Erro', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Eficiência: Número de passos
ax6 = fig.add_subplot(gs[2, 0])
labels_fixo = [f'{dt:.0e}' for dt in passos_fixos]
n_passos_fixo_list = [resultados_fixo[dt]['n_passos'] for dt in passos_fixos]
labels_adapt = [f'{tol:.0e}' for tol in tolerancias]
n_passos_adapt_list = [resultados_adaptativo[tol]['resultado']['n_passos'] 
                        for tol in tolerancias]

x_pos = np.arange(len(labels_fixo))
width = 0.35

ax6.bar(x_pos - width/2, n_passos_fixo_list, width, label='Passo Fixo', color='#FF6B6B')
ax6.bar(x_pos + width/2, n_passos_adapt_list, width, label='Adaptativo', color='#4ECDC4')

ax6.set_xlabel('Precisão (dt ou tol)', fontsize=11)
ax6.set_ylabel('Número de Passos', fontsize=11)
ax6.set_title('Eficiência: Número de Passos', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(labels_fixo)
ax6.legend(fontsize=9)
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3, axis='y')

# Eficiência: Tempo computacional
ax7 = fig.add_subplot(gs[2, 1])
tempo_fixo_list = [resultados_fixo[dt]['tempo'] for dt in passos_fixos]
tempo_adapt_list = [resultados_adaptativo[tol]['tempo'] for tol in tolerancias]

ax7.bar(x_pos - width/2, tempo_fixo_list, width, label='Passo Fixo', color='#FF6B6B')
ax7.bar(x_pos + width/2, tempo_adapt_list, width, label='Adaptativo', color='#4ECDC4')

ax7.set_xlabel('Precisão (dt ou tol)', fontsize=11)
ax7.set_ylabel('Tempo Computacional (s)', fontsize=11)
ax7.set_title('Eficiência: Tempo de Execução', fontsize=12, fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(labels_fixo)
ax7.legend(fontsize=9)
ax7.set_yscale('log')
ax7.grid(True, alpha=0.3, axis='y')

# Tabela de resumo
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

summary_text = f"""
RESUMO COMPARATIVO
{'='*35}

Caso: tol=1e-6 vs dt=1e-3

Número de Passos:
  • Fixo:      {n_passos_fixo:>10,}
  • Adaptativo:{n_passos_adapt:>10,}
  • Redução:   {n_passos_fixo/n_passos_adapt:>10.1f}×

Tempo Computacional:
  • Fixo:      {tempo_fixo:>10.3f} s
  • Adaptativo:{tempo_adapt:>10.3f} s
  • Speedup:   {tempo_fixo/tempo_adapt:>10.1f}×

Precisão:
  • Erro A:    {erro_A:>10.2e}
  • Erro B:    {erro_B:>10.2e}

Passo Adaptativo:
  • Mínimo:    {np.min(res_adapt['h_history']):>10.2e} s
  • Máximo:    {np.max(res_adapt['h_history']):>10.2e} s
  • Razão:     {np.max(res_adapt['h_history'])/np.min(res_adapt['h_history']):>10.1f}×

CONCLUSÃO: Método adaptativo é
{n_passos_fixo/n_passos_adapt:.0f}× mais eficiente com
precisão comparável!
"""

ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Comparação: RK4 Passo Fixo vs RK4 Adaptativo (Step Doubling)\n' +
             'Sistema de Múltiplas Escalas de Tempo',
             fontsize=14, fontweight='bold')

plt.savefig('/home/claude/comparacao_fixo_vs_adaptativo.png', dpi=150, bbox_inches='tight')
print("Gráfico salvo: comparacao_fixo_vs_adaptativo.png")

print("\n" + "="*80)
print("CONCLUSÃO")
print("="*80)
print(f"\nO método RK4 adaptativo com step doubling demonstrou:")
print(f"  ✓ Redução de {n_passos_fixo/n_passos_adapt:.0f}× no número de passos")
print(f"  ✓ Speedup de {tempo_fixo/tempo_adapt:.1f}× no tempo de execução")
print(f"  ✓ Precisão comparável (erro ~ {max(erro_A, erro_B):.2e})")
print(f"  ✓ Adaptação automática às múltiplas escalas de tempo")
print(f"\nO ganho de eficiência é especialmente significativo em sistemas")
print(f"com múltiplas escalas de tempo, como o problema de gotas!")
print("="*80)

plt.show()
