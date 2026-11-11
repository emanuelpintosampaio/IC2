import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from DGH import Dg_estrela, H_estrela 

# Constantes
r_i = 30e-6           # Raio inicial
T_i_em_k = 285        # Temperatura inicial da gota em K
m_0 = 1.2e-15         # Condição Inicial
C_ar =  0.0154        # Concentração do gás carbônico no ar
T_a_em_k = 291.15     # Temperatura do ar em K
T_mar_em_k = 285      # Temperatura do mar em K
S = 34                # Salinidade do mar
R = 0.082             # Constante universal dos gases

# Parâmetros das funções exponenciais (Iury 2018)
tau_r = 21.588038138415623              
tau_T = 0.017809805752997256             
r_eq = 11.508935981764845e-6             
T_eq_em_k = 15.921331148869596 + 273.15  

# Funções para r(t) e T(t)
def raio(t):
    return r_eq + (r_i - r_eq) * np.exp(-t / tau_r)

def temperatura(t):
    return T_eq_em_k + (T_i_em_k - T_eq_em_k) * np.exp(-t / tau_T)

# H_estrela (constante)
H_estrela_valor = H_estrela.calcular_H_estrela(T_mar_em_k, S)

# EDO modificada para considerar casos
def edo_massa(m, t, case):
    # Determinar r(t) e T(t) conforme o caso
    if case == 1:
        r_current = raio(t)
        T_current = temperatura(t)
    elif case == 2:
        r_current = raio(t)
        T_current = T_i_em_k
    elif case == 3:
        r_current = r_i
        T_current = temperatura(t)
    else:
        raise ValueError("Caso inválido")
    
    # Calcular Dg_estrela com raio atual
    Dg_current = Dg_estrela.calcular_Dg_estrela(r_current, T_a_em_k, R)
    
    # Equação da EDO
    dm_dt = 4 * np.pi * r_current * Dg_current * (
        C_ar - (3 * m) / (4 * np.pi * (r_current**3) * H_estrela_valor * R * T_current)
    )
    return dm_dt

# Interface para escolha do caso
print("Opções de simulação:")
print("1 - T(t) e r(t) variando")
print("2 - Somente r(t) variando")
print("3 - Somente T(t) variando")
choice = int(input("Escolha (1-3): "))
cases = {1: 'Ambos variando', 2: 'Raio variando', 3: 'Temperatura variando'}
case = choice if choice in [1,2,3] else 1

# Resolver EDO
t = np.logspace(-4, 2, 500000)
m = odeint(edo_massa, m_0, t, args=(case,))

# Plotagem
subplots = 3 if case == 1 else 2
plt.figure(figsize=(8, 10), dpi=150)
# Subplot 1: Massa
plt.subplot(subplots, 1, 1)
plt.plot(t, m, color='blue', label='Massa')
plt.xscale('log')
plt.xlabel('t (s)')
plt.ylabel('m(t) (mol)')
plt.legend()
plt.grid(True)

# Subplot 2: Raio (casos 1 e 2)
if case in [1, 2]:
    plt.subplot(subplots, 1, 2)
    r_values = raio(t)
    plt.plot(t, r_values * 1e6, color='blue', label='Raio')
    plt.xscale('log')
    plt.xlabel('t (s)')
    plt.ylabel('r(t) (µm)')
    plt.legend()
    plt.grid(True)

# Subplot 3: Temperatura (casos 1 e 3)
if case in [1, 3]:
    pos = 3 if case == 1 else 2
    plt.subplot(subplots, 1, pos)
    T_values = temperatura(t)
    plt.plot(t, T_values, color='blue', label='Temperatura')
    plt.xscale('log')
    plt.xlabel('t (s)')
    plt.ylabel('T(t) (K)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()