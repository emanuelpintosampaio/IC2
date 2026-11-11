import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from GRAFICO.DGH import Dg_estrela,H_estrela 
from matplotlib.ticker import ScalarFormatter


r_i = 30e-6           # Raio inicial
T_i_em_k = 285        # Temperatura inicial da gota em K
m_0 = 1.2e-15         # Condição Inicial
C_ar =  0.0154        # Concentração do gás carbônico no ar
T_a_em_k = 290        # Temperatura do ar em K
T_mar_em_k = 285      # Temperatura do mar em k
S = 34                # Salinidade do mar
R = 0.082             # Constante universal dos gases

# H_estrela 
H_estrela = H_estrela.calcular_H_estrela(T_mar_em_k, S)

# Dg_estrela
Dg_estrela = Dg_estrela.calcular_Dg_estrela(r_i, T_a_em_k,R)

# Resolvendo a EDO para a massa da gotícula ao longo do tempo
def edo_massa(m,t):
    dm_dt = 4 * np.pi * r_i * Dg_estrela * ((C_ar) -((3 * m) / (4 * np.pi * (r_i**3) * H_estrela * R * T_i_em_k)))
    return dm_dt

t = np.logspace(-4,2, 1000) 
plt.figure(figsize=(10,3))
m = odeint(edo_massa, m_0, t)

# Configuração do gráfico
plt.plot(t, m,color='#00FF00BF',linestyle='-', linewidth=2.5)
plt.xscale('log')   
plt.yscale('log')
plt.xlabel('t (s)')
plt.ylabel('m(t) (mol)')
plt.grid(False)
plt.tight_layout()
plt.show()