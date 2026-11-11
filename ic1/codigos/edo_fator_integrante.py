import numpy as np
import matplotlib.pyplot as plt
from GRAFICO.DGH import Dg_estrela, H_estrela  

r_i = 30e-6           # Raio inicial
T_i_em_k = 285        # Temperatura inicial da gota em K
T_gota = 285          # Temperatura da gota em Kelvin
m_0 = 1.2e-15         # Condição Inicial
C_ar =  0.0154        # Concentração do gás carbônico no ar
T_a_em_k = 291.15     # Temperatura do ar em K
T_mar_em_k = 285      # Temperatura do mar em k
S = 34                # Salinidade do mar
R = 0.082             # Constante universal dos gases
M_CO2 = 0.044         # Massa molar do CO2 (kg/mol)
Dg = 0.16             # Difusividade do CO2 no ar (cm^2/s)

# H_estrela 
H_estrela = H_estrela.calcular_H_estrela(T_mar_em_k, S)

# Dg_estrela
Dg_estrela = Dg_estrela.calcular_Dg_estrela(r_i, T_a_em_k,R)


# Cálculo da constante "c" para um valor específico de m(10^-4)
term1 = (4/3) * np.pi * (r_i**3) * C_ar * H_estrela * R * T_gota  
exponential_factor = np.exp((-3 * Dg_estrela * 1e-4) / ((r_i**2) * H_estrela * R * T_gota))
const_m = 1.2e-15 
c = (const_m - term1) / exponential_factor  

# Definição do vetor de tempo em escala logarítmica
t = np.logspace(-4, 2, 1000)  

# Cálculo da evolução da massa ao longo do tempo
term2 = c*np.exp((-3 * Dg_estrela * t) / ((r_i**2) * H_estrela * R * T_gota))
m = term1 + term2  

# Plotagem do gráfico
plt.figure(figsize=(10, 5)) 
plt.plot(t, m, color='blue', label=r'$m(10^{-4})=1.2 \times 10^{-15}$')  
plt.xscale('log')  
plt.yscale('log')  
plt.xlabel(r't (s)')  
plt.ylabel(r'm(t) (mol)')  
plt.legend(loc='lower right') 
plt.grid(False)  
plt.show()  