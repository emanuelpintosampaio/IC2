import numpy as np
import matplotlib.pyplot as plt

# H_estrela
alpha1 = -58.0931
alpha2 = 90.5069
alpha3 = 22.2940
beta1 = 0.027766
beta2 = -0.025888
beta3 = 0.0050578
T_mar = 285
S_mar = 34
G1 = alpha1 + alpha2 * (100 / T_mar) + alpha3 * np.log(T_mar / 100)
G2 = S_mar * (beta1 + beta2 * (T_mar / 100) + beta3 * (T_mar / 100) ** 2)
H_estrela = np.exp(G1 + G2)

# Dg_estrela
T_ar = 290
Dg = 0.16
alpha_epsilon = 14 * (10**-5)
M_CO2 = 0.044
r_i = 30 * (10**-6)  # Raio inicial
R = 0.082
v_epsilon = np.sqrt(8 * R * T_ar / (np.pi * M_CO2))
Dg_estrela = Dg / (1 + ((4 * Dg) / (r_i * alpha_epsilon * v_epsilon)))

# Parâmetros adicionais
C_ar = 0.0154  # Concentração do gás carbônico no ar
tau_r =22.232418077492976  # Constante de tempo para o raio 
r_eq = 1.1508935799340816e-05  # Raio de equilíbrio 

# Ajustar o passo de tempo
dt = 5e-6  # Passo de tempo (em segundos)
t_total = 26.59505253244308  # Tempo total de simulação (em segundos)
n_steps = int(t_total / dt)  # Número de passos

# Vetores para armazenar os resultados
tempo = np.linspace(0, t_total, n_steps + 1)
massa = np.zeros(n_steps + 1)
raio = np.zeros(n_steps + 1)

# Condições iniciais
massa[0] = 1.2 * (10**-15)  # Massa inicial da gotícula (em kg)
raio[0] = r_i  # Raio inicial da gotícula

# Loop temporal - Euler Backward
for n in range(n_steps):
    # Raio da gotícula no próximo passo
    r_next = r_eq + (r_i - r_eq) * np.exp(-tempo[n + 1] / tau_r)
    raio[n + 1] = r_next

    # Coeficientes do esquema de Euler-Backward
    numerador = massa[n] + dt * 4 * np.pi * r_next * Dg_estrela * C_ar
    denominador = 1 + dt * (3 * Dg_estrela) / ((r_next**2) * H_estrela * R * T_ar)

    # Massa no próximo passo
    massa[n + 1] = numerador / denominador

# Plotando os resultados
plt.figure(figsize=(8, 10))

# Gráfico da massa
plt.subplot(2, 1, 1)
plt.loglog(tempo, massa, label="Massa da gotícula")
plt.xlabel(r't (s)')
plt.ylabel(r'm(t) (mol)')
plt.legend()

# Gráfico do raio
plt.subplot(2, 1, 2)
plt.loglog(tempo, raio, label="Raio da gotícula")
plt.xlabel(r't (s)')
plt.ylabel(r'r(t) ($\mu m$)')
plt.legend()

plt.tight_layout()
plt.show()
