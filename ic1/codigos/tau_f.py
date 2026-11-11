from NEWTON import metodo_newton
import numpy as np

# Dados 
g = 9.81  
v_ar = 1.32e-5  
rho_mar = 1025  
rho_ar = 1.225  
r_i = 500e-6  

# Parâmetros constantes
const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_mar / rho_ar) - 1)

# Função implícita para f(U_f)
def f(U_f):
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    return U_f - const / a

# Derivada da função f(U_f)
def df(U_f):
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    d_a = -0.158 * (2 / 3) * ((2 * r_i / v_ar) ** (2 / 3)) * U_f ** (-1 / 3)
    return 1 - const * d_a / (a ** 2)

# Chute inicial
U_f0 = 0.1  # Chute inicial 

# Parâmetros do método de Newton
epsilon = 1e-6  # Tolerância
Iter = 50  # Número máximo de iterações

# Aplicar o método de Newton
U_f_sol = metodo_newton.Newton(f, df, U_f0, epsilon, Iter)
print("Velocidade de equilíbrio (U_f) encontrada:", U_f_sol)

#----------------------------------------------------------------------------------------

H_s = 6 # Altura significativa da onda
tau_f = H_s / (2 * U_f_sol)  # tau_f  tempo de vida na atmosfera
print(r'Tempo que a gota permanece na atmosfera (𝜏_f) encontrado:',tau_f)