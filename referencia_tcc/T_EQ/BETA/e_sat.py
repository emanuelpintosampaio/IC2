import math

def calcular_esat(T_a):
    return 611*math.exp(17.27*T_a/(T_a+237.3))

"""
Modo de usar
T_a = 18   # Temperatura do ar em °C

e_sat = calcular_esat(T_a)
print(r'O valor de e_sat é:',e_sat)
"""