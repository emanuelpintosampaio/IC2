import math

def calcular_esat2(T_a):
    return 611*math.exp(17.27*T_a/(T_a+237.3))
