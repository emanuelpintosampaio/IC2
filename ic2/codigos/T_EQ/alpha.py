def calcular_alpha(T_a_em_k, a=17.502, b=240.97):
    denominador = (b + T_a_em_k - 273.15) ** 2
    alphaa = (a * b * T_a_em_k) / denominador
    return alphaa
