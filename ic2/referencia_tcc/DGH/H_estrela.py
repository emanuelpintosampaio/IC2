import numpy as np

def calcular_H_estrela(T_mar_em_k, S):
    alpha1 = -58.0931
    alpha2 = 90.5069
    alpha3 = 22.2940
    beta1 = 0.027766
    beta2 = -0.025888
    beta3 = 0.0050578
    G1 = alpha1 + alpha2 * (100 / T_mar_em_k) + alpha3 * np.log(T_mar_em_k / 100)
    G2 = S * (beta1 + beta2 * (T_mar_em_k / 100) + beta3 * (T_mar_em_k / 100) ** 2)
    return np.exp(G1 + G2)

