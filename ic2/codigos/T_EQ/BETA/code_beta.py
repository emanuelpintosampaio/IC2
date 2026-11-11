def calcular_beta(e_sat, T_a_em_k, L_v, M_H2O, D_linha_w, R, k_linha_a):
    betaa = (e_sat / T_a_em_k) * (L_v * M_H2O * D_linha_w) / (R * k_linha_a)
    return betaa

