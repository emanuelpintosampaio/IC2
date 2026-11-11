def  calcular_derivada_H2(r_i,r_eq,r_tau_r2,partial_r_tau_r2):
    return (((r_i - r_eq) / (r_tau_r2 - r_eq)) * partial_r_tau_r2)
