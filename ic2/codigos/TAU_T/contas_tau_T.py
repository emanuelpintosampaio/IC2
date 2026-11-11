def calculate_tau_T(rho_s , r_i,k_linha_a,D_linha_w,partial_rho_vr,L_v,c_ps = 4000):
    term1 = 3/(rho_s * c_ps * ( r_i **2))
    term2 = k_linha_a + L_v * D_linha_w * partial_rho_vr
    A = term1 * term2
    return 1/A
