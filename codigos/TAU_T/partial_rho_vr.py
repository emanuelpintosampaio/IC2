def calcular_partial_rho_vr(T_a,rho_vr,a=17.502,b=240.97):
    c = (b+T_a -273.15)**2
    term = (a*b)/c - 1 / T_a
    return rho_vr*term