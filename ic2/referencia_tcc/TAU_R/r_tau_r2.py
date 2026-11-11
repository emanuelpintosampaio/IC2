import numpy as np
def calculate_r_tau_r2(r_i,r_eq):
    return (((1/np.e)**(1/2)) * (r_i - r_eq)) + r_eq


"""
r_i = 100e-6 # = 100 \mu m
r_eq = 6.123495092205673e-05 # = 61.23495092205673 \mu m
a = calculate_r_tau_r2(r_i,r_eq)
print(f'{a*1e6}')
"""
#--------------------------------------------
"""
EXPERIMENTO (OK)
r_i = 100e-6
r_eq = 6.123495092205673e-05
a = calculate_r_tau_r2(r_i,r_eq)
print(f'{a}')
"""