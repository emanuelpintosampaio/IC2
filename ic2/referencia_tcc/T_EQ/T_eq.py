from Delta_T_contas import calculate_delta_T
from alpha import calcular_alpha
from BETA import code_beta,D_linha_w,e_sat,k_linha_a,L_v
from EXP_Y import code_exp_y,m_s,Phi_s,rho_s,sigma_s

# Constantes
T_a = 18              # Temperatura do ar em °C
T_a_em_k = 291.15     # Temperatura do ar em K
T_gota = 20           # Temperatura da gota em °C
T_gota_em_k = 293.15  # Temperatura da gota em K
P = 1000              # Pressão em mb
M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
r_i = 100e-6          # Raio 
R = 8.31              # kg·m^2/(s^2·mol·K)
M_NaCl = 58.443e-3    # Massa molecular de sal na água em kg/mol
rho_w = 1025          # Densidade da água pura em kg/m^3
v_ion = 2             # Número de íons por molécula de NaCl dissociada
s = 34/1000           # Salinidade fracionária (34 psu)
f=0.9                 # Umidade relativa fracionária
T0 = 273.15
P0 = 1013.15

# Parâmetros
e_sat = e_sat.calcular_esat(T_a)
L_v = L_v.calcular_lv(T_gota)
D_linha_w = D_linha_w.calculate_Dw_prime(T_gota_em_k, R,P, r_i, M_H2O , T0 , P0 , alpha_c = 0.036 , Delta_w = 8e-8)
k_linha_a = k_linha_a.calculate_K_linha_a(T_gota_em_k, P, r_i , R , T0 , P0, M_a = 28.9644e-3 , alpha_T = 0.7 , Delta_T = 2.16e-7, c_pa = 1.006e3)
rho_ww = m_s.calcular_rho_ww(T_gota)
m_s = m_s.calcular_ms(rho_w,r_i,s)
v_a = rho_s.calcular_v_a(T_gota, m_s, M_NaCl, r_i)
m_w = rho_s.calcular_massa_agua(r_i, rho_ww)
sigma_s = sigma_s.calculate_sigma_s(T_gota, m_s,m_w)
Phi_s = Phi_s.calcular_phi_s(m_s, M_NaCl, m_w)
rho_s = rho_s.calcular_rho_spray(rho_ww, m_s, m_w, v_a, M_NaCl)

Delta_T = calculate_delta_T(
    T_a_em_k,
    alpha=calcular_alpha(T_a_em_k, a=17.502, b=514.12),
    beta= code_beta.calcular_beta(e_sat, T_a_em_k, L_v, M_H2O, D_linha_w, R, k_linha_a),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O , sigma_s , T_a_em_k , rho_w , R, v_ion , Phi_s , m_s , r_i , rho_s ,  M_NaCl),
    f=f)

# Resultado
if Delta_T:
    tau_T1 = Delta_T[0] + T_a
    tau_T2 = Delta_T[1] + T_a
    print(f"ΔT1: {Delta_T[0]:.4f} °C, ΔT2: {Delta_T[1]:.4f} °C")
    print(f"T_eq: {tau_T1:.4f} °C, T_eq: {tau_T2:.4f} °C")
else:
    print("Não há soluções reais para os parâmetros fornecidos.")