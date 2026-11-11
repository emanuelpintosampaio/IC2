from NEWTON import metodo_newton_graph  
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s as expy_m_s, Phi_s as expy_Phi_s, rho_s as expy_rho_s, sigma_s as expy_sigma_s
from T_EQ import Delta_T_contas, alpha
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


if __name__ == "__main__":
    T_a = 16.85           # Temperatura do ar em °C
    T_a_em_k = 290        # Temperatura do ar em K
    T_mar_em_k = 285      # Temperatura do mar em k
    T_gota = 11.85        # Temperatura da gota em °C
    T_gota_em_k = 285     # Temperatura da gota em K
    P = 1000              # Pressão em mb
    M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
    r_i = 30e-6           # Raio em metros
    R = 8.31              # Constante universal dos gases
    M_NaCl = 58.443e-3    # Massa molecular de sal na água em kg/mol
    rho_w = 1025          # Densidade da água do mar kg/m^3
    v_ion = 2             # Número de íons por molécula de NaCl dissociada
    s = 34/1000           # Salinidade fracionária (34 psu)
    f = 0.8               # Umidade relativa fracionária
    S = 34                # Salinidade do mar
    T0 = 273.15
    P0 = 1013.25
    """
    T_a = 18           # Temperatura do ar em °C
    T_a_em_k = 291.15        # Temperatura do ar em K
    T_gota = 20        # Temperatura da gota em °C
    T_gota_em_k = 293.15     # Temperatura da gota em K
    P = 1000              # Pressão em mb
    M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
    r_i = 100e-6           # Raio em metros
    R = 8.31              # Constante universal dos gases
    M_NaCl = 58.443e-3    # Massa molecular de sal na água em kg/mol
    rho_w = 1025          # Densidade da água do mar kg/m^3
    v_ion = 2             # Número de íons por molécula de NaCl dissociada
    s = 34/1000           # Salinidade fracionária (34 psu)
    f = 0.9               # Umidade relativa fracionária
    S = 34                # Salinidade do mar
    T0 = 273.15
    P0 = 1013.25
    """
    # Cálculos preliminares
    e_sat_value = e_sat.calcular_esat(T_a)  # Pressão de vapor de saturação
    L_v_value = L_v.calcular_lv(T_gota)     # Calor latente de vaporização
    D_linha_w_value = D_linha_w.calculate_Dw_prime(
        T_gota_em_k, R, P, r_i, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8
    )
    k_linha_a_value = k_linha_a.calculate_K_linha_a(
        T_gota, T_gota_em_k, P, r_i, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3
    )
    
    # Cálculos para solução
    rho_ww = expy_m_s.calcular_rho_ww(T_gota)
    m_s_value = expy_m_s.calcular_ms(rho_ww, r_i, s)
    v_a_value = expy_rho_s.calcular_v_a(T_gota, m_s_value, M_NaCl, r_i)
    m_w_value = expy_rho_s.calcular_massa_agua(r_i, rho_ww)
    sigma_s_value = expy_sigma_s.calculate_sigma_s(T_gota, m_s_value, m_w_value)
    Phi_s_value = expy_Phi_s.calcular_phi_s(m_s_value, M_NaCl, m_w_value)
    rho_s_value = expy_rho_s.calcular_rho_spray(rho_ww, m_s_value, m_w_value, v_a_value, M_NaCl)

    # Cálculo de Delta_T
    alpha_value = alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97)
    beta_value = code_beta.calcular_beta(e_sat_value, T_a_em_k, L_v_value, M_H2O, D_linha_w_value, R, k_linha_a_value)
    exp_y_value = code_exp_y.calcular_exp_y(
        M_H2O, sigma_s_value, T_a_em_k, rho_w, R, v_ion, Phi_s_value, m_s_value, r_i, rho_s_value, M_NaCl
    )
    Delta_T = Delta_T_contas.calculate_delta_T(
        T_a_em_k, alpha=alpha_value, beta=beta_value, b=240.97, exp_y=exp_y_value, f=f
    )

    if not Delta_T:
        raise ValueError("Não há soluções reais para os parâmetros fornecidos para calcular T_eq.")
    
    T_eq = Delta_T[0] + T_a
    T_eq_em_k = T_eq + 273.15

    rho_ww2 = expy_m_s.calcular_rho_ww(T_eq)
    m_s_value2 = expy_m_s.calcular_ms(rho_ww2, r_i, s)
    v_a_value2 = expy_rho_s.calcular_v_a(T_eq, m_s_value2, M_NaCl, r_i)
    m_w_value2 = expy_rho_s.calcular_massa_agua(r_i, rho_ww2)
    sigma_s_value2 = expy_sigma_s.calculate_sigma_s(T_eq, m_s_value2, m_w_value2)
    Phi_s_value2 = expy_Phi_s.calcular_phi_s(m_s_value2, M_NaCl, m_w_value2)
    rho_s_value2 = expy_rho_s.calcular_rho_spray(rho_ww2, m_s_value2, m_w_value2, v_a_value2, M_NaCl)        

    
    # Função zeta(r_i)
    def zeta(r_i):
        term1 = (f - 1)
        term2 = (2 * M_H2O * sigma_s_value2) / (R * T_eq_em_k * rho_w * r_i)
        term3 = (v_ion * Phi_s_value2 * m_s_value2 * (M_H2O / M_NaCl)) / ((4 * np.pi * rho_s_value2 * (r_i**3 / 3)) - m_s_value2)
        return term1 - term2 + term3

    # Derivada de zeta em relação a r_i
    def dzeta_dr(r_i):
        term1 = (2 * M_H2O * sigma_s_value2) / (R * T_eq_em_k * rho_w * (r_i**2))
        denominator = (4 * np.pi * rho_s_value2 * (r_i**3)) / 3 - m_s_value2
        term2 = (v_ion * Phi_s_value2 * m_s_value2 * (M_H2O / M_NaCl) * 4 * np.pi * rho_s_value2 * (r_i**2)) / (denominator**2)
        return term1 - term2
    
    # Raio inicial
    r0 = (3 * m_s_value2 / (4 * np.pi * rho_s_value2)) ** (1/3)
    r0_initial = 1.1 * r0  

    # Método de Newton
    epsilon = 1e-6
    max_iter = 1000
    r0_initial_pos = 1.1 * r0
    
    # Aplicar método de Newton e capturar iterações
    r_eq, iteracoes_pos = None, []
    try:
        r_eq, iteracoes_pos = metodo_newton_graph.Newton(zeta, dzeta_dr, r0_initial_pos, epsilon, max_iter)
        print(f"Raio de equilíbrio positivo: {r_eq * 1e6:.2f} µm")
    except ValueError as e:
        print(f"Erro (positivo): {e}")

    # Plotagem
    r_i_values = np.linspace(-500e-6, 500e-6, 5000000)
    zeta_values = [zeta(r) for r in r_i_values]
    plt.figure(figsize=(10, 4)) 

    plt.plot(r_i_values * 1e6, zeta_values, label=r'$\zeta(r)$', color='blue', linewidth=1.5)
    
    # Plotar iterações do método de Newton
   # if iteracoes_pos:
   #     iteracoes_y = [zeta(r) for r in iteracoes_pos]
   #     plt.scatter(
   #         [r * 1e6 for r in iteracoes_pos],
   #         iteracoes_y,
   #         color='green', s=50, marker='o',
   #         zorder=5,label= 'pontos')
    # Linhas de referência
    plt.axvline(r0 * 1e6, color='red', linestyle='-', linewidth=1.5, 
            label=fr'Descontinuidade: $\left( \frac{{3m_s}}{{4\pi \rho_s}} \right)^{{1/3}} = {r0*1e6:.2f}\ \mu m$')
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, label=r'$\zeta(r) = 0$')
    #plt.axhline(-0.1, color='yellow', linestyle='-', linewidth=1.3,
     #       label=r'$\displaystyle \lim_{r \to \infty} \zeta(r) = -0.1$')

    #plt.scatter(r_eq * 1e6, 0, color='lime', s=300, marker='*',
    #        edgecolor='black', zorder=5,
    #        label=fr'$r_{{eq}} = {r_eq*1e6:.2f}\ \mu m$')
 
    plt.xlabel(r'$r \ (\mu m)$')
    plt.ylabel(r'$\zeta(r)$')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    print(r_eq)
    