import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from GRAFICO.DGH import Dg_estrela, H_estrela  
from NEWTON import metodo_newton
from TAU_T import partial_rho_vr,rho_vr,contas_tau_T,e_sat_esp
from T_EQ.BETA import code_beta,D_linha_w,e_sat,k_linha_a,L_v
from T_EQ.EXP_Y import code_exp_y,m_s,Phi_s,rho_s,sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_R.partial_rt_teste import calcular_tau_r
from scipy.integrate import odeint  

if __name__ == "__main__":
    import numpy as np
   
# Constantes Iury(2018)
    T_a = 16.85           # Temperatura do ar em °C
    T_a_em_k = 290        # Temperatura do ar em K
    T_mar_em_k = 285      # Temperatura do mar em k
    T_gota = 11.85        # Temperatura da gota em °C
    T_gota_em_k = 285     # Temperatura da gota em K
    P = 1000              # Pressão em mb
    M_H2O = 18.016e-3     # Massa molecular da água em kg/mol
    r_i = 30e-6          # Raio em metros
    R = 8.31              # Constante universal dos gases
    M_NaCl = 58.443e-3    # Massa molecular de sal na água em kg/mol
    rho_w = 1025          # Densidade da água do mar kg/m^3
    v_ion = 2             # Número de íons por molécula de NaCl dissociada
    s = 34/1000           # Salinidade fracionária (34 psu)
    f = 0.8               # Umidade relativa fracionária
    C_ar =  0.0154        # Concentração do gás carbônico no ar
    S = 34                # Salinidade do mar
    R_atm = 0.082         # Constante universal dos gases
    g = 9.81              # Gravidade
    v_ar = 1.32e-5        # Viscosidade cinemática do ar
    rho_ar = 1.225        # Densidade do ar
    H_s = 6               # Altura significativa da onda 
    T0 = 273.15
    P0 = 1013.25

# Função para calcular U_f (velocidade terminal)
    def func(U_f):
        const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
        a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
        return U_f - const / a

    def dfunc(U_f):
        const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
        a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
        d_a = -0.158 * (2/3) * (2 * r_i / v_ar)**(2/3) * U_f**(-1/3)
        return 1 - (const * d_a) / (a**2)

# Aplicar método de Newton para encontrar U_f
    U_f0 = 0.1  # Chute inicial
    U_f_sol = metodo_newton.Newton(func, dfunc, U_f0, 1e-6, 50)
    tau_f = H_s / (2 * U_f_sol)  # Tempo de residência
    print(f"[DEBUG] U_f encontrado: {U_f_sol:.6e}")
    print(f"[DEBUG] tau_f: {tau_f:.6e}")

# Parâmetros para T_eq e r_eq
    e_sat = e_sat.calcular_esat(T_a) # Pressão de vapor de saturação 
    L_v = L_v.calcular_lv(T_gota) # Calor latente de vaporização da água
    D_linha_w = D_linha_w.calculate_Dw_prime(T_gota_em_k, R,P, r_i, M_H2O , T0 , P0 , alpha_c = 0.036 , Delta_w = 8e-8) # Difusividade do vapor d'água ajustada para efeitos de não-continuidade
    k_linha_a = k_linha_a.calculate_K_linha_a(T_gota,T_gota_em_k, P, r_i , R , T0 , P0, M_a = 28.9644e-3 , alpha_T = 0.7 , delta_T = 2.16e-7, c_pa = 1.006e3) # Condutividade térmica ajustada para efeitos de não-continuidade
    rho_ww = m_s.calcular_rho_ww(T_gota) # Densidade da água pura em uma gota 
    m_s = m_s.calcular_ms(rho_ww,r_i,s) # Massa de sal em uma gotícula de spray
    v_a = rho_s.calcular_v_a(T_gota, m_s, M_NaCl, r_i) # Volume molar aparente de uma solução aquosa
    m_w = rho_s.calcular_massa_agua(r_i, rho_ww) # Massa de água pura em uma gotícula de spray
    sigma_s = sigma_s.calculate_sigma_s(T_gota, m_s,m_w) # Tensão superficial de uma superfície plana de água
    Phi_s = Phi_s.calcular_phi_s(m_s, M_NaCl, m_w)
    rho_s = rho_s.calcular_rho_spray(rho_ww, m_s, m_w, v_a, M_NaCl)
    Delta_T = Delta_T_contas.calculate_delta_T(
    T_a_em_k,
    alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta= code_beta.calcular_beta(e_sat, T_a_em_k, L_v, M_H2O, D_linha_w, R, k_linha_a),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O , sigma_s , T_a_em_k , rho_w , R, v_ion , Phi_s , m_s , r_i , rho_s ,  M_NaCl),
    f=f)

    # Parâmetros para tau_T
    rho_vr = rho_vr.calcular_rho_vr(M_H2O,e_sat_esp.calcular_esat(T_a),code_exp_y.calcular_exp_y(M_H2O , sigma_s , T_a_em_k , rho_w , R, v_ion , Phi_s , m_s , r_i , rho_s ,  M_NaCl),R,T_a_em_k)
    partial_rho_vr = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k,rho_vr,a=17.502,b=240.97)
    tau_T = contas_tau_T.calculate_tau_T(rho_s , r_i , k_linha_a , D_linha_w ,partial_rho_vr ,L_v ,c_ps = 4000)

    T_eq = Delta_T[0] + T_a
    T_eq_em_k = T_eq + 273.15


    def zeta(r_i):
        term1 = (f - 1)
        term2 = (2 * M_H2O * sigma_s) / (R * T_eq_em_k * rho_w * r_i)
        denominator = (4 * np.pi * rho_s * (r_i**3 / 3)) - m_s
        term3 = (v_ion * Phi_s * m_s * (M_H2O / M_NaCl)) / denominator
        return term1 - term2 + term3

    def dzeta_dr(r_i):
        term1 = (2 * M_H2O * sigma_s) / (R * T_eq_em_k * rho_w * (r_i**2))
        denominator = (4 * np.pi * rho_s * (r_i**3)) / 3 - m_s
        term2 = (v_ion * Phi_s * m_s * (M_H2O / M_NaCl) * 4 * np.pi * rho_s * (r_i**2)) / (denominator**2)
        return term1 - term2

    r0 = (1.1) * (3 * m_s / (4 * np.pi * rho_s)) ** (1 / 3)
    r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)
    tau_r = calcular_tau_r(f, M_H2O, sigma_s, R, T_a_em_k, rho_w, r_i, rho_s, m_s, v_ion, Phi_s, M_NaCl, D_linha_w, e_sat, L_v, k_linha_a, r_eq)[1]

    while True:
        print("\nOpções de simulação:")
        print("1 - Simulação com T(t) e r(t) variando no tempo")
        print("2 - Simulação com somente r(t) variando no tempo")
        print("3 - Simulação com somente T(t) variando no tempo")
        print("4 - Simulação com r(t) e T(t) constantes no tempo")
        print("f - Sair")
        choice = input("Escolha a simulação desejada (1-4) ou 'f' para sair: ").strip()
        
        if choice.lower() == 'f':
            print("Encerrando o programa...")
            break
        
        if choice not in ['1', '2', '3', '4']:
            print("Escolha inválida! Tente novamente.")
            continue
        
        # Parâmetros da simulação
        dt = 1e-5
        n_steps = int(tau_f / dt)
        tempo = np.linspace(0, tau_f, n_steps)
        massa = np.zeros(n_steps)
        raio = np.zeros(n_steps)
        temperatura = np.zeros(n_steps)
        
        # Condições iniciais
        massa[0] = 1.2e-15
        raio[0] = r_i if choice in ['1', '2'] else r_i  # Fixo para escolha 3 e 4
        temperatura[0] = T_gota_em_k if choice in ['1', '3'] else T_gota_em_k  # Fixo para escolha 2 e 4
        
        # Opção 4: Solução analítica da EDO (não usa Euler)
        if choice == '4':
            # Calcular H_estrela e Dg_estrela com valores fixos
            H_estrela_fixo = H_estrela.calcular_H_estrela(T_mar_em_k, S)
            Dg_estrela_fixo = Dg_estrela.calcular_Dg_estrela(r_i, T_a_em_k, R_atm)
            
            # Definir a EDO para a massa
            def edo_massa(m, t):
                termo = (3 * m) / (4 * np.pi * (r_i**3) * H_estrela_fixo * R_atm * T_gota_em_k)
                dm_dt = 4 * np.pi * r_i * Dg_estrela_fixo * (C_ar - termo)
                return dm_dt
            
            # Resolver a EDO
            massa_ode = odeint(edo_massa, massa[0], tempo)
            
            # Plotar o resultado
            plt.figure(figsize=(8, 5))
            plt.plot(tempo, massa_ode, 'b-', label="Massa da gotícula")
            plt.xlabel('t (s)')
            plt.ylabel('m(t) (mol)')
            plt.xscale('log')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Perguntar se deseja repetir
            repetir = input("\nDeseja realizar outra simulação? (s/n): ").strip().lower()
            if repetir != 's':
                print("Encerrando o programa...")
                break
            else:
                continue  # Reinicia o loop
        resultados = {}
        # Método de Euler Backward para opções 1, 2 e 3
        for n in range(n_steps - 1):
            t_current = tempo[n + 1]
            
            # Atualizar raio
            if choice in ['1', '2']:
                raio_next = r_eq + (r_i - r_eq) * np.exp(-t_current / tau_r)
            else:
                raio_next = r_i
            raio[n + 1] = raio_next
            
            # Atualizar temperatura
            if choice in ['1', '3']:
                T_next = T_eq_em_k + (T_gota_em_k - T_eq_em_k) * np.exp(-t_current / tau_T)
            else:
                T_next = T_gota_em_k
            temperatura[n + 1] = T_next
            
            # Recalcular H_estrela e Dg_estrela com valores atuais
            H_atual = H_estrela.calcular_H_estrela(T_mar_em_k, S) 
            Dg_atual = Dg_estrela.calcular_Dg_estrela(r_i, T_a_em_k, R_atm)
            
            # Cálculo da massa
            raio_used = raio_next if choice != ['3','4'] else r_i
            T_used = T_next if choice in ['1', '3'] else T_gota_em_k
            
            numerador = massa[n] + dt * 4 * np.pi * raio_used * Dg_atual * C_ar 
            denominador = 1 + dt * (3 * Dg_atual) / (raio_used**2 * H_atual * R_atm * T_used) 
            massa[n + 1] = numerador / denominador
        resultados['RT'] = massa
        
        # Determinar número de subplots
        if choice == '1':
            subplots = 3
        elif choice in ['2', '3']:
            subplots = 2
        else:
            subplots = 1
        
        # Plotagem
        plt.figure(figsize=(8, 10))
        
        # Subplot da massa
        plt.subplot(subplots, 1, 1)
        plt.plot(tempo, massa, 'b-', label=f'E-RT - Massa final: {resultados["RT"][-1]:.3e} mol')
        plt.xlabel('t (s)')
        plt.ylabel('m(t) (mol)')
        plt.xscale('log')
        plt.legend()
        
        # Subplot do raio (se aplicável)
        if choice in ['1', '2']:
            plt.subplot(subplots, 1, 2)
            plt.plot(tempo, raio * 1e6, 'b-', label="Raio da gotícula")
            plt.xlabel('t (s)')
            plt.ylabel('r(t) (µm)')
            plt.xscale('log')
            ax = plt.gca()
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.ticklabel_format(style='plain', axis='y')
            plt.legend()
        
        # Subplot da temperatura (se aplicável)
        if choice in ['1', '3']:
            pos = 3 if choice == '1' else 2
            plt.subplot(subplots, 1, pos)
            plt.plot(tempo, temperatura, 'b-', label="Temperatura")           
            plt.xlabel('t (s)')
            plt.ylabel('T(t) (K)')
            plt.xscale('log')  

            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Repetir simulação (para opções 1,2,3)
        repetir = input("\nDeseja realizar outra simulação? (s/n): ").strip().lower()
        if repetir != 's':
            print("Encerrando o programa...")
            break
    print(tau_r)
    print(tau_T)
    print(r_eq)
    print(T_eq)


