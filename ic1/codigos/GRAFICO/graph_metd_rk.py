import numpy as np
import matplotlib.pyplot as plt
from DGH import Dg_estrela, H_estrela  # Certifique-se de que estas classes estão definidas

# Constantes Iury (2018)
r_i = 30e-6                
r_eq = 11.508935981764845e-6             
tau_r = 21.588038138415623               
T_i = 285               
T_eq_em_k = 15.921331148869596 + 273.15  
tau_T = 0.017809805752997256             
T_mar_em_k = 285     
T_a_em_k = 291.15

C_ar = 0.0154               
S = 34                      
R = 0.082                     

# Coeficientes do método de Cash-Karp (RK5)
c2 = 1/5
c3 = 3/10
c4 = 3/5
c5 = 1
c6 = 7/8

a21 = 1/5
a31 = 3/40
a32 = 9/40
a41 = 3/10
a42 = -9/10
a43 = 6/5
a51 = -11/54
a52 = 5/2
a53 = -70/27
a54 = 35/27
a61 = 1631/55296
a62 = 175/512
a63 = 575/13824
a64 = 44275/110592
a65 = 253/4096

b1 = 37/378
b2 = 0
b3 = 250/621
b4 = 125/594
b5 = 0
b6 = 512/1771

while True:
    print("\nOpções de simulação:")
    print("1 - Simulação com T(t) e r(t) variando no tempo")
    print("2 - Simulação com somente r(t) variando no tempo")
    print("3 - Simulação com somente T(t) variando no tempo")
    choice = input("Escolha a simulação desejada (1 - 2 - 3) ou 'f' para sair: ").strip()
    
    if choice.lower() == 'f':
        print("Encerrando o programa...")
        break
    
    if choice not in ['1', '2', '3']:
        print("Escolha inválida! Tente novamente.")
        continue
    
    # Configurações iniciais
    subplots = 3 if choice == '1' else 2
    r_fixo = r_i  # Fixo para escolha 3

    # Parâmetros da simulação
    t_total = 1e2
    dt = 1e-4
    n_steps = int(t_total // dt) + 1
    tempo = np.linspace(0, t_total, n_steps)
    massa = np.zeros(n_steps)
    raio = np.zeros(n_steps)
    temperatura = np.zeros(n_steps)
    
    # Condições iniciais
    massa[0] = 1.2e-15
    raio[0] = r_i if choice in ['1', '2'] else r_fixo
    temperatura[0] = T_i if choice in ['1', '3'] else T_i
    
    # Método de Runge-Kutta Ordem 5 (Cash-Karp)
    for n in range(n_steps - 1):
        t_n = tempo[n]
        m_n = massa[n]
        h = dt
        
        # Função para calcular dm/dt
        def f(t, m):
            # Calcular raio
            if choice in ['1', '2']:
                r = r_eq + (r_i - r_eq) * np.exp(-t / tau_r)
            else:
                r = r_fixo
            
            # Calcular temperatura
            if choice in ['1', '3']:
                T = T_eq_em_k + (T_i - T_eq_em_k) * np.exp(-t / tau_T)
            else:
                T = T_i
            
            # Calcular H e Dg
            H = H_estrela.calcular_H_estrela(T_mar_em_k, S)
            Dg = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R)
            
            # Equação diferencial
            termo1 = 4 * np.pi * r * Dg * C_ar
            termo2 = (3 * Dg * m) / (r**2 * H * R * T)
            return termo1 - termo2
        
        # Cálculo dos estágios RK5
        k1 = f(t_n, m_n)
        k2 = f(t_n + c2*h, m_n + a21*k1*h)
        k3 = f(t_n + c3*h, m_n + (a31*k1 + a32*k2)*h)
        k4 = f(t_n + c4*h, m_n + (a41*k1 + a42*k2 + a43*k3)*h)
        k5 = f(t_n + c5*h, m_n + (a51*k1 + a52*k2 + a53*k3 + a54*k4)*h)
        k6 = f(t_n + c6*h, m_n + (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)*h)
        
        # Atualizar massa
        massa[n+1] = m_n + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        
        # Atualizar raio e temperatura
        t_current = t_n + h
        # Raio
        if choice in ['1', '2']:
            raio_next = r_eq + (r_i - r_eq) * np.exp(-t_current / tau_r)
        else:
            raio_next = r_fixo
        raio[n + 1] = raio_next
        # Temperatura
        if choice in ['1', '3']:
            T_next = T_eq_em_k + (T_i - T_eq_em_k) * np.exp(-t_current / tau_T)
        else:
            T_next = T_i
        temperatura[n + 1] = T_next
    
    # Plotagem (ajustada para temperatura)
    plt.figure(figsize=(8, 10), dpi=150)
    
    # Subplot da massa
    plt.subplot(subplots, 1, 1)
    plt.plot(tempo, massa, 'b-', label="Massa da gotícula")
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
    
    # Repetir simulação
    repetir = input("\nDeseja realizar outra simulação? (s/n): ").strip().lower()
    if repetir != 's':
        print("Encerrando o programa...")
        break