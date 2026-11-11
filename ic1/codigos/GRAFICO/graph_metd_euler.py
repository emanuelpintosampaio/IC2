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

# Constantes Andreas (2005) (comentado)
"""
r_i = 100e-6                
r_eq = 61.28e-6             
tau_r = 306.94               
T_i = 293.15               
T_eq_em_k = 17.069 + 273.15  
tau_T = 0.176             
T_mar_em_k = 293.15     
T_a_em_k = 291.15
"""

C_ar = 0.0154               
S = 34                      
R = 0.082                     

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
    T_fixo = T_i  # Fixo para escolha 2 

    # Parâmetros da simulação
    t_total = 1e4
    dt = 1e-4
    n_steps = int(t_total // dt) + 1
    tempo = np.linspace(0, t_total, n_steps)
    massa = np.zeros(n_steps)
    raio = np.zeros(n_steps)
    temperatura = np.zeros(n_steps)
    
    # Condições iniciais
    massa[0] = 1.2e-15
    raio[0] = r_i if choice in ['1', '2'] else r_fixo
    temperatura[0] = T_i if choice in ['1', '3'] else T_fixo
    
    # Método de Euler Backward 
    for n in range(n_steps - 1):
        t_current = tempo[n + 1]
        
        # Atualizar raio
        if choice in ['1', '2']:
            raio_next = r_eq + (r_i - r_eq) * np.exp(-t_current / tau_r)
        else:
            raio_next = r_fixo
        raio[n + 1] = raio_next
        
        # Atualizar temperatura
        if choice in ['1', '3']:
            T_next = T_eq_em_k + (T_i - T_eq_em_k) * np.exp(-t_current / tau_T)
        else:
            T_next = T_fixo
        temperatura[n + 1] = T_next
        
        # Recalcular H_estrela e Dg_estrela com valores atuais
        H_atual = H_estrela.calcular_H_estrela(T_mar_em_k, S)  # Se H dependesse de T_used, ajustar aqui
        Dg_atual = Dg_estrela.calcular_Dg_estrela(r_i, T_a_em_k, R)
        
        # Cálculo da massa
        raio_used = raio_next if choice != '3' else r_fixo
        T_used = T_next if choice in ['1', '3'] else T_i
        
        numerador = massa[n] + dt * 4 * np.pi * raio_used * Dg_atual * C_ar 
        denominador = 1 + dt * (3 * Dg_atual) / (raio_used**2 * H_atual * R * T_used) 
        massa[n + 1] = numerador / denominador
    
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
        plt.xscale('log')  # Escala linear para melhor visualização
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Repetir simulação
    repetir = input("\nDeseja realizar outra simulação? (s/n): ").strip().lower()
    if repetir != 's':
        print("Encerrando o programa...")
        break