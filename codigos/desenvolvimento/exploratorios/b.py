# -*- coding: utf-8 -*-

"""
Simulação Numérica da Difusão de CO₂ em Gotículas de Spray Marinho

Este script implementa um modelo computacional completo para simular a absorção
de dióxido de carbono (CO₂) por gotículas de spray marinho. A estrutura é
orientada a objetos e modular, integrando a física detalhada da microfísica
de gotículas com um solver numérico robusto (Runge-Kutta de 3ª Ordem).

O código é projetado para executar e comparar dois cenários principais, conforme
definido nos planos de pesquisa de Iniciação Científica (IC1 e IC2):

- **Cenário A (Modelo Completo):** Resolve o sistema acoplado de equações
  diferenciais para a evolução temporal da massa de CO₂, raio e temperatura
  da gotícula. Este cenário representa a física mais realista.

- **Cenário B (Modelo Simplificado):** Calcula a absorção de CO₂ assumindo
  que o raio e a temperatura da gotícula permanecem constantes em seus valores
  iniciais. Este cenário serve como uma linha de base (controle) para quantificar
  o impacto das variações microfísicas.

Os resultados de ambas as simulações são salvos em arquivos CSV para análise
comparativa posterior.

Estrutura do Código:
1.  **Configuração (SimulationParameters):** Centraliza todas as constantes e
    parâmetros de entrada.
2.  **Entidades Físicas (Atmosphere, Droplet):** Representam as condições
    atmosféricas e o estado da gotícula.
3.  **Funções Auxiliares:** Funções para cálculos físico-químicos.
4.  **Motor de Simulação (Simulation):** Orquestra a simulação, implementa o
    solver RK3 e as equações diferenciais para cada cenário.
5.  **Bloco de Execução Principal:** Configura, executa as simulações e
    gerencia o salvamento dos dados.
"""

import numpy as np
import pandas as pd
import os

# ==============================================================================
# 1. CONFIGURAÇÃO DA SIMULAÇÃO
# ==============================================================================

class SimulationParameters:
    """
    Agrega todos os parâmetros de configuração para a simulação, incluindo
    condições ambientais, propriedades da gotícula e definições numéricas.
    A nomenclatura das variáveis segue a terminologia dos projetos de pesquisa.
    """
    def __init__(self):
        # --- Condições Ambientais e Físicas ---
        self.T_a = 16.85          # Temperatura do ar [°C]
        self.T_a_k = 290.0        # Temperatura do ar [K]
        self.P = 1000.0           # Pressão atmosférica [hPa]
        self.f = 0.8              # Umidade relativa [adimensional, 0 a 1]
        self.S = 34.0             # Salinidade do mar [psu]
        self.H_s = 6.0            # Altura significativa da onda [m]
        self.C_ar = 0.0154        # Concentração de CO₂ no ar [mol/m³]
        self.g = 9.81             # Aceleração da gravidade [m/s²]

        # --- Propriedades Iniciais da Gotícula ---
        self.r_i = 30e-6          # Raio inicial da gotícula [m]
        self.T_gota = 11.85       # Temperatura inicial da gotícula [°C]
        self.T_gota_k = 285.0     # Temperatura inicial da gotícula [K]
        self.m_co2_initial = 1.2e-15 # Massa inicial de CO₂ [mol]

        # --- Constantes Físico-Químicas ---
        self.R_gas = 8.314        # Constante universal dos gases [J/(mol·K)]
        self.M_h2o = 18.016e-3    # Massa molar da água [kg/mol]
        self.M_nacl = 58.443e-3   # Massa molar do NaCl [kg/mol]
        self.rho_w = 1025.0       # Densidade da água do mar [kg/m³]
        self.rho_ar = 1.225       # Densidade do ar [kg/m³]
        self.v_ar = 1.32e-5       # Viscosidade cinemática do ar [m²/s]
        self.v_ion = 2            # Número de íons por molécula de NaCl
        self.c_ps = 4000.0        # Calor específico da água do mar [J kg⁻¹ K⁻¹]
        self.M_a = 28.9644e-3     # Massa molar do ar [kg/mol]
        self.c_pa = 1.006e3        # Calor específico do ar a pressão constante [J kg⁻¹ K⁻¹]

        # --- Parâmetros de Simulação Numérica ---
        self.dt = 1e-4            # Passo de tempo [s]

# ==============================================================================
# 2. FUNÇÕES AUXILIARES E CÁLCULOS FÍSICOS
# ==============================================================================

def newton_method(f, df, x0, tol=1e-6, max_iter=50):
    """Implementação simples do método de Newton para encontrar a raiz de f(x)."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0:
            break
        x = x - fx / dfx
    return x

def calculate_H_star(T_k, S):
    """
    Calcula a constante da Lei de Henry adimensional (H* = c_aq / c_gas).
    Ref: Weiss (1974), adaptado por Gonçalves & Innocentini (2018).
    """
    # Coeficientes para a equação de Weiss (1974)
    a1, a2, a3 = -58.0931, 90.5069, 22.2940
    b1, b2, b3 = 0.027766, -0.025888, 0.0050578
    
    term1 = a1 + a2 * (100 / T_k) + a3 * np.log(T_k / 100)
    term2 = S * (b1 + b2 * (T_k / 100) + b3 * (T_k / 100)**2)
    
    return np.exp(term1 + term2)

def calculate_Dg_star(r, T_a_k):
    """
    Calcula o coeficiente de difusão de CO₂ modificado pela curvatura (D*_g).
    Ref: Pruppacher & Klett (2010), Eq. 17-126.
    """
    D_g = 0.16e-4       # Coeficiente de difusão de CO₂ no ar [m²/s]
    alpha_eps = 1.4e-5  # Coeficiente de acomodação para CO₂
    M_co2 = 0.044       # Massa molar do CO₂ [kg/mol]
    R_gas = 8.314       # Constante dos gases [J/(mol·K)]
    
    v_eps = np.sqrt(8 * R_gas * T_a_k / (np.pi * M_co2))
    
    denominator = 1 + (4 * D_g) / (r * alpha_eps * v_eps)
    return D_g / denominator

# ==============================================================================
# 3. ENTIDADES FÍSICAS
# ==============================================================================

class Atmosphere:
    """Armazena as propriedades do ambiente atmosférico."""
    def __init__(self, params: SimulationParameters):
        self.T_a_k = params.T_a_k
        self.pressure = params.P
        self.relative_humidity = params.f
        self.C_ar = params.C_ar
        self.salinity = params.S

class Droplet:
    """Representa o estado de uma única gotícula que evolui com o tempo."""
    def __init__(self, params: SimulationParameters):
        self.radius = params.r_i
        self.temperature_k = params.T_gota_k
        self.mass_co2 = params.m_co2_initial

    def get_state_vector(self) -> np.array:
        """Retorna o estado atual como um vetor [raio, temperatura_k, massa_co2]."""
        return np.array([self.radius, self.temperature_k, self.mass_co2])

    def set_state_from_vector(self, state_vector: np.array):
        """Atualiza o estado da gotícula a partir de um vetor de estado."""
        self.radius, self.temperature_k, self.mass_co2 = state_vector

    def __repr__(self):
        return (f"Droplet(r={self.radius*1e6:.2f} µm, T={self.temperature_k:.2f} K, "
                f"m_CO2={self.mass_co2:.2e} mol)")

# ==============================================================================
# 4. MOTOR DE SIMULAÇÃO
# ==============================================================================

class Simulation:
    """
    Orquestra a simulação completa, gerenciando a evolução temporal do sistema,
    aplicando o solver numérico e calculando as derivadas físicas.
    """
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.atmosphere = Atmosphere(params)
        self.droplet = Droplet(params)
        self.tau_f = self._precompute_residence_time()

        # Pré-cálculo da massa de sal (constante para a gotícula)
        s_frac = self.params.S / 1000.0
        if s_frac >= 1.0:
            raise ValueError("A salinidade deve ser menor que 1000 psu.")
        # Ref: IC I Emanuel.md, Eq. (18)
        self.m_s = ((4.0/3.0) * np.pi * self.params.r_i**3 * self.params.rho_w *
                    (s_frac / (1.0 - s_frac)))

    def _precompute_residence_time(self) -> float:
        """Calcula o tempo de residência da gotícula na atmosfera (tau_f)."""
        p = self.params
        
        def func_Uf(U_f):
            # Equação transcendental para a velocidade terminal U_f
            re_term = (2 * p.r_i * U_f / p.v_ar)**(2/3)
            return U_f - ((2 * p.r_i**2 * p.g) / (9 * p.v_ar) * (p.rho_w / p.rho_ar - 1)) / (1 + 0.158 * re_term)

        def dfunc_Uf(U_f):
            h = 1e-6
            return (func_Uf(U_f + h) - func_Uf(U_f - h)) / (2 * h)

        U_f_sol = newton_method(func_Uf, dfunc_Uf, 0.1)
        tau_f = p.H_s / (2 * U_f_sol) if U_f_sol > 0 else float('inf')
        print(f"Tempo de Residência da Gotícula (τ_f) pré-calculado: {tau_f:.4f} s")
        return tau_f

    def _calculate_derivatives_full(self, state_vector: np.array) -> np.array:
        """
        Calcula as derivadas do sistema de EDOs (dr/dt, dT/dt, dm/dt) com base
        na física completa do modelo (Cenário A).
        """
        # 1. Desempacotar variáveis
        r, T_k, m_co2 = state_vector
        p = self.params
        T_c, T_a_c = T_k - 273.15, p.T_a_k - 273.15

        if r <= 1e-9: return np.array([0.0, 0.0, 0.0])

        # 2. Propriedades Físicas Intermediárias
        m_w = (4.0/3.0) * np.pi * r**3 * p.rho_w - self.m_s
        if m_w <= 0: return np.array([0.0, 0.0, 0.0])

        # Termo de Köhler (atividade da água, a_w)
        molality = np.clip((self.m_s / p.M_nacl) / m_w, 0, 6.0)
        phi_s = (0.9270 - 2.164e-2 * molality + 3.486e-2 * molality**2 - 5.956e-3 * molality**3 + 3.911e-4 * molality**4)
        sigma_w = 7.610e-2 - 1.55e-4 * T_c
        sigma_s = sigma_w + 2.77e-5 * (self.m_s / m_w) if m_w > 0 else sigma_w
        term_A = (2.0 * p.M_h2o * sigma_s) / (p.R_gas * T_k * p.rho_w * r)
        term_B = (p.v_ion * phi_s * self.m_s * (p.M_h2o / p.M_nacl)) / m_w
        a_w = np.exp(term_A - term_B)

        # Propriedades de transporte
        L_v = (25.00 - 0.02274 * T_c) * 1e5
        esat_T = 611.0 * np.exp((17.27 * T_c) / (T_c + 237.15))
        ka = 2.411e-2 * (1 + 3.309e-3 * T_c - 1.441e-6 * T_c**2)
        rho_a_T = (p.P * 100) / (287.05 * T_k)
        k_prime_a = ka / (1 + (ka / (r * 0.7 * rho_a_T * p.c_pa)) * np.sqrt(2 * np.pi * p.M_a / (p.R_gas * T_k)))
        Dw = 2.11e-5 * (T_k / 273.15)**1.94 * (1013.25 / p.P)
        D_prime_w = Dw / (1 + (Dw / (r * 0.036)) * np.sqrt(2 * np.pi * p.M_h2o / (p.R_gas * T_k)))

        # 3. Calcular as Derivadas
        # a) dm_dt: Variação da massa de CO₂ (mol/s)
        C_gota = m_co2 / ((4.0/3.0) * np.pi * r**3)
        H_star = calculate_H_star(T_k, p.S)
        D_g_star = calculate_Dg_star(r, p.T_a_k)
        dm_dt = 4.0 * np.pi * r * D_g_star * (p.C_ar - C_gota / H_star)

        # b) dT_dt: Variação da temperatura (K/s)
        esat_Ta = 611.0 * np.exp((17.27 * T_a_c) / (T_a_c + 237.15))
        rho_v_sat_air = (esat_Ta * p.M_h2o) / (p.R_gas * p.T_a_k)
        rho_v_air = p.f * rho_v_sat_air
        rho_v_surf = a_w * (esat_T * p.M_h2o) / (p.R_gas * T_k)
        sensible_heat = k_prime_a * (p.T_a_k - T_k)
        latent_heat = L_v * D_prime_w * (rho_v_air - rho_v_surf)
        dT_dt = (3.0 / (p.rho_w * p.c_ps * r**2)) * (sensible_heat + latent_heat)

        # c) dr_dt: Variação do raio (m/s)
        eta1 = (p.rho_w * p.R_gas * T_k) / (D_prime_w * p.M_h2o * esat_T)
        eta2 = (p.rho_w * L_v / (k_prime_a * T_k)) * ((L_v * p.M_h2o) / (p.R_gas * T_k) - 1.0)
        eta = eta1 + eta2
        dr_dt = (p.f - a_w) / (r * eta)

        return np.array([dr_dt, dT_dt, dm_dt])

    def _calculate_derivatives_simplified(self, state_vector: np.array) -> np.array:
        """
        Calcula as derivadas para o Cenário B (simplificado), onde apenas
        a massa de CO₂ varia. Raio e temperatura são constantes.
        """
        _, _, m_co2 = state_vector
        p = self.params
        
        # Raio e Temperatura são mantidos constantes nos seus valores iniciais
        r = p.r_i
        T_k = p.T_gota_k

        # Calcula a derivada apenas para a massa de CO₂
        C_gota = m_co2 / ((4.0/3.0) * np.pi * r**3)
        H_star = calculate_H_star(T_k, p.S)
        D_g_star = calculate_Dg_star(r, p.T_a_k)
        dm_dt = 4.0 * np.pi * r * D_g_star * (p.C_ar - C_gota / H_star)

        # Derivadas do raio e temperatura são zero por definição neste cenário
        dr_dt = 0.0
        dT_dt = 0.0

        return np.array([dr_dt, dT_dt, dm_dt])

    def solve_step_rk3(self, y: np.array, dt: float, F: callable) -> np.array:
        """
        Resolve um passo de tempo do sistema de EDOs usando o método de
        Runge-Kutta de 3ª ordem (Wicker e Skamarock, 2002).
        """
        # Estágio 1
        y1 = y + dt * F(y)
        # Estágio 2
        y2 = 0.75 * y + 0.25 * (y1 + dt * F(y1))
        # Estágio 3 e estado final
        y_next = (1.0/3.0) * y + (2.0/3.0) * (y2 + dt * F(y2))
        return y_next

    def run(self, scenario: str) -> dict:
        """
        Executa uma simulação completa para um cenário específico ('A' ou 'B').

        Args:
            scenario (str): 'A' para o modelo completo, 'B' para o simplificado.

        Returns:
            dict: Dicionário com os resultados da simulação (séries temporais).
        """
        if scenario == 'A':
            derivative_func = self._calculate_derivatives_full
            print("\nIniciando Simulação: Cenário A (Modelo Completo)...")
        elif scenario == 'B':
            derivative_func = self._calculate_derivatives_simplified
            print("\nIniciando Simulação: Cenário B (Modelo Simplificado)...")
        else:
            raise ValueError("Cenário deve ser 'A' ou 'B'.")

        # Reinicia a gotícula para o estado inicial
        self.droplet = Droplet(self.params)
        current_state = self.droplet.get_state_vector()

        # Configura a simulação
        n_steps = int(self.tau_f / self.params.dt)
        time_array = np.linspace(0, self.tau_f, n_steps + 1)
        
        # Inicializa o armazenamento de resultados
        results = {'time_s': [0], 'radius_m': [current_state[0]],
                   'temperature_K': [current_state[1]], 'mass_co2_mol': [current_state[2]]}

        # Loop principal da simulação
        for i in range(n_steps):
            # Resolve um passo de tempo com o solver RK3
            current_state = self.solve_step_rk3(current_state, self.params.dt, derivative_func)
            
            # Limita valores fisicamente para evitar instabilidades numéricas
            current_state[0] = max(current_state[0], 1e-9) # Raio mínimo
            
            # Armazena os resultados
            results['time_s'].append(time_array[i+1])
            results['radius_m'].append(current_state[0])
            results['temperature_K'].append(current_state[1])
            results['mass_co2_mol'].append(current_state[2])
        
        print(f"Simulação concluída. Massa final de CO₂: {current_state[2]:.4e} mol")
        return results

# ==============================================================================
# 5. BLOCO DE EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    """
    Ponto de entrada do script. Configura e executa as simulações para os
    cenários definidos, salvando os resultados em arquivos CSV.
    """
    print("="*60)
    print("INICIANDO SIMULAÇÃO DE DIFUSÃO DE CO₂ EM GOTÍCULAS MARINHAS")
    print("="*60)

    # 1. Carregar os parâmetros da simulação
    sim_params = SimulationParameters()
    
    # 2. Inicializar o motor de simulação
    simulation_engine = Simulation(sim_params)

    # 3. Executar Cenário A e salvar resultados
    results_A = simulation_engine.run(scenario='A')
    df_A = pd.DataFrame(results_A)
    output_filename_A = "resultados_cenario_A.csv"
    df_A.to_csv(output_filename_A, index=False, float_format='%.6e')
    print(f"Resultados do Cenário A salvos em: {os.path.abspath(output_filename_A)}")

    # 4. Executar Cenário B e salvar resultados
    results_B = simulation_engine.run(scenario='B')
    df_B = pd.DataFrame(results_B)
    output_filename_B = "resultados_cenario_B.csv"
    df_B.to_csv(output_filename_B, index=False, float_format='%.6e')
    print(f"Resultados do Cenário B salvos em: {os.path.abspath(output_filename_B)}")

    print("\nTodas as simulações foram concluídas com sucesso.")
    print("="*60)

if __name__ == "__main__":
    main()
