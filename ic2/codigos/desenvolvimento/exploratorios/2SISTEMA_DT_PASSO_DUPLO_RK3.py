import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTES FÍSICAS E CONDIÇÕES AMBIENTAIS
# =============================================================================
T_a = 18               # Temperatura do ar em °C
T_a_em_k = T_a + 273.15
T_mar_em_k = 293.15
T_gota = 19.8
T_gota_em_k = T_gota + 273.15
P = 1000                # Pressão em mb
M_H2O = 18.016e-3       # kg/mol
r_i = 100e-6            # raio inicial (m)
R = 8.31                # constante dos gases
M_NaCl = 58.443e-3      # kg/mol
rho_w = 1025            # densidade da água do mar (kg/m³)
v_ion = 2
s = 34/1000             # salinidade fracionária
f = 0.9                 # umidade relativa
C_ar = 0.0154           # concentração de CO₂ no ar (mol/m³ ?)
S = 34                  # salinidade (psu)
R_atm = 0.082           # constante dos gases (L atm / mol K) – cuidado com unidades!
g = 9.81
v_ar = 1.32e-5
rho_ar = 1.225
H_s = 6                 # altura significativa da onda (m)
T0 = 273.15
P0 = 1013.25
c_ps = 4000             # calor específico do spray (J/kg K)

# =============================================================================
# VELOCIDADE DE QUEDA E TEMPO DE VIDA DA GOTA (Andreas)
# =============================================================================
def func(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    return U_f - const / a

def dfunc(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2/3))
    d_a = -0.158 * (2/3) * (2 * r_i / v_ar)**(2/3) * U_f**(-1/3)
    return 1 - (const * d_a) / (a**2)

U_f0 = 0.1
U_f_sol = metodo_newton.Newton(func, dfunc, U_f0, 1e-6, 50)
tau_f = H_s / (2 * U_f_sol)          # tempo de vida da gota

# =============================================================================
# PROPRIEDADES TERMODINÂMICAS INICIAIS (muitas são constantes na simulação)
# =============================================================================
e_sat_val = e_sat.calcular_esat(T_a)
L_v_val = L_v.calcular_lv(T_gota)
D_linha_w_val = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O,
                                             T0, P0, alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R,
                                              T0, P0, M_a=28.9644e-3,
                                              alpha_T=0.7, delta_T=2.16e-7,
                                              c_pa=1.006e3)

rho_ww = m_s.calcular_rho_ww(T_gota)
m_s_val = m_s.calcular_ms(rho_ww, r_i, s)          # massa de sal (kg)
v_a = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w = rho_s.calcular_massa_agua(r_i, rho_ww)       # massa de água inicial (kg)
sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
Phi_s_val = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
rho_s_val = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

Delta_T = Delta_T_contas.calculate_delta_T(
    T_a_em_k,
    alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O,
                                 D_linha_w_val, R, k_linha_a_val),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R,
                                    v_ion, Phi_s_val, m_s_val, r_i,
                                    rho_s_val, M_NaCl),
    f=f
)

rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k)

rho_vr_val = rho_vr.calcular_rho_vr(
    M_H2O,
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R,
                              v_ion, Phi_s_val, m_s_val, r_i,
                              rho_s_val, M_NaCl),
    R, T_a_em_k
)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k,
                                                            rho_vr_val,
                                                            a=17.502, b=240.97)

tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val,
                                      D_linha_w_val, partial_rho_vr_val,
                                      L_v_val, c_ps)

T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

# Funções auxiliares para o raio de equilíbrio (não usadas diretamente na integração)
def zeta(r_i):
    term1 = (f - 1)
    term2 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r_i)
    denominator = (4 * np.pi * rho_s_val * (r_i**3 / 3)) - m_s_val
    term3 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / denominator
    return term1 - term2 + term3

def dzeta_dr(r_i):
    term1 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * (r_i**2))
    denominator = (4 * np.pi * rho_s_val * (r_i**3)) / 3 - m_s_val
    term2 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl) *
             4 * np.pi * rho_s_val * (r_i**2)) / (denominator**2)
    return term1 - term2

r0 = (1.1) * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1 / 3)
r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)

tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i,
                        rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl,
                        D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val,
                        r_eq)[1]

# =============================================================================
# SISTEMA RÁPIDO: APENAS A TEMPERATURA
# =============================================================================
def f_fast_T(T, r):
    """
    Derivada da temperatura (sistema rápido).
    T em Kelvin, r em metros.
    """
    # Massa de água atual (depende de r e da densidade da água na temperatura T)
    rho_ww_novo = m_s.calcular_rho_ww(T - 273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)

    # Propriedades que dependem da temperatura
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0,
                                                  alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T - 273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T - 273.15, T, P, r, R,
                                                   T0, P0, M_a=28.9644e-3,
                                                   alpha_T=0.7, delta_T=2.16e-7,
                                                   c_pa=1.006e3)

    sigma_s_novo = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)

    exp_y_novo = code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k,
                                           rho_w, R, v_ion, Phi_s_novo,
                                           m_s_val, r, rho_s_val, M_NaCl)
    rho_vr_novo = rho_vr.calcular_rho_vr(M_H2O,
                                         e_sat_esp.calcular_esat(T - 273.15),
                                         exp_y_novo, R, T_a_em_k)

    dT_dt = 3 / (rho_s_val * c_ps * r**2) * (
                k_linha_a_novo * (T_a_em_k - T) +
                L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo)
            )
    return dT_dt


# Variável global para armazenar a última temperatura de equilíbrio calculada
# (melhora o chute inicial nas chamadas seguintes)
_ultimo_T_eq = [T_gota_em_k]

def relaxar_temperatura(m, r, T_inicial, tol=1e-4, dt_micro=1e-4, max_iter=5000):
    """
    Avança a temperatura até atingir o equilíbrio (|dT/dt| < tol)
    usando Euler explícito com passo fixo dt_micro.
    Retorna a temperatura de equilíbrio.
    """
    T = T_inicial
    for _ in range(max_iter):
        dT = f_fast_T(T, r)
        if abs(dT) < tol:
            break
        T = T + dt_micro * dT
    # Atualiza a variável global para o próximo chute
    _ultimo_T_eq[0] = T
    return T

# =============================================================================
# SISTEMA LENTO: MASSA DE CO₂ E RAIO (com temperatura sempre em equilíbrio)
# =============================================================================
def f_system_lento(t, y):
    """
    y = [m (mol CO₂), r (m)]
    Retorna [dm/dt, dr/dt] usando a temperatura de equilíbrio instantânea.
    """
    m, r = y

    # 1) Obtém a temperatura de equilíbrio para o estado atual
    T_eq_atual = relaxar_temperatura(m, r, _ultimo_T_eq[0])

    # 2) dm/dt (transferência de CO₂)
    H_novo = H_estrela.calcular_H_estrela(T_eq_atual, S)
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol                     # concentração de CO₂ na gota (mol/m³)
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T_eq_atual))

    # 3) dr/dt (evaporação/condensação)
    #    Recalcula propriedades que dependem da temperatura de equilíbrio
    rho_ww_novo = m_s.calcular_rho_ww(T_eq_atual - 273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T_eq_atual - 273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)

    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T_eq_atual, R, P, r, M_H2O,
                                                  T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T_eq_atual - 273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T_eq_atual - 273.15, T_eq_atual,
                                                   P, r, R, T0, P0,
                                                   M_a=28.9644e-3, alpha_T=0.7,
                                                   delta_T=2.16e-7, c_pa=1.006e3)

    # Termo Y (equilíbrio de Köhler modificado)
    Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r) -
         (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val)))

    # Denominador do fluxo
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (
                L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2

    dr_dt = ((f - 1) - Y) / (r * den)

    return np.array([dm_dt, dr_dt])

# =============================================================================
# INTEGRADOR RK3 ADAPTATIVO COM PASSO DUPLO (inalterado)
# =============================================================================
def passo_rk3(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + (dt/2) * k1)
    k3 = f(t + dt, y - dt * k1 + 2 * dt * k2)
    return y + (dt/6) * (k1 + 4*k2 + k3)

def solver_rk3_adaptativo(f_system, y0, t_final, tol):
    t = 0
    dt = 1e-4               # passo inicial
    y = y0
    t_lista = [t]
    y_lista = [y]
    p = 3                    # ordem do RK3
    eta = 0.9
    dt_min = 1e-10
    dt_max = np.inf

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        # Solução com um passo inteiro
        y_1_passo = passo_rk3(f_system, t, y, dt)

        # Solução com dois subpassos
        y_temp = passo_rk3(f_system, t, y, dt/2)
        y_2_passos = passo_rk3(f_system, t + dt/2, y_temp, dt/2)

        # Estimativa do erro
        diff = y_2_passos - y_1_passo
        erro_est = np.abs(diff) / ((2**p) - 1)
        erro_relativo = erro_est / np.abs(y_2_passos + 1e-15)   # evita divisão por zero
        erro_max = np.max(erro_relativo)

        if erro_max < tol or dt < dt_min:
            t = t + dt
            y = y_2_passos
            t_lista.append(t)
            y_lista.append(y)

            # Ajusta o próximo passo
            new_dt = eta * dt * (tol / (erro_max + 1e-15))**(1/(p+1))
            new_dt = min(new_dt, dt_max)
            new_dt = max(new_dt, dt_min)
            dt = new_dt
        else:
            dt = eta * dt * (tol / (erro_max + 1e-15))**(1/(p+1))
            dt = max(dt, dt_min)

    return np.array(t_lista), np.array(y_lista)

# =============================================================================
# EXECUÇÃO
# =============================================================================
# Condição inicial: [massa de CO₂ (mol), raio (m)]
y0 = np.array([1.2e-15, r_i])

tolerancia = 1e-8
tempo_rk3, sol_lenta = solver_rk3_adaptativo(f_system_lento, y0, 1e3, tol=tolerancia)

# Recupera a temperatura de equilíbrio ao longo do tempo
temperaturas = []
T_guess = T_gota_em_k
for m, r in sol_lenta:
    T_eq_ponto = relaxar_temperatura(m, r, T_guess)
    temperaturas.append(T_eq_ponto)
    T_guess = T_eq_ponto   # usa a temperatura do ponto anterior como chute
temperaturas = np.array(temperaturas)

# Resultados finais
massa_final = sol_lenta[-1, 0]
raio_final = sol_lenta[-1, 1]
temp_final = temperaturas[-1]

print(f"Raio final           : {raio_final:.6e} m")
print(f"Temperatura final    : {temp_final-273.15:.2f} °C")
print(f"Massa final          : {massa_final:.3e} mol")
print(f"Número de passos     : {len(tempo_rk3)}")

# =============================================================================
# GRÁFICOS (opcional, descomente se desejar)
# =============================================================================
"""
plt.rcParams['text.usetex'] = False
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.delaxes(axs[1, 1])

# Massa
axs[0, 0].plot(tempo_rk3, sol_lenta[:, 0], "#2B00FF",
               label=f'Massa final = {massa_final:.3e} mol')
axs[0, 0].set_xlabel('Tempo (s)')
axs[0, 0].set_ylabel('Massa de CO₂ (mol)')
axs[0, 0].set_xscale('log')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].set_title('Evolução da Massa')

# Raio
axs[0, 1].plot(tempo_rk3, sol_lenta[:, 1] * 1e6, "#FF0000",
               label=f'Raio final = {raio_final*1e6:.3f} μm')
axs[0, 1].set_xlabel('Tempo (s)')
axs[0, 1].set_ylabel('Raio (μm)')
axs[0, 1].set_xscale('log')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].set_title('Evolução do Raio')

# Temperatura
axs[1, 0].plot(tempo_rk3, temperaturas - 273.15, "#00AA00",
               label=f'Temperatura final = {temp_final-273.15:.2f} °C')
axs[1, 0].set_xlabel('Tempo (s)')
axs[1, 0].set_ylabel('Temperatura (°C)')
axs[1, 0].set_xscale('log')
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].set_title('Evolução da Temperatura (equilíbrio rápido)')

plt.suptitle(f'RK3 Adaptativo com Separação de Escalas\nTol = {tolerancia}',
             fontsize=16)
plt.tight_layout()
plt.show()
"""