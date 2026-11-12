import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import matplotlib.pyplot as plt

# ================== CONSTANTES (idênticas ao passo duplo) ==================
T_a = 18
T_a_em_k = T_a + 273.15
T_mar_em_k = 293.15
T_gota = 19.8
T_gota_em_k = T_gota + 273.15
P = 1000
M_H2O = 18.016e-3
r_i = 100e-6
R = 8.31
M_NaCl = 58.443e-3
rho_w = 1025
v_ion = 2
s = 34/1000
f = 0.9
C_ar = 0.0154
S = 34
R_atm = 0.082
g = 9.81
v_ar = 1.32e-5
rho_ar = 1.225
H_s = 6
T0 = 273.15
P0 = 1013.25
c_ps = 4000

# Cálculos preliminares (velocidade final, tempos característicos, etc.)
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
tau_f = H_s / (2 * U_f_sol)

e_sat_val = e_sat.calcular_esat(T_a)
L_v_val = L_v.calcular_lv(T_gota)
D_linha_w_val = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)

rho_ww = m_s.calcular_rho_ww(T_gota)
m_s_val = m_s.calcular_ms(rho_ww, r_i, s)
v_a = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w = rho_s.calcular_massa_agua(r_i, rho_ww)
sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
Phi_s_val = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
rho_s_val = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

Delta_T = Delta_T_contas.calculate_delta_T(
    T_a_em_k, 
    alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O, D_linha_w_val, R, k_linha_a_val),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    f=f
)
rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k) 

rho_vr_val = rho_vr.calcular_rho_vr(
    M_H2O, 
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    R, 
    T_a_em_k
)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val, a=17.502, b=240.97)
tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps)

T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

def zeta(r_i):
    term1 = (f - 1)
    term2 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r_i)
    denominator = (4 * np.pi * rho_s_val * (r_i**3 / 3)) - m_s_val
    term3 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / denominator
    return term1 - term2 + term3

def dzeta_dr(r_i):
    term1 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * (r_i**2))
    denominator = (4 * np.pi * rho_s_val * (r_i**3)) / 3 - m_s_val
    term2 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl) * 4 * np.pi * rho_s_val * (r_i**2)) / (denominator**2)
    return term1 - term2

r0 = (1.1) * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1 / 3)
r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)
tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

# ================== FUNÇÃO DO SISTEMA (mesma do passo duplo) ==================
def f_system(t, variaveis_gota):
    m, r, T = variaveis_gota
    
    # dm/dt (troca gasosa – CO2)
    H_novo = H_estrela.calcular_H_estrela(T, S)
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))
    
    # dr/dt (equação do raio)
    rho_ww_novo = m_s.calcular_rho_ww(T-273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T-273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T-273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T-273.15, T, P, r, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
    
    Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r)) - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val))
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2
    dr_dt = ((f - 1) - Y) / (r * den)
    
    # dT/dt (equação da temperatura)
    rho_vr_novo = rho_vr.calcular_rho_vr(
        M_H2O,
        e_sat_esp.calcular_esat(T-273.15),
        code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R, v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
        R,
        T_a_em_k
    )
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo))
    
    return np.array([dm_dt, dr_dt, dT_dt])

# ================== SOLVER RKF45 VETORIAL ==================
def rkf45_adaptativo(f, y0, t_final, tol, dt_inicial=1e-6, eta=0.9, dt_min=1e-13, dt_max=np.inf):
    """
    Resolve um sistema de EDOs y' = f(t,y) usando o método RKF45 com controle adaptativo de passo.
    
    Parâmetros
    ----------
    f : callable
        Função que retorna a derivada (array) dado t e y (array).
    y0 : array_like
        Condição inicial.
    t_final : float
        Tempo final de integração.
    tol : float
        Tolerância relativa para o erro estimado (critério de máximo erro relativo entre componentes).
    dt_inicial : float, opcional
        Passo inicial.
    eta : float, opcional
        Fator de segurança para ajuste do passo (padrão 0.9).
    dt_min, dt_max : float, opcional
        Passos mínimo e máximo permitidos.
    
    Retorna
    -------
    t_list : ndarray
        Tempos nos quais a solução foi armazenada.
    y_list : ndarray
        Valores das variáveis em cada tempo (shape = (n_passos+1, n_eq)).
    dt_list : ndarray
        Passos utilizados (comprimento = n_passos).
    """
    t = 0.0
    dt = dt_inicial
    y = np.array(y0, dtype=float)
    n_eq = len(y)

    t_list = [t]
    y_list = [y.copy()]
    dt_list = []

    # Coeficientes do tableau RKF45 (Cash-Karp)
    c2 = 1/4; a21 = 1/4
    c3 = 3/8; a31 = 3/32; a32 = 9/32
    c4 = 12/13; a41 = 1932/2197; a42 = -7200/2197; a43 = 7296/2197
    c5 = 1; a51 = 439/216; a52 = -8; a53 = 3680/513; a54 = -845/4104
    c6 = 1/2; a61 = -8/27; a62 = 2; a63 = -3544/2565; a64 = 1859/4104; a65 = -11/40

    # Coeficientes para a solução de 4ª ordem (b) e de 5ª ordem (bh)
    b1 = 25/216; b3 = 1408/2565; b4 = 2197/4104; b5 = -1/5
    bh1 = 16/135; bh3 = 6656/12825; bh4 = 28561/56430; bh5 = -9/50; bh6 = 2/55

    p = 4  # ordem do método base (usada no cálculo do novo passo)

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        # Estágios (todos vetoriais)
        k1 = f(t, y)
        k2 = f(t + c2*dt, y + dt*(a21*k1))
        k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
        k4 = f(t + c4*dt, y + dt*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*dt, y + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*dt, y + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

        # Soluções de 4ª e 5ª ordem
        y4 = y + dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5)
        y5 = y + dt * (bh1*k1 + bh3*k3 + bh4*k4 + bh5*k5 + bh6*k6)

        # Estimativa do erro local (diferença entre as duas ordens)
        erro_est = np.abs(y5 - y4)

        # Critério de aceitação: máximo erro relativo (protegido para valores próximos de zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            erro_rel = np.where(np.abs(y5) > 1e-30, erro_est / np.abs(y5), erro_est / 1e-30)
        erro_max = np.max(erro_rel)

        # Aceita passo se erro for aceitável ou se dt já é mínimo
        if erro_max < tol or dt <= dt_min:
            t += dt
            y = y5  # avança com a solução de ordem 5 (mais precisa)
            t_list.append(t)
            y_list.append(y.copy())
            dt_list.append(dt)

            # Calcula novo dt (evita divisão por zero)
            if erro_max > 1e-30:
                new_dt = eta * dt * (tol / erro_max) ** (1/(p+1))
            else:
                new_dt = eta * dt * 2.0   # se erro muito pequeno, dobra
            dt = np.clip(new_dt, dt_min, dt_max)

        else:
            # Passo rejeitado: reduz dt e tenta novamente (sem avançar t)
            dt = eta * dt * (tol / erro_max) ** (1/(p+1))
            dt = max(dt, dt_min)

    return np.array(t_list), np.array(y_list), np.array(dt_list)

# ================== EXECUÇÃO ==================
variaveis_iniciais = np.array([1.2e-15, r_i, T_gota_em_k])  # [massa (mol), raio (m), temperatura (K)]
tolerancia = 1e-8
t_final = 1e3

tempo_rkf45, var_rkf45, dt_rkf45 = rkf45_adaptativo(f_system, variaveis_iniciais, t_final, tolerancia)

massa_final = var_rkf45[-1, 0]
raio_final = var_rkf45[-1, 1]
temp_final = var_rkf45[-1, 2]

print(f"Raio final           : {raio_final}")
print(f"Temperatura final    : {temp_final}")
print(f"Massa final          : {massa_final}")
print(f"Número de passos     : {len(tempo_rkf45)}")


# ================== GRÁFICOS ==================
"""
plt.rcParams['text.usetex'] = False
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# Massa vs tempo
ax[0,0].semilogx(tempo_rkf45, var_rkf45[:, 0], 'o-', linewidth=2,
                 label=f'Massa final: {massa_final} mol')
ax[0,0].set_xlabel('Tempo (s)')
ax[0,0].set_ylabel('Massa (mol)')
ax[0,0].set_title('Evolução da Massa')
ax[0,0].grid(True, alpha=0.3)
ax[0,0].legend()

# Raio vs tempo
ax[0,1].semilogx(tempo_rkf45, var_rkf45[:, 1] * 1e6, 'o-', linewidth=2,
                 label=f'Raio final: {raio_final*1e6} µm')
ax[0,1].set_xlabel('Tempo (s)')
ax[0,1].set_ylabel('Raio (µm)')
ax[0,1].set_title('Evolução do Raio')
ax[0,1].grid(True, alpha=0.3)
ax[0,1].legend()

# Temperatura vs tempo
ax[1,0].semilogx(tempo_rkf45, var_rkf45[:, 2] - 273.15, 'o-', linewidth=2,
                 label=f'Temperatura final: {temp_final-273.15} °C')
ax[1,0].set_xlabel('Tempo (s)')
ax[1,0].set_ylabel('Temperatura (°C)')
ax[1,0].set_title('Evolução da Temperatura')
ax[1,0].grid(True, alpha=0.3)
ax[1,0].legend()

# Passo dt vs tempo
ax[1,1].loglog(tempo_rkf45[1:], dt_rkf45, 'o-', markersize=3, linewidth=1,
               label='dt usado')
ax[1,1].set_xlabel('Tempo (s)')
ax[1,1].set_ylabel('Passo de tempo dt (s)')
ax[1,1].set_title('Evolução do Passo Adaptativo')
ax[1,1].grid(True, alpha=0.3, which='both')
ax[1,1].legend()

fig.suptitle(f'RKF45 - {len(tempo_rkf45)} passos', fontsize=14)
plt.tight_layout()
plt.show()
"""