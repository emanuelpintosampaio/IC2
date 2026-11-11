import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros físicos (ANDREAS)
# ----------------------------------------------------------------------
T_a = 18                     # Temperatura do ar em °C
T_a_em_k = T_a + 273.15      # Temperatura do ar em K
T_mar_em_k = 293.15          # Temperatura do mar em K
T_gota = 19.8                # Temperatura da gota em °C
T_gota_em_k = T_gota + 273.15  # Temperatura da gota em K
P = 1000                     # Pressão em mb
M_H2O = 18.016e-3            # Massa molecular da água (kg/mol)
r_i = 100e-6                 # Raio inicial (m)
R = 8.31                     # Constante universal dos gases (J/mol·K)
M_NaCl = 58.443e-3           # Massa molecular do NaCl (kg/mol)
rho_w = 1025                 # Densidade da água do mar (kg/m³)
v_ion = 2                    # Número de íons por molécula de NaCl
s = 34/1000                  # Salinidade fracionária (34 psu)
f = 0.9                      # Umidade relativa fracionária
C_ar = 0.0154                # Concentração de CO₂ no ar
S = 34                       # Salinidade do mar (psu)
R_atm = 0.082                # Constante universal dos gases (L·atm/mol·K)
g = 9.81                     # Gravidade (m/s²)
v_ar = 1.32e-5               # Viscosidade cinemática do ar (m²/s)
rho_ar = 1.225               # Densidade do ar (kg/m³)
H_s = 6                      # Altura significativa da onda (m)
T0 = 273.15                  # Temperatura de referência (K)
P0 = 1013.25                 # Pressão de referência (mb)
c_ps = 4000                  # Calor específico do spray (J/kg·K)

# ----------------------------------------------------------------------
# Velocidade final de queda (U_f) e tempo de vida da gota (tau_f)
# ----------------------------------------------------------------------
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
tau_f = H_s / (2 * U_f_sol)   # tempo de vida da gota (s)

# ----------------------------------------------------------------------
# Cálculo de grandezas termodinâmicas no estado inicial
# ----------------------------------------------------------------------
e_sat_val = e_sat.calcular_esat(T_a)
L_v_val = L_v.calcular_lv(T_gota)
D_linha_w_val = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O, T0, P0,
                                             alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R, T0, P0,
                                              M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7,
                                              c_pa=1.006e3)

rho_ww = m_s.calcular_rho_ww(T_gota)
m_s_val = m_s.calcular_ms(rho_ww, r_i, s)
v_a = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w = rho_s.calcular_massa_agua(r_i, rho_ww)
sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
Phi_s_val = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
rho_s_val = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

# Delta_T (diferença de temperatura de equilíbrio)
Delta_T = Delta_T_contas.calculate_delta_T(
    T_a_em_k,
    alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O, D_linha_w_val, R, k_linha_a_val),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion,
                                    Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    f=f
)

rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k)

rho_vr_val = rho_vr.calcular_rho_vr(
    M_H2O,
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R, v_ion,
                              Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    R,
    T_a_em_k
)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val, a=17.502, b=240.97)

# Tempo de decaimento exponencial da temperatura (tau_T)
tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val,
                                      partial_rho_vr_val, L_v_val, c_ps)

# Temperatura de equilíbrio
T_eq = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

# ----------------------------------------------------------------------
# Raio de equilíbrio (r_eq)
# ----------------------------------------------------------------------
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

# Tempo de decaimento exponencial do raio (tau_r)
tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val,
                       m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val,
                       e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

# ----------------------------------------------------------------------
# Função que define a EDO para a temperatura (dT/dt)
# ----------------------------------------------------------------------
def f_T(t, T):
    r = r_i   # raio fixo (acoplar futuro)
    rho_ww_novo = m_s.calcular_rho_ww(T - 273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0,
                                                  alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T - 273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T - 273.15, T, P, r, R, T0, P0,
                                                    M_a=28.9644e-3, alpha_T=0.7,
                                                    delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo = rho_vr.calcular_rho_vr(
        M_H2O,
        e_sat_esp.calcular_esat(T - 273.15),
        code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R, v_ion,
                                  Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
        R,
        T_a_em_k
    )

    # dT/dt
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * (
        k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo)
    )
    return dT_dt

# ----------------------------------------------------------------------
# Solver RKF45 adaptativo para a temperatura
# ----------------------------------------------------------------------
def rkf45_adaptativo_temperatura(f, y0, t_final, tol):

    t = 0.0
    dt = 1e-6                     # passo inicial
    y = y0

    t_lista = [t]
    y_lista = [y]
    dt_lista = []

    p = 4                         # ordem do método base
    eta = 0.9                      # fator de segurança
    dt_min = 1e-13
    dt_max = np.inf

    # Coeficientes do tableau RKF45 (Cash-Karp)
    c2 = 1/4; a21 = 1/4
    c3 = 3/8; a31 = 3/32; a32 = 9/32
    c4 = 12/13; a41 = 1932/2197; a42 = -7200/2197; a43 = 7296/2197
    c5 = 1; a51 = 439/216; a52 = -8; a53 = 3680/513; a54 = -845/4104
    c6 = 1/2; a61 = -8/27; a62 = 2; a63 = -3544/2565; a64 = 1859/4104; a65 = -11/40

    # Coeficientes para as soluções de ordem 4 (b) e ordem 5 (bh)
    b1 = 25/216; b3 = 1408/2565; b4 = 2197/4104; b5 = -1/5
    bh1 = 16/135; bh3 = 6656/12825; bh4 = 28561/56430; bh5 = -9/50; bh6 = 2/55

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        # Estágios RKF45
        k1 = f(t, y)
        k2 = f(t + c2*dt, y + dt*(a21*k1))
        k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
        k4 = f(t + c4*dt, y + dt*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*dt, y + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*dt, y + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

        # Soluções de 4ª e 5ª ordem
        sol_quarta  = dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5)
        sol_quinta = dt * (bh1*k1 + bh3*k3 + bh4*k4 + bh5*k5 + bh6*k6)

        y_new = y + sol_quinta   # avanço com a solução de ordem 5

        # Estimativa do erro local (diferença entre as duas ordens)
        erro_est = abs(sol_quinta - sol_quarta)

        # Critério de erro relativo
        erro_relativo = erro_est / (abs(y_new))   # evita divisão por zero

        # Aceitação ou rejeição do passo
        if erro_relativo < tol or dt <= dt_min:
            t += dt
            y = y_new
            t_lista.append(t)
            y_lista.append(y)
            dt_lista.append(dt)

            if erro_relativo > 1e-30:
                new_dt = eta * dt * (tol / erro_relativo) ** (1/(p+1))
            else:
                new_dt = eta * dt * 2.0   # se erro muito pequeno, dobra
            dt = np.clip(new_dt, dt_min, dt_max)
        else:
            # Passo rejeitado: reduz dt e tenta novamente
            dt = eta * dt * (tol / erro_relativo) ** (1/(p+1))
            dt = max(dt, dt_min)

    return np.array(t_lista), np.array(y_lista), np.array(dt_lista)

# ----------------------------------------------------------------------
# Execução do solver RKF45 para a temperatura
# ----------------------------------------------------------------------
tolerancia = 1e-8
tempo, Temp, dt_usado = rkf45_adaptativo_temperatura(f_T, T_gota_em_k, 1e3, tolerancia)

print(f"Temperatura final    : {Temp[-1]}")
print(f"Número de passos     : {len(tempo)}")

# Gráficos
"""
plt.rcParams['text.usetex'] = False
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Temperatura vs tempo (escala log em x)
ax1.semilogx(tempo, Temp - T0, 'o-', linewidth=2,
            label=f'Temperatura final = {Temp[-1]-273.15} °C')
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Temperatura da gota (°C)')
ax1.set_title('Evolução da temperatura')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: Passo dt vs tempo (log-log)
ax2.loglog(tempo[1:], dt_usado, 'o-', linewidth=2)
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Passo de tempo dt (s)')
ax2.set_title('Evolução do passo adaptativo')
ax2.grid(True, alpha=0.3, which='both')
fig.suptitle(f'RKF45 - {len(tempo)} passos')

plt.tight_layout()
plt.show()
"""
