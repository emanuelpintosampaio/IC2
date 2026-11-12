import numpy as np
import matplotlib.pyplot as plt

# Importações dos módulos personalizados (mesmos do código de passo duplo)
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  

# =============================================================================
# Parâmetros físicos (idênticos ao primeiro código)
# =============================================================================
T_a = 18                     # Temperatura do ar em °C
T_a_em_k = T_a + 273.15
T_mar_em_k = 293.15
T_gota = 19.8                 # Temperatura inicial da gota em °C
T_gota_em_k = T_gota + 273.15
P = 1000                      # Pressão em mb
M_H2O = 18.016e-3             # kg/mol
r_i = 100e-6                  # Raio inicial (m)
R = 8.31                      # Constante dos gases (J/mol/K)
M_NaCl = 58.443e-3            # kg/mol
rho_w = 1025                  # kg/m³
v_ion = 2
s = 34/1000                   # Salinidade fracionária
f = 0.9                       # Umidade relativa
c_ps = 4000                   # Calor específico do spray (J/kg/K)
T0 = 273.15
P0 = 1013.25

# Grandezas calculadas que não variam com r,T (ou cuja variação é desprezada)
e_sat_val = e_sat.calcular_esat(T_a)
rho_ww = m_s.calcular_rho_ww(T_gota)
m_s_val = m_s.calcular_ms(rho_ww, r_i, s)
v_a = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w = rho_s.calcular_massa_agua(r_i, rho_ww)
rho_s_val = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)
rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k)

# =============================================================================
# Função que define o sistema de EDOs: dr/dt e dT/dt
# =============================================================================
def f_r_T(t, vars):
    r, T = vars

    # Atualiza propriedades que dependem do estado atual
    rho_ww_novo = m_s.calcular_rho_ww(T - 273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(
        T, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T - 273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(
        T - 273.15, T, P, r, R, T0, P0,
        M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)

    rho_vr_novo = rho_vr.calcular_rho_vr(
        M_H2O,
        e_sat_esp.calcular_esat(T - 273.15),
        code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R,
                                   v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
        R, T_a_em_k)

    # Termo Y (equação do raio)
    Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r)) \
        - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val))

    # Denominadores (equação do raio)
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) \
           * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2

    dr_dt = ((f - 1) - Y) / (r * den)

    # Equação da temperatura
    dT_dt = 3 / (rho_s_val * c_ps * r**2) \
            * (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo))

    return np.array([dr_dt, dT_dt])

# =============================================================================
# Resolvedor RKF45 adaptativo para sistemas de EDOs
# =============================================================================
def rkf45_sistema(f, y0, t_final, tol, dt_min=1e-13, dt_max=np.inf, eta=0.9):
    """
    Resolve o sistema de EDOs y' = f(t, y) usando o método RKF45 com controle
    adaptativo de passo baseado no máximo erro relativo entre as componentes.
    """
    t = 0.0
    dt = 1e-6                 # passo inicial
    y = np.array(y0, dtype=float)

    # Listas para armazenar a trajetória
    t_lista = [t]
    y_lista = [y.copy()]
    dt_lista = []

    # Coeficientes do tableau RKF45
    c2 = 1/4; a21 = 1/4
    c3 = 3/8; a31 = 3/32; a32 = 9/32
    c4 = 12/13; a41 = 1932/2197; a42 = -7200/2197; a43 = 7296/2197
    c5 = 1; a51 = 439/216; a52 = -8; a53 = 3680/513; a54 = -845/4104
    c6 = 1/2; a61 = -8/27; a62 = 2; a63 = -3544/2565; a64 = 1859/4104; a65 = -11/40

    # Coeficientes para as soluções de ordem 4 (b) e ordem 5 (bh)
    b1 = 25/216; b3 = 1408/2565; b4 = 2197/4104; b5 = -1/5
    bh1 = 16/135; bh3 = 6656/12825; bh4 = 28561/56430; bh5 = -9/50; bh6 = 2/55

    p = 4   # ordem do método embutido (usada no controle de passo)

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        # Estágios RKF45 (cada ki é um vetor)
        k1 = f(t, y)
        k2 = f(t + c2*dt, y + dt*(a21*k1))
        k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
        k4 = f(t + c4*dt, y + dt*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*dt, y + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*dt, y + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

        # Soluções de baixa (4ª) e alta (5ª) ordem
        sol4 = dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5)
        sol5 = dt * (bh1*k1 + bh3*k3 + bh4*k4 + bh5*k5 + bh6*k6)

        y_novo = y + sol5   # avançamos com a solução de ordem 5

        # Estimativa do erro local (diferença entre as duas ordens)
        erro_est = np.abs(sol5 - sol4)
        print(erro_est[0])

        # Cálculo do erro relativo máximo entre as componentes
        with np.errstate(divide='ignore', invalid='ignore'):
            erro_rel = erro_est[0]

        # Aceitação ou rejeição do passo
        if erro_rel < tol or dt <= dt_min:
            t += dt
            y = y_novo
            t_lista.append(t)
            y_lista.append(y.copy())
            dt_lista.append(dt)

            if erro_rel > 1e-30:
                novo_dt = eta * dt * (tol / erro_rel) ** (1/(p+1))
            else:
                novo_dt = eta * dt * 2.0   # erro muito pequeno, dobra o passo
            dt = np.clip(novo_dt, dt_min, dt_max)
        else:
            # Passo rejeitado: reduz dt e tenta novamente
            dt = eta * dt * (tol / erro_rel) ** (1/(p+1))
            dt = max(dt, dt_min)

    return np.array(t_lista), np.array(y_lista), np.array(dt_lista)

# =============================================================================
# Execução
# =============================================================================
tolerancia = 1e-8
y0 = np.array([r_i, T_gota_em_k])

t, y, dt_usado = rkf45_sistema(f_r_T, y0, t_final=1e3, tol=tolerancia)

r = y[:, 0]
T = y[:, 1]

print(f"Temperatura final    : {T[-1]} K")
print(f"Raio final           : {r[-1]*1e6} µm")
print(f"Número de passos     : {len(t)}")




# =============================================================================
# Gráficos
# =============================================================================
"""
plt.rcParams['text.usetex'] = False
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Temperatura vs tempo (escala log em x)
ax1.semilogx(t, T - 273.15, 'o-', linewidth=2,
             label=f'Temperatura final = {T[-1]-273.15} °C')
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Temperatura da gota (°C)')
ax1.set_title('Evolução da temperatura')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: Raio vs tempo (escala log em x)
ax2.semilogx(t, r * 1e6, 'o-', linewidth=2,
             label=f'Raio final = {r[-1]*1e6} μm')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Raio (µm)')
ax2.set_title('Evolução do raio')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Gráfico 3: Passo dt vs tempo (log-log)
ax3.loglog(t[1:], dt_usado, 'o-', markersize=3, linewidth=2)
ax3.set_xlabel('Tempo (s)')
ax3.set_ylabel('Passo de tempo dt (s)')
ax3.set_title('Evolução do passo adaptativo')
ax3.grid(True, alpha=0.3, which='both')

fig.suptitle(f'RKF45 – {len(t)} passos')

plt.tight_layout()
plt.show()
"""