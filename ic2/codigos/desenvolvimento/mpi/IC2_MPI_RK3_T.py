"""
IC2_RK3_MPI.py
--------------
Cálculo paralelo (MPI) da evolução temporal da temperatura de gotículas espuma.

Estratégia de paralelização:
  - Cada processo MPI recebe um subconjunto de raios iniciais (r_i_lista).
  - O processo 0 (root) distribui os raios com scatter, coleta os resultados
    com gather e salva/exibe os dados.

Execução:
  mpirun -n 4 python IC2_RK3_MPI.py

Dependências:
  mpi4py, numpy  (além dos módulos locais do projeto)
"""

import numpy as np
import math
from mpi4py import MPI

# ── módulos locais do projeto ────────────────────────────────────────────────
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela
# ────────────────────────────────────────────────────────────────────────────

# ── comunicador MPI ──────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # id do processo (0 = root)
size = comm.Get_size()   # número total de processos
# ────────────────────────────────────────────────────────────────────────────


# ============================================================
# 1. PARÂMETROS GLOBAIS (iguais em todos os processos)
# ============================================================
T_a          = 18
T_a_em_k     = T_a + 273.15
T_mar_em_k   = 293.15
T_gota       = 19.8
T_gota_em_k  = T_gota + 273.15
P            = 1000
M_H2O        = 18.016e-3
R            = 8.31
M_NaCl       = 58.443e-3
rho_w        = 1025
v_ion        = 2
s            = 34 / 1000
f            = 0.9
C_ar         = 0.0154
S            = 34
R_atm        = 0.082
g            = 9.81
v_ar         = 1.32e-5
rho_ar       = 1.225
H_s          = 6
T0           = 273.15
P0           = 1013.25
c_ps         = 4000
dt_global    = 1e-4          # passo de tempo global


# ============================================================
# 2. LISTA DE RAIOS INICIAIS A SIMULAR
#    (definida apenas no root; distribuída via scatter)
# ============================================================
if rank == 0:
    # Exemplo: 8 raios entre 30 µm e 500 µm
    r_i_lista = np.linspace(30e-6, 500e-6, 3)
    print(f"[root] Simulando {len(r_i_lista)} gotículas com {size} processo(s) MPI.")
    print(f"[root] Raios (µm): {r_i_lista * 1e6}")

    # Distribui os raios em blocos para cada processo
    # Usa listas de Python para scatter (cada elemento = raios do processo i)
    chunks = [[] for _ in range(size)]
    for idx, r in enumerate(r_i_lista):
        chunks[idx % size].append(r)
else:
    chunks = None

# Scatter: cada processo recebe sua sublista de raios
meus_raios = comm.scatter(chunks, root=0)
print(f"[rank {rank}] Recebi {len(meus_raios)} raio(s): {[f'{r*1e6:.1f} µm' for r in meus_raios]}")


# ============================================================
# 3. FUNÇÕES AUXILIARES
# ============================================================

def calcular_tau_f(r_i_local):
    """Tempo de vida da gotícula via velocidade terminal."""
    def func(U_f):
        const = (2 * r_i_local**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
        a = 1 + 0.158 * ((2 * r_i_local * U_f / v_ar) ** (2/3))
        return U_f - const / a

    def dfunc(U_f):
        const = (2 * r_i_local**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
        a = 1 + 0.158 * ((2 * r_i_local * U_f / v_ar) ** (2/3))
        d_a = -0.158 * (2/3) * (2 * r_i_local / v_ar)**(2/3) * U_f**(-1/3)
        return 1 - (const * d_a) / (a**2)

    U_f_sol = metodo_newton.Newton(func, dfunc, 0.1, 1e-6, 50)
    return H_s / (2 * U_f_sol)


def calcular_parametros_iniciais(r_i_local):
    """Calcula todos os parâmetros físicos para um raio inicial dado."""
    e_sat_val       = e_sat.calcular_esat(T_a)
    L_v_val         = L_v.calcular_lv(T_gota)
    D_linha_w_val   = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i_local, M_H2O, T0, P0,
                                                    alpha_c=0.036, Delta_w=8e-8)
    k_linha_a_val   = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i_local, R, T0, P0,
                                                     M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7,
                                                     c_pa=1.006e3)
    rho_ww          = m_s.calcular_rho_ww(T_gota)
    m_s_val         = m_s.calcular_ms(rho_ww, r_i_local, s)
    v_a             = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i_local)
    m_w             = rho_s.calcular_massa_agua(r_i_local, rho_ww)
    sigma_s_val     = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
    Phi_s_val       = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
    rho_s_val       = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

    Delta_T_val     = Delta_T_contas.calculate_delta_T(
        T_a_em_k,
        alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
        beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O,
                                     D_linha_w_val, R, k_linha_a_val),
        b=240.97,
        exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R,
                                        v_ion, Phi_s_val, m_s_val, r_i_local, rho_s_val, M_NaCl),
        f=f
    )
    rho_v_val       = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k)

    exp_y_val       = code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R,
                                                 v_ion, Phi_s_val, m_s_val, r_i_local, rho_s_val, M_NaCl)
    rho_vr_val      = rho_vr.calcular_rho_vr(M_H2O, e_sat_esp.calcular_esat(T_a),
                                              exp_y_val, R, T_a_em_k)
    partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(T_a_em_k, rho_vr_val,
                                                                  a=17.502, b=240.97)
    tau_T_val       = contas_tau_T.calculate_tau_T(rho_s_val, r_i_local, k_linha_a_val,
                                                    D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps)
    T_eq_val        = Delta_T_val[0] + T_a
    T_eq_em_k       = T_eq_val + 273.15

    def zeta(r):
        denom = (4 * np.pi * rho_s_val * (r**3 / 3)) - m_s_val
        return ((f - 1)
                - (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r)
                + (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / denom)

    def dzeta_dr(r):
        denom = (4 * np.pi * rho_s_val * (r**3)) / 3 - m_s_val
        t1 = (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r**2)
        t2 = (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)
               * 4 * np.pi * rho_s_val * r**2) / denom**2
        return t1 - t2

    r0      = 1.1 * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1/3)
    r_eq    = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)
    tau_r   = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i_local,
                              rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl,
                              D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

    return dict(
        e_sat_val=e_sat_val, L_v_val=L_v_val, D_linha_w_val=D_linha_w_val,
        k_linha_a_val=k_linha_a_val, m_s_val=m_s_val, sigma_s_val=sigma_s_val,
        Phi_s_val=Phi_s_val, rho_s_val=rho_s_val, rho_v_val=rho_v_val,
        tau_T=tau_T_val, tau_r=tau_r, r_eq=r_eq, T_eq_em_k=T_eq_em_k
    )


# ============================================================
# 4. SISTEMA DE EDOs  (m, r, T)
# ============================================================

def f_system(t, variaveis, params):
    """Retorna [dm/dt, dr/dt, dT/dt] para o estado atual."""
    m, r, T = variaveis
    m_s_val   = params['m_s_val']
    rho_s_val = params['rho_s_val']
    rho_v_val = params['rho_v_val']
    e_sat_val = params['e_sat_val']

    # ── dm/dt ────────────────────────────────────────────────
    H_novo  = H_estrela.calcular_H_estrela(T, S)
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol     = (4/3) * np.pi * r**3
    C_gota  = m / vol
    dm_dt   = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))

    # ── dr/dt ────────────────────────────────────────────────
    rho_ww_novo   = m_s.calcular_rho_ww(T - 273.15)
    m_w_novo      = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo  = sigma_s.calculate_sigma_s(T - 273.15, m_s_val, m_w_novo)
    Phi_s_novo    = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_lw_novo     = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0,
                                                  alpha_c=0.036, Delta_w=8e-8)
    L_v_novo      = L_v.calcular_lv(T - 273.15)
    k_la_novo     = k_linha_a.calculate_K_linha_a(T - 273.15, T, P, r, R, T0, P0,
                                                   M_a=28.9644e-3, alpha_T=0.7,
                                                   delta_T=2.16e-7, c_pa=1.006e3)
    exp_y_novo    = code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R,
                                               v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl)
    rho_vr_novo   = rho_vr.calcular_rho_vr(M_H2O, e_sat_esp.calcular_esat(T - 273.15),
                                             exp_y_novo, R, T_a_em_k)

    Y    = ((2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r))
            - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val)))
    den1 = rho_s_val * R * T_a_em_k / (D_lw_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_la_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    dr_dt = ((f - 1) - Y) / (r * (den1 + den2))

    # ── dT/dt ────────────────────────────────────────────────
    dT_dt = (3 / (rho_s_val * c_ps * r**2)
             * (k_la_novo * (T_a_em_k - T) + L_v_novo * D_lw_novo * (rho_v_val - rho_vr_novo)))

    return np.array([dm_dt, dr_dt, dT_dt])


# ============================================================
# 5. INTEGRADORES RK3
# ============================================================

def rk3_passo(f_sys, t, y, dt, params):
    """Um passo RK3 (Shu-Osher) para o vetor y."""
    k1 = f_sys(t,          y,                           params)
    k2 = f_sys(t + dt/2,   y + (dt/2) * k1,             params)
    k3 = f_sys(t + dt,     y - dt * k1 + 2*dt * k2,     params)
    return y + (dt/6) * (k1 + 4*k2 + k3)


def rk3_subpassos_T(f_sys, y0, tempo, n_sub, params):
    """
    RK3 com subpassos na equação da temperatura.
    
    A cada passo global dt:
      1. Integra T com n_sub subpassos (dt_sub = dt/n_sub), m e r fixos.
      2. Integra m e r com um passo global usando T atualizado.
    """
    num_pontos = len(tempo)
    y = np.zeros((num_pontos, 3))
    y[0] = y0

    for n in range(num_pontos - 1):
        dt  = tempo[n+1] - tempo[n]
        dts = dt / n_sub
        m_n, r_n, T_n = y[n]

        # — subpassos para T ——————————————————————————————
        T_cur = T_n
        for _ in range(n_sub):
            k1_T = f_sys(0, np.array([m_n, r_n, T_cur]),         params)[2]
            k2_T = f_sys(0, np.array([m_n, r_n, T_cur + (dts/2)*k1_T]), params)[2]
            k3_T = f_sys(0, np.array([m_n, r_n, T_cur - dts*k1_T + 2*dts*k2_T]), params)[2]
            T_cur += (dts/6) * (k1_T + 4*k2_T + k3_T)
        T_novo = T_cur

        # — passo global para m e r ————————————————————————
        d1   = f_sys(0, np.array([m_n,                         r_n,                         T_novo]), params)
        d2   = f_sys(0, np.array([m_n + (dt/2)*d1[0],          r_n + (dt/2)*d1[1],          T_novo]), params)
        d3   = f_sys(0, np.array([m_n - dt*d1[0] + 2*dt*d2[0], r_n - dt*d1[1] + 2*dt*d2[1], T_novo]), params)
        m_novo = m_n + (dt/6) * (d1[0] + 4*d2[0] + d3[0])
        r_novo = r_n + (dt/6) * (d1[1] + 4*d2[1] + d3[1])

        y[n+1] = [m_novo, r_novo, T_novo]

    return y


# ============================================================
# 6. LOOP PRINCIPAL — cada processo calcula seus raios
# ============================================================

resultados_locais = []   # lista de dicts com resultados de cada gotícula

for r_i_local in meus_raios:
    print(f"[rank {rank}] Iniciando r_i = {r_i_local*1e6:.1f} µm ...")

    # Parâmetros físicos para este raio
    params = calcular_parametros_iniciais(r_i_local)

    # Tempo de vida e discretização
    tau_f   = calcular_tau_f(r_i_local)
    n_steps = int(tau_f / dt_global)
    tempo   = np.linspace(0, tau_f, n_steps)

    # Número de subpassos para T (garante dt_sub < tau_T)
    n_sub = max(1, math.ceil(dt_global / params['tau_T']))

    # Condições iniciais: [m_CO2_inicial, r_i, T_gota_em_k]
    y0 = np.array([1.2e-15, r_i_local, T_gota_em_k])

    # Integração
    sol = rk3_subpassos_T(f_system, y0, tempo, n_sub, params)

    resultados_locais.append({
        'r_i_um'        : r_i_local * 1e6,
        'rank'          : rank,
        'n_sub'         : n_sub,
        'tau_T'         : params['tau_T'],
        'tau_f'         : tau_f,
        'massa_final'   : sol[-1, 0],
        'raio_final_um' : sol[-1, 1] * 1e6,
        'T_final_C'     : sol[-1, 2] - 273.15,
        'T_eq_C'        : params['T_eq_em_k'] - 273.15,
        # Guarda a série temporal de T para análise
        'tempo'         : tempo,
        'T_serie'       : sol[:, 2] - 273.15,
        'r_serie_um'    : sol[:, 1] * 1e6,
    })
    print(f"[rank {rank}] r_i={r_i_local*1e6:.1f} µm — "
          f"T_final={sol[-1,2]-273.15:.4f} °C  "
          f"r_final={sol[-1,1]*1e6:.2f} µm  "
          f"n_sub={n_sub}")


# ============================================================
# 7. COLETA DOS RESULTADOS NO ROOT (gather)
# ============================================================

# gather retorna uma lista de listas (uma por processo)
todos_resultados_aninhados = comm.gather(resultados_locais, root=0)

if rank == 0:
    # Achata a lista de listas
    todos = [item for sublista in todos_resultados_aninhados for item in sublista]
    # Ordena por raio inicial
    todos.sort(key=lambda d: d['r_i_um'])

    print("\n" + "="*65)
    print(f"{'r_i (µm)':>10} | {'T_final (°C)':>13} | {'T_eq (°C)':>10} | "
          f"{'r_final (µm)':>13} | {'n_sub':>6}")
    print("-"*65)
    for res in todos:
        print(f"{res['r_i_um']:>10.1f} | {res['T_final_C']:>13.4f} | "
              f"{res['T_eq_C']:>10.4f} | {res['raio_final_um']:>13.2f} | "
              f"{res['n_sub']:>6d}")
    print("="*65)

    # ── Gráfico: evolução de T para todos os raios ──────────
    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        for res in todos:
            lbl = f"r_i = {res['r_i_um']:.0f} µm"
            axs[0].plot(res['tempo'], res['T_serie'], label=lbl)
            axs[1].plot(res['tempo'], res['r_serie_um'], label=lbl)

        axs[0].set_xlabel('Tempo (s)')
        axs[0].set_ylabel('Temperatura (°C)')
        axs[0].set_xscale('log')
        axs[0].set_title('Evolução da Temperatura — RK3 + subpassos (MPI)')
        axs[0].legend(fontsize=8)
        axs[0].grid(True, alpha=0.3)

        axs[1].set_xlabel('Tempo (s)')
        axs[1].set_ylabel('Raio (µm)')
        axs[1].set_xscale('log')
        axs[1].set_title('Evolução do Raio — RK3 + subpassos (MPI)')
        axs[1].legend(fontsize=8)
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('resultado_MPI.png', dpi=150)
        print("\n[root] Gráfico salvo em resultado_MPI.png")
        plt.show()

    except ImportError:
        print("[root] matplotlib não disponível — gráfico ignorado.")