"""

    mpiexec -n 4 python mpi_IC2_completo_RK3_subpassos.py
"""

from mpi4py import MPI
import numpy as np
import math

from NEWTON import metodo_newton
from T_EQ.BETA import D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from T_EQ.BETA import code_beta
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela

# 1. INICIALIZAÇÃO MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 2. PARÂMETROS FÍSICOS  (idênticos ao código serial)
T_a          = 18
T_a_em_k     = T_a + 273.15
T_mar_em_k   = 293.15
T_gota       = 19.8
T_gota_em_k  = T_gota + 273.15
P            = 1000
M_H2O        = 18.016e-3
r_i          = 100e-6
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

# 3. PRÉ-CÁLCULOS  (todos os ranks calculam — evita broadcast de muitas vars)

# Velocidade terminal → Newton
def func_Uf(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    return U_f - const / a

def dfunc_Uf(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    d_a = -0.158 * (2 / 3) * (2 * r_i / v_ar) ** (2 / 3) * U_f ** (-1 / 3)
    return 1 - (const * d_a) / (a**2)

U_f_sol = metodo_newton.Newton(func_Uf, dfunc_Uf, 0.1, 1e-6, 50)
tau_f   = H_s / (2 * U_f_sol)

e_sat_val     = e_sat.calcular_esat(T_a)
L_v_val       = L_v.calcular_lv(T_gota)
D_linha_w_val = D_linha_w.calculate_Dw_prime(
    T_gota_em_k, R, P, r_i, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val = k_linha_a.calculate_K_linha_a(
    T_gota, T_gota_em_k, P, r_i, R, T0, P0,
    M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)

rho_ww    = m_s.calcular_rho_ww(T_gota)
m_s_val   = m_s.calcular_ms(rho_ww, r_i, s)
v_a       = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w       = rho_s.calcular_massa_agua(r_i, rho_ww)
sigma_s_val = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
Phi_s_val   = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
rho_s_val   = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

Delta_T = Delta_T_contas.calculate_delta_T(
    T_a_em_k,
    alpha=alpha.calcular_alpha(T_a_em_k, a=17.502, b=240.97),
    beta=code_beta.calcular_beta(e_sat_val, T_a_em_k, L_v_val, M_H2O,
                                 D_linha_w_val, R, k_linha_a_val),
    b=240.97,
    exp_y=code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R,
                                    v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    f=f
)

rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k)

rho_vr_val = rho_vr.calcular_rho_vr(
    M_H2O,
    e_sat_esp.calcular_esat(T_a),
    code_exp_y.calcular_exp_y(M_H2O, sigma_s_val, T_a_em_k, rho_w, R,
                              v_ion, Phi_s_val, m_s_val, r_i, rho_s_val, M_NaCl),
    R, T_a_em_k
)

partial_rho_vr_val = partial_rho_vr.calcular_partial_rho_vr(
    T_a_em_k, rho_vr_val, a=17.502, b=240.97)

tau_T = contas_tau_T.calculate_tau_T(
    rho_s_val, r_i, k_linha_a_val, D_linha_w_val,
    partial_rho_vr_val, L_v_val, c_ps)

T_eq      = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

# Raio de equilíbrio → Newton
def zeta(r):
    num = (4 * np.pi * rho_s_val * (r**3 / 3)) - m_s_val
    return ((f - 1)
            - (2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r)
            + (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)) / num)

def dzeta_dr(r):
    den = (4 * np.pi * rho_s_val * (r**3)) / 3 - m_s_val
    return ((2 * M_H2O * sigma_s_val) / (R * T_eq_em_k * rho_w * r**2)
            - (v_ion * Phi_s_val * m_s_val * (M_H2O / M_NaCl)
               * 4 * np.pi * rho_s_val * r**2) / den**2)

r0_newton = 1.1 * (3 * m_s_val / (4 * np.pi * rho_s_val)) ** (1 / 3)
r_eq      = metodo_newton.Newton(zeta, dzeta_dr, r0_newton, 1e-6, 100)

tau_r = calcular_tau_r(
    f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val,
    m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val,
    e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]

# Condição inicial de m: equilíbrio com a superfície do oceano
H_0   = H_estrela.calcular_H_estrela(T_gota_em_k, S)
Dg_0  = Dg_estrela.calcular_Dg_estrela(r_i, T_a_em_k, R_atm)
tau_m_inicial = (H_0 * R_atm * T_gota_em_k * r_i**2) / (3.0 * Dg_0)

vol_i = (4.0 / 3.0) * np.pi * r_i**3
m_0   = C_ar * H_0 * R_atm * T_mar_em_k * vol_i

# 4. DERIVADAS DO SISTEMA  {r, T, m}

def func_dTdt(r, T):
    """dT/dt — r fixo, T variável."""
    T_c      = T - 273.15
    rho_ww_n = m_s.calcular_rho_ww(T_c)
    m_w_n    = rho_s.calcular_massa_agua(r, rho_ww_n)
    sigma_n  = sigma_s.calculate_sigma_s(T_c, m_s_val, m_w_n)
    Phi_n    = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_n)
    D_n      = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0,
                                             alpha_c=0.036, Delta_w=8e-8)
    Lv_n     = L_v.calcular_lv(T_c)
    k_n      = k_linha_a.calculate_K_linha_a(T_c, T, P, r, R, T0, P0,
                                              M_a=28.9644e-3, alpha_T=0.7,
                                              delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_n = rho_vr.calcular_rho_vr(
        M_H2O,
        e_sat_esp.calcular_esat(T_c),
        code_exp_y.calcular_exp_y(M_H2O, sigma_n, T_a_em_k, rho_w, R,
                                  v_ion, Phi_n, m_s_val, r, rho_s_val, M_NaCl),
        R, T_a_em_k
    )
    return (3.0 / (rho_s_val * c_ps * r**2)) * (
        k_n * (T_a_em_k - T) + Lv_n * D_n * (rho_v_val - rho_vr_n)
    )


def func_drdt(r, T):
    """dr/dt — T fixo, r variável."""
    T_c      = T - 273.15
    rho_ww_n = m_s.calcular_rho_ww(T_c)
    m_w_n    = rho_s.calcular_massa_agua(r, rho_ww_n)
    sigma_n  = sigma_s.calculate_sigma_s(T_c, m_s_val, m_w_n)
    Phi_n    = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_n)
    D_n      = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0,
                                             alpha_c=0.036, Delta_w=8e-8)
    Lv_n     = L_v.calcular_lv(T_c)
    k_n      = k_linha_a.calculate_K_linha_a(T_c, T, P, r, R, T0, P0,
                                              M_a=28.9644e-3, alpha_T=0.7,
                                              delta_T=2.16e-7, c_pa=1.006e3)
    Y    = ((2 * M_H2O * sigma_n) / (R * T_a_em_k * rho_w * r)
            - (v_ion * Phi_n * m_s_val * (M_H2O / M_NaCl)) / (m_w_n - m_s_val))
    den1 = rho_s_val * R * T_a_em_k / (D_n * M_H2O * e_sat_val)
    den2 = (rho_s_val * Lv_n / (k_n * T_a_em_k)) * (Lv_n * M_H2O / (R * T_a_em_k) - 1)
    return ((f - 1) - Y) / (r * (den1 + den2))


def func_dmdt(r, T, m):
    """dm/dt — r e T fixos, m variável.
    Transferência de gás dissolvido (Lei de Henry + difusão molecular).
    """
    H_n   = H_estrela.calcular_H_estrela(T, S)
    Dg_n  = Dg_estrela.calcular_Dg_estrela(r, T, R_atm)
    vol   = (4.0 / 3.0) * np.pi * r**3
    C_gota = m / vol
    return 4.0 * np.pi * r * Dg_n * (C_ar - C_gota / (H_n * R_atm * T))

# 5. SSP-RK3 para cada variável (Shu & Osher 1988)

def ssp_rk3_T(T_n, r_fix, dt):
    F0 = func_dTdt(r_fix, T_n)
    T1 = T_n + dt * F0
    F1 = func_dTdt(r_fix, T1)
    T2 = (3/4)*T_n + (1/4)*(T1 + dt * F1)
    F2 = func_dTdt(r_fix, T2)
    return (1/3)*T_n + (2/3)*(T2 + dt * F2)

def ssp_rk3_r(r_n, T_fix, dt):
    F0 = func_drdt(r_n, T_fix)
    r1 = r_n + dt * F0
    F1 = func_drdt(r1, T_fix)
    r2 = (3/4)*r_n + (1/4)*(r1 + dt * F1)
    F2 = func_drdt(r2, T_fix)
    return (1/3)*r_n + (2/3)*(r2 + dt * F2)

def ssp_rk3_m(m_n, r_fix, T_fix, dt):
    F0 = func_dmdt(r_fix, T_fix, m_n)
    m1 = m_n + dt * F0
    F1 = func_dmdt(r_fix, T_fix, m1)
    m2 = (3/4)*m_n + (1/4)*(m1 + dt * F1)
    F2 = func_dmdt(r_fix, T_fix, m2)
    return (1/3)*m_n + (2/3)*(m2 + dt * F2)

# 6. INTEGRADOR DE FATIA  —  splitting Lie-Trotter: T → r → m

def integra_fatia(estado_inicial, tempo_fatia, tau_T, tau_r):
    """
    Integra o sistema {r, T, m} sobre `tempo_fatia` partindo de `estado_inicial`.

    Retorna:
        y : array (n_pontos, 3)  com colunas [r, T, m]
    """
    n = len(tempo_fatia)
    y = np.zeros((n, 3))
    y[0] = estado_inicial          # [r0, T0_K, m0]

    for i in range(n - 1):
        dt_global    = tempo_fatia[i + 1] - tempo_fatia[i]
        r_n, T_n, m_n = y[i]

        # ── Subpassos de T (processo rápido)
        n_sub_T  = math.ceil(dt_global / tau_T)
        dt_sub_T = dt_global / n_sub_T
        T_atual  = T_n
        for _ in range(n_sub_T):
            T_atual = ssp_rk3_T(T_atual, r_n, dt_sub_T)
        T_novo = T_atual

        # ── Subpassos de r (processo rápido)
        n_sub_r  = math.ceil(dt_global / tau_r)
        dt_sub_r = dt_global / n_sub_r
        r_atual  = r_n
        for _ in range(n_sub_r):
            r_atual = ssp_rk3_r(r_atual, T_novo, dt_sub_r)
        r_novo = r_atual

        # ── Subpassos de m  (tau_m dinâmico com r_novo e T_novo)
        H_novo      = H_estrela.calcular_H_estrela(T_novo, S)
        Dg_novo     = Dg_estrela.calcular_Dg_estrela(r_novo, T_novo, R_atm)
        tau_m_inst  = max((H_novo * R_atm * T_novo * r_novo**2) / (3.0 * Dg_novo), 1e-9)

        n_sub_m  = math.ceil(dt_global / tau_m_inst)
        dt_sub_m = dt_global / n_sub_m
        m_atual  = m_n
        for _ in range(n_sub_m):
            m_atual = ssp_rk3_m(m_atual, r_novo, T_novo, dt_sub_m)
        m_novo = m_atual

        y[i + 1] = [r_novo, T_novo, m_novo]

    return y

# 7. DIVISÃO DO TEMPO GLOBAL E PIPELINE MPI
dt_global_fixo = 1
tempo_final    = 1e3
n_steps        = int(tempo_final / dt_global_fixo)
tempo_global   = np.linspace(0, tempo_final, n_steps)

fatias_indices = np.array_split(np.arange(n_steps), size)
meus_indices   = fatias_indices[rank]

# Estado inicial global
estado_entrada = np.array([r_i, T_gota_em_k, m_0])

resultados_locais = None   # preenchido quando este rank rodar sua fatia

# Pipeline sequencial: rank k só começa depois que rank k-1 terminou e enviou
# o estado final como condição inicial para k.
for k in range(size):
    comm.Barrier()

    if k == rank:
        tempo_fatia = tempo_global[meus_indices]

        # Recebe estado inicial do rank anterior (exceto rank 0)
        if rank > 0:
            estado_entrada = comm.recv(source=rank - 1, tag=rank)

        # Integra a fatia deste rank
        resultados_locais = integra_fatia(estado_entrada, tempo_fatia, tau_T, tau_r)

        # Envia estado final ao rank seguinte
        if rank < size - 1:
            comm.send(resultados_locais[-1], dest=rank + 1, tag=rank + 1)

    comm.Barrier()

# 8. COLETA E SAÍDA (rank 0)
todos_resultados = comm.gather(resultados_locais, root=0)

if rank == 0:
    resultado_global = np.concatenate(todos_resultados, axis=0)  # (n_steps, 3)

    raio_evolucao = resultado_global[:, 0]
    temp_evolucao = resultado_global[:, 1]
    mass_evolucao = resultado_global[:, 2]

    #print(f"  Processos MPI            : {size}")
    #print(f"  τ_T                      : {tau_T:.3e} s")
    #print(f"  τ_r                      : {tau_r:.3e} s")
    #print(f"  τ_m_inicial              : {tau_m_inicial:.3e} s")
    #print(f"  Passo global             : {dt_global_fixo} s")
    #print(f"  Total de passos globais  : {n_steps}")
    print(f"Raio final               : {raio_evolucao[-1] * 1e6} µm")
    print(f"Temperatura final        : {temp_evolucao[-1] - 273.15} °C")
    print(f"Massa de gás final       : {mass_evolucao[-1]}")



""""
    # ── Gráfico (apenas rank 0 plota)
    try:
        import matplotlib.pyplot as plt

        plt.rcParams['text.usetex'] = False
        fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
        fig.suptitle(
            f'r(t), T(t) e m(t) + Subpassos  +  MPI ({size} ranks)\n'
            f'dt_global = {dt_global_fixo} s  |  '
            f'τ_T = {tau_T:.3e} s  |  '
            f'τ_r = {tau_r:.3e} s  |  '
            f'{n_steps} pontos',
            fontsize=12
        )

        # Raio
        axs[0].semilogx(tempo_global, raio_evolucao * 1e6, 'o-', color='#0D00FF',
                        linewidth=2.0, markersize=3,
                        label=f'r_final = {raio_evolucao[-1]*1e6} µm')
        axs[0].set_ylabel('Raio da Gota (µm)', fontsize=12)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=10)

        # Temperatura
        axs[1].semilogx(tempo_global, temp_evolucao - 273.15, 'o-', color='#FF0000',
                        linewidth=2.0, markersize=3,
                        label=f'T_final = {temp_evolucao[-1]-273.15} °C')
        axs[1].set_ylabel('Temperatura da Gota (°C)', fontsize=12)
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=10)

        # Massa de gás
        axs[2].semilogx(tempo_global, mass_evolucao, 'o-', color='#008000',
                        linewidth=2.0, markersize=3,
                        label=f'm_final = {mass_evolucao[-1]}')
        axs[2].set_xlabel('Tempo (s)', fontsize=12)
        axs[2].set_ylabel('Massa (mol)', fontsize=12)
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    except ImportError:
        print("matplotlib não disponível — pulando gráfico.")
"""