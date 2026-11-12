"""
  mpiexec -n 4 python IC2_r_RK3_MPI.py
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

# 1. INICIALIZAÇÃO MPI
comm  = MPI.COMM_WORLD      # comunicador global (todos os processos)
rank  = comm.Get_rank()     # identificador deste processo  (0, 1, 2, ...)
size  = comm.Get_size()     # total de processos MPI rodando

# 2. PARÂMETROS FÍSICOS  (idênticos ao código original)
T_a         = 18
T_a_em_k    = T_a + 273.15
T_gota      = 19.8
T_gota_em_k = T_gota + 273.15
P           = 1000
M_H2O       = 18.016e-3
r_i         = 100e-6
R           = 8.31
M_NaCl      = 58.443e-3
rho_w       = 1025
v_ion       = 2
s           = 34 / 1000
f           = 0.9
g           = 9.81
v_ar        = 1.32e-5
rho_ar      = 1.225
H_s         = 6
T0          = 273.15
P0          = 1013.25
c_ps        = 4000

# 3. PRÉ-CÁLCULOS  (todos os ranks calculam — evita broadcast de muitas vars)

# Velocidade terminal → Newton
def func(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    return U_f - const / a

def dfunc(U_f):
    const = (2 * r_i**2 * g) / (9 * v_ar) * ((rho_w / rho_ar) - 1)
    a = 1 + 0.158 * ((2 * r_i * U_f / v_ar) ** (2 / 3))
    d_a = -0.158 * (2 / 3) * (2 * r_i / v_ar) ** (2 / 3) * U_f ** (-1 / 3)
    return 1 - (const * d_a) / (a**2)

U_f_sol = metodo_newton.Newton(func, dfunc, 0.1, 1e-6, 50)

e_sat_val       = e_sat.calcular_esat(T_a)
L_v_val         = L_v.calcular_lv(T_gota)
D_linha_w_val   = D_linha_w.calculate_Dw_prime(T_gota_em_k, R, P, r_i, M_H2O,
                                                T0, P0, alpha_c=0.036, Delta_w=8e-8)
k_linha_a_val   = k_linha_a.calculate_K_linha_a(T_gota, T_gota_em_k, P, r_i, R,
                                                 T0, P0, M_a=28.9644e-3, alpha_T=0.7,
                                                 delta_T=2.16e-7, c_pa=1.006e3)
rho_ww          = m_s.calcular_rho_ww(T_gota)
m_s_val         = m_s.calcular_ms(rho_ww, r_i, s)
v_a             = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w             = rho_s.calcular_massa_agua(r_i, rho_ww)
sigma_s_val     = sigma_s.calculate_sigma_s(T_gota, m_s_val, m_w)
Phi_s_val       = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w)
rho_s_val       = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)

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

T_eq      = Delta_T[0] + T_a
T_eq_em_k = T_eq + 273.15

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

# 4. EDO  dr/dt  e  SSP-RK3

def func_drdt(r):
    """dr/dt com temperatura fixada em T_eq"""
    rho_ww_n    = m_s.calcular_rho_ww(T_eq)
    m_w_n       = rho_s.calcular_massa_agua(r, rho_ww_n)
    sigma_s_n   = sigma_s.calculate_sigma_s(T_eq, m_s_val, m_w_n)
    Phi_s_n     = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_n)
    D_n         = D_linha_w.calculate_Dw_prime(T_eq_em_k, R, P, r, M_H2O,
                                               T0, P0, alpha_c=0.036, Delta_w=8e-8)
    Lv_n        = L_v.calcular_lv(T_eq)
    k_n         = k_linha_a.calculate_K_linha_a(T_eq, T_eq_em_k, P, r, R, T0, P0,
                                                M_a=28.9644e-3, alpha_T=0.7,
                                                delta_T=2.16e-7, c_pa=1.006e3)
    Y    = ((2 * M_H2O * sigma_s_n) / (R * T_a_em_k * rho_w * r)
            - (v_ion * Phi_s_n * m_s_val * (M_H2O / M_NaCl)) / (m_w_n - m_s_val))
    den1 = rho_s_val * R * T_a_em_k / (D_n * M_H2O * e_sat_val)
    den2 = (rho_s_val * Lv_n / (k_n * T_a_em_k)) * (Lv_n * M_H2O / (R * T_a_em_k) - 1)
    return ((f - 1) - Y) / (r * (den1 + den2))


def ssp_rk3_passo(r_n, dt):
    """Um passo SSP-RK3 (Shu & Osher 1988)."""
    F0 = func_drdt(r_n)
    r1 = r_n + dt * F0
    F1 = func_drdt(r1)
    r2 = (3/4)*r_n + (1/4)*(r1 + dt * F1)
    F2 = func_drdt(r2)
    return (1/3)*r_n + (2/3)*(r2 + dt * F2)


def integra_fatia(r_inicial, tempo_fatia, tau_r):
    n  = len(tempo_fatia)
    y  = np.zeros(n)
    y[0] = r_inicial

    for i in range(n - 1):
        dt_global = tempo_fatia[i + 1] - tempo_fatia[i]

        n_sub  = math.ceil(dt_global / tau_r)
        dt_sub = dt_global / n_sub

        r_atual = y[i]
        for _ in range(n_sub):
            r_atual = ssp_rk3_passo(r_atual, dt_sub)
        y[i + 1] = r_atual

    return y


dt_global_fixo = 100 
tempo_final    = 1e3          
n_steps        = int(tempo_final / dt_global_fixo)
tempo_global   = np.linspace(0, tempo_final, n_steps)

fatias_indices = np.array_split(np.arange(n_steps), size)
meus_indices   = fatias_indices[rank]         

# Condição inicial global
r_entrada = r_i    

resultados_locais = None   # será preenchido pelo rank ao rodar sua fatia

for k in range(size):
    comm.Barrier()

    if k == rank:
        tempo_fatia = tempo_global[meus_indices]

        if rank > 0:
            r_entrada = comm.recv(source=rank - 1, tag=rank)

        resultados_locais = integra_fatia(r_entrada, tempo_fatia, tau_r)

        if rank < size - 1:
            comm.send(resultados_locais[-1], dest=rank + 1, tag=rank + 1)

    comm.Barrier()

todos_resultados = comm.gather(resultados_locais, root=0)

if rank == 0:
    raio_global = np.concatenate(todos_resultados)

    print("=" * 55)
    print(f"  Processos MPI           : {size}")
    print(f"  tau_r                   : {tau_r:.3e} s")
    print(f"  Passo global            : {dt_global_fixo} s")
    print(f"  Total de passos globais : {n_steps}")
    print(f"  Raio final              : {raio_global[-1]*1e6:.4f} µm")
    print("=" * 55)

    # ── Gráfico (apenas rank 0 plota) 
    try:
        import matplotlib.pyplot as plt

        plt.rcParams['text.usetex'] = False
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(tempo_global, raio_global * 1e6, 'o-', color='#0D00FF',
                    linewidth=2.0, markersize=4,
                    label=f'r_final = {raio_global[-1]*1e6:.2f} µm')
        ax.set_xlabel('Tempo (s)', fontsize=12)
        ax.set_ylabel('Raio da Gota (µm)', fontsize=12)
        ax.set_title(
            f'Evolução do Raio + Subpassos + MPI ({size} ranks)\n'
            f'dt_global = {dt_global_fixo} s  |  τ_r = {tau_r:.3e} s',
            fontsize=12
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib não disponível — pulando gráfico.")