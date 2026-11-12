import numpy as np
from mpi4py import MPI
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import matplotlib.pyplot as plt

# Inicialização MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Barreira para sincronizar início
comm.Barrier()

# Parâmetros fixos (podem ser lidos de arquivo ou definidos aqui)
T_a = 18
T_a_em_k = T_a + 273.15
T_mar_em_k = 293.15
T_gota = 19.8
T_gota_em_k = T_gota + 273.15
P = 1000
M_H2O = 18.016e-3
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

# Parâmetros que podem variar (exemplo: diferentes raios iniciais)
# Cada processo recebe um subconjunto
raios_iniciais = np.logspace(-6, -3, 20)  # 20 raios diferentes
if rank == 0:
    # Divisão do trabalho
    indices = np.array_split(range(len(raios_iniciais)), size)
else:
    indices = None
# Broadcast da divisão para todos os ranks
indices = comm.scatter(indices, root=0)

# Lista para armazenar resultados locais
resultados_locais = []

for idx in indices:
    r_i = raios_iniciais[idx]
    print(f"Rank {rank} processando r_i = {r_i:.3e} m")

    # --- Cálculos preliminares (como no original) ---
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

    # Função do sistema (usa tau_T do escopo)
    def f_system(t, variaveis_gota):
        m, r, T = variaveis_gota
        H_novo = H_estrela.calcular_H_estrela(T, S)
        Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
        vol = (4 / 3) * np.pi * r**3
        C_gota = m / vol
        dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))

        rho_ww_novo = m_s.calcular_rho_ww(T-273.15)
        m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
        sigma_s_novo = sigma_s.calculate_sigma_s(T-273.15, m_s_val, m_w_novo)
        Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
        D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
        L_v_novo = L_v.calcular_lv(T-273.15)
        k_linha_a_novo = k_linha_a.calculate_K_linha_a(T-273.15, T, P, r, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
        rho_vr_novo = rho_vr.calcular_rho_vr(M_H2O, e_sat_esp.calcular_esat(T-273.15),
                                             code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R, v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
                                             R, T_a_em_k)

        Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r)) - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val))
        den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
        den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
        den = den1 + den2
        dr_dt = ((f - 1) - Y) / (r * den)

        dT_dt = 3 / (rho_s_val * c_ps * r**2) * (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo))

        return np.array([dm_dt, dr_dt, dT_dt])

    # Solver RKF45 adaptativo (igual ao original)
    def rkf45_adaptativo(f, y0, t_final, tol):
        t = 0
        dt = 1e-6
        y = np.array(y0, dtype=float)
        t_lista = [t]
        y_lista = [y]

        p = 4
        eta = 0.9
        dt_min = 1e-10
        dt_max = np.inf

        c2=1/4; a21=1/4
        c3=3/8; a31=3/32; a32=9/32
        c4=12/13; a41=1932/2197; a42=-7200/2197; a43=7296/2197
        c5=1; a51=439/216; a52=-8; a53=3680/513; a54=-845/4104
        c6=1/2; a61=-8/27; a62=2; a63=-3544/2565; a64=1859/4104; a65=-11/40

        b1=25/216; b3=1408/2565; b4=2197/4104; b5=-1/5
        bh1=16/135; bh3=6656/12825; bh4=28561/56430; bh5=-9/50; bh6=2/55

        while t < t_final:
            if t + dt > t_final:
                dt = t_final - t

            k1 = f(t, y)
            k2 = f(t + c2*dt, y + dt*(a21*k1))
            k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
            k4 = f(t + c4*dt, y + dt*(a41*k1 + a42*k2 + a43*k3))
            k5 = f(t + c5*dt, y + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
            k6 = f(t + c6*dt, y + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

            low_update = dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5)
            high_update = dt * (bh1*k1 + bh3*k3 + bh4*k4 + bh5*k5 + bh6*k6)

            y_new = y + high_update

            erro_est = np.abs(high_update - low_update)
            atol = np.array([1e-18, 1e-8, 1e-4])
            rtol = tol
            erro_toleravel = atol + rtol * np.abs(y_new)
            erro_relativo = erro_est / erro_toleravel

            # Controle adaptativo baseado no tempo (como solicitado)
            if t < tau_T:
                erro_max = erro_relativo[2]  # temperatura
            else:
                erro_max = np.max(erro_relativo[0:2])  # massa e raio

            erro_max = max(erro_max, 1e-20)

            if erro_max <= 1 or dt <= dt_min:
                t = t + dt
                y = y_new
                t_lista.append(t)
                y_lista.append(y)

                new_dt = eta * dt * (1.0 / erro_max) ** (1/(p+1))
                dt = np.clip(new_dt, dt_min, dt_max)
            else:
                dt = eta * dt * (1.0 / erro_max)**(1/(p+1))
                dt = max(dt, dt_min)

        return np.array(t_lista), np.array(y_lista)

    # Executar simulação
    tolerancia = 1e-8
    variaveis_iniciais = np.array([m_s_val, r_i, T_gota_em_k])  # massa inicial (m_s_val? cuidado: massa inicial deve ser total, não apenas sal)
    # Nota: A massa inicial no código original é 1.2e-15, o que não parece corresponder a m_s_val. Ajuste conforme necessário.
    # Vamos manter o valor original como exemplo, mas idealmente seria calculado.
    variaveis_iniciais = np.array([1.2e-15, r_i, T_gota_em_k])

    tempo, variaveis = rkf45_adaptativo(f_system, variaveis_iniciais, 1e3, tol=tolerancia)

    # Armazenar resultados: último estado e parâmetros
    resultados_locais.append({
        'r_i': r_i,
        'm_final': variaveis[-1, 0],
        'r_final': variaveis[-1, 1],
        'T_final': variaveis[-1, 2],
        'tau_T': tau_T,
        'tau_r': tau_r,
        'tau_f': tau_f
    })

# Barreira para garantir que todos terminaram suas simulações
comm.Barrier()

# Coletar todos os resultados no processo raiz
todos_resultados = comm.gather(resultados_locais, root=0)

# Processo raiz consolida e salva/plota
if rank == 0:
    # Achatar lista de listas
    resultados_finais = [item for sublist in todos_resultados for item in sublist]
    # Ordenar por raio inicial
    resultados_finais.sort(key=lambda x: x['r_i'])

    # Exemplo: salvar em arquivo
    with open('resultados_mpi.txt', 'w') as f:
        for res in resultados_finais:
            f.write(f"{res['r_i']:.3e} {res['r_final']:.3e} {res['T_final']:.2f} {res['m_final']:.3e}\n")

    # Plotar (opcional)
    raios_iniciais_plot = [r['r_i'] for r in resultados_finais]
    raios_finais_plot = [r['r_final']*1e6 for r in resultados_finais]
    plt.figure()
    plt.loglog(raios_iniciais_plot, raios_finais_plot, 'o-')
    plt.xlabel('Raio inicial (m)')
    plt.ylabel('Raio final (μm)')
    plt.title('Evolução do raio final em função do raio inicial')
    plt.grid(True)
    plt.savefig('resultados_mpi.png')
    plt.show()

# Barreira final (opcional)
comm.Barrier()