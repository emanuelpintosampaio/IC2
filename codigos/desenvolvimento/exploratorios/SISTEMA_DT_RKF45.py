import numpy as np
from NEWTON import metodo_newton
from T_EQ.BETA import code_beta, D_linha_w, e_sat, k_linha_a, L_v
from T_EQ.EXP_Y import code_exp_y, m_s, Phi_s, rho_s, sigma_s
from T_EQ import Delta_T_contas, alpha
from TAU_T import partial_rho_vr, rho_vr, contas_tau_T, e_sat_esp, rho_v
from TAU_R.partial_rt_teste import calcular_tau_r
from GRAFICO.DGH import Dg_estrela, H_estrela  
import matplotlib.pyplot as plt

#ANDREAS
T_a = 18           # Temperatura do ar em °C ANDREAS(18)
T_a_em_k = T_a + 273.15  # Temperatura do ar em K ANDREAS(291.15)
T_mar_em_k = 293.15  # Temperatura do mar em K ANDREAS(293.15)
T_gota = 19.8      # Temperatura da gota em °C ANDREAS(20.2)
T_gota_em_k = T_gota + 273.15  # Temperatura da gota em K ANDREAS(293.35)
P = 1000           # Pressão em mb
M_H2O = 18.016e-3  # Massa molecular da água em kg/mol
r_i = 100e-6       # Raio em metros
R = 8.31           # Constante universal dos gases
M_NaCl = 58.443e-3  # Massa molecular de sal na água em kg/mol
rho_w = 1025       # Densidade da água do mar kg/m^3
v_ion = 2          # Número de íons por molécula de NaCl dissociada
s = 34/1000        # Salinidade fracionária (34 psu)
f = 0.9            # Umidade relativa fracionária ANDREAS(0.8)
C_ar = 0.0154      # Concentração do gás carbônico no ar
S = 34             # Salinidade do mar
R_atm = 0.082      # Constante universal dos gases
g = 9.81           # Gravidade
v_ar = 1.32e-5     # Viscosidade cinemática do ar
rho_ar = 1.225     # Densidade do ar
H_s = 6            # Altura significativa da onda 
T0 = 273.15
P0 = 1013.25
c_ps = 4000        # Calor específico da spray


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

# Tempo de vida da gota 
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

# Tempo de decaimento exponencial da temperatura 
tau_T = contas_tau_T.calculate_tau_T(rho_s_val, r_i, k_linha_a_val, D_linha_w_val, partial_rho_vr_val, L_v_val, c_ps)

# Temperatura de equilíbrio
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

# Raio de equilíbrio
r_eq = metodo_newton.Newton(zeta, dzeta_dr, r0, 1e-6, 100)

# Tempo de decaimento da exponencial do raio
tau_r = calcular_tau_r(f, M_H2O, sigma_s_val, R, T_a_em_k, rho_w, r_i, rho_s_val, m_s_val, v_ion, Phi_s_val, M_NaCl, D_linha_w_val, e_sat_val, L_v_val, k_linha_a_val, r_eq)[1]


def f_system(t, variaveis_gota):
    m, r, T = variaveis_gota
    
    # dm/dt 
    H_novo = H_estrela.calcular_H_estrela(T, S)
    Dg_novo = Dg_estrela.calcular_Dg_estrela(r, T_a_em_k, R_atm)
    vol = (4 / 3) * np.pi * r**3
    C_gota = m / vol
    dm_dt = 4 * np.pi * r * Dg_novo * (C_ar - C_gota / (H_novo * R_atm * T))
    
    # dr/dt
    rho_ww_novo = m_s.calcular_rho_ww(T-273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T-273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T-273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T-273.15, T, P, r, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo = rho_vr.calcular_rho_vr(M_H2O,e_sat_esp.calcular_esat(T-273.15),code_exp_y.calcular_exp_y(M_H2O , sigma_s_novo , T_a_em_k , rho_w , R, v_ion , Phi_s_novo , m_s_val , r , rho_s_val ,  M_NaCl),R,T_a_em_k) 
    
    Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r)) - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo-m_s_val))
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2
    dr_dt = ((f - 1) - Y) / (r * den)
    
    # dT/dt
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo))
    
    return np.array([dm_dt, dr_dt, dT_dt])


# Só um passo do rk3 (não utilizado diretamente, mas mantido)
def passo_rk3(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + (dt/2) * k1)
    k3 = f(t + dt, y - dt * k1 + 2 * dt * k2)
    return y + (dt/6) * (k1 + 4*k2 + k3)

nomes_variaveis = ["m", "r", "T"]


def rkf45_adaptativo(f, y0, t_final, tol):
    """
    Solver adaptativo Runge-Kutta-Fehlberg 4(5) 
    Baseado na estrutura procedural do solver_rk3_adaptativo.
    Agora retorna também a lista de dt utilizados.
    """
    t = 0
    dt = 1e-6 # Passo inicial
    y = np.array(y0, dtype=float)

    # Listas para armazenar a evolução temporal
    t_lista = [t]
    y_lista = [y]
    dt_lista = []               # <--- NOVO: lista para armazenar os dt usados
    
    # Parâmetros de controle
    p = 4          # Ordem do método base (RK4)
    eta = 0.9      # Fator de segurança
    dt_min = 1e-10
    dt_max = np.inf
    
    # Coeficientes do Butcher Tableau RKF45 
    c2=1/4; a21=1/4
    c3=3/8; a31=3/32; a32=9/32
    c4=12/13; a41=1932/2197; a42=-7200/2197; a43=7296/2197
    c5=1; a51=439/216; a52=-8; a53=3680/513; a54=-845/4104
    c6=1/2; a61=-8/27; a62=2; a63=-3544/2565; a64=1859/4104; a65=-11/40
    
    # Coeficientes para as soluções de ordem 4 (b) e ordem 5 (bh) 
    b1=25/216; b3=1408/2565; b4=2197/4104; b5=-1/5
    bh1=16/135; bh3=6656/12825; bh4=28561/56430; bh5=-9/50; bh6=2/55

    nomes_variaveis = ["m", "r", "T"]

    while t < t_final:
        # Garante que não ultrapasse o tempo final
        if t + dt > t_final:
            dt = t_final - t
        
        # Estágios RKF45 
        k1 = f(t, y)
        k2 = f(t + c2*dt, y + dt*(a21*k1))
        k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
        k4 = f(t + c4*dt, y + dt*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*dt, y + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*dt, y + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        
        # Soluções de baixa (4ª) e alta (5ª) ordem 
        # Nota: Coeficientes b2, b6 e bh2 são zero.
        low_update = dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5)
        high_update = dt * (bh1*k1 + bh3*k3 + bh4*k4 + bh5*k5 + bh6*k6)
        
        y_new = y + high_update # Avançamos com a ordem 5 (Extrapolação Local) 
        
        # Estimativa do erro de truncamento local (diferença entre as ordens) 
        erro_est = np.abs(high_update - low_update)
        
        # Tolerâncias específicas do seu problema
        atol = np.array([1e-18, 1e-8, 1e-4]) 
        rtol = tol
        
        # Erro máximo permitido e normalização
        erro_toleravel = atol + rtol * np.abs(y_new)
        erro_relativo = erro_est / erro_toleravel

        # Lógica de seleção de variável (Arquitetura original do seu código)
        # Nota: Assume que tau_T está disponível no escopo ou definido antes
        if t < tau_T:
            erro_max = erro_relativo[2] # Controlado pela Temperatura
        else:
            erro_max = np.max(erro_relativo[0:2]) # Controlado por Massa e Raio

        # Evita divisão por zero
        erro_max = max(erro_max, 1e-20) 
            
        # Aceitação ou Rejeição do passo
        if erro_max <= 1 or dt <= dt_min:
            t = t + dt
            y = y_new 
            t_lista.append(t)
            y_lista.append(y)
            dt_lista.append(dt)               # <--- NOVO: guarda o dt usado
            
            # Novo dt para o próximo passo (Eq 4.4 do PDF) [cite: 386]
            # Usando p=4 para RKF45
            new_dt = eta * dt * (1.0 / erro_max) ** (1/(p+1))
            dt = np.clip(new_dt, dt_min, dt_max)
        else:
            # Rejeita o passo e reduz dt
            dt = eta * dt * (1.0 / erro_max)**(1/(p+1))
            dt = max(dt, dt_min)

    return np.array(t_lista), np.array(y_lista), np.array(dt_lista)   # <--- NOVO retorno


# Tolerância do método adaptativo
tolerancia = 1e-8
variaveis_iniciais = np.array([1.2e-15, r_i, T_gota_em_k])

# Chamada modificada para receber também dt_lista
tempo_rkf45, variaveis_finais_rkf45, dt_rkf45 = rkf45_adaptativo(f_system, variaveis_iniciais, 1e3, tol=tolerancia)

massa_final_rkf45 = variaveis_finais_rkf45[-1,0]
raio_final_rkf45 = variaveis_finais_rkf45[-1,1]
temperatura_final_rkf45 = variaveis_finais_rkf45[-1,2]
print(f"Raio final           : {raio_final_rkf45}")
print(f"Temperatura final    : {temperatura_final_rkf45}")
print(f"Massa final          : {massa_final_rkf45}")
print(f"Número de passos     : {len(tempo_rkf45)}")

# Gráficos (reativados e adaptados para RKF45)
plt.rcParams['text.usetex'] = False
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# Não deletamos mais o subplot (1,1) – ele será usado para o dt

# Gráfico 1: Massa
axs[0, 0].plot(tempo_rkf45, variaveis_finais_rkf45[:,0], "#2B00FF", label=f'RKF45 - {massa_final_rkf45:.3e} mol')
axs[0, 0].set_xlabel('Tempo (s)'); axs[0, 0].set_ylabel('Massa (mol)')
axs[0, 0].set_xscale('log'); axs[0, 0].legend(); axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].set_title('Evolução da Massa')

# Gráfico 2: Raio
axs[0, 1].plot(tempo_rkf45, variaveis_finais_rkf45[:,1]*1e6, "#FF0000", label=f'RKF45 - {raio_final_rkf45*1e6:.3f} μm')
axs[0, 1].set_xlabel('Tempo (s)'); axs[0, 1].set_ylabel('Raio (μm)')
axs[0, 1].set_xscale('log'); axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].set_title('Evolução do Raio')

# Gráfico 3: Temperatura
axs[1, 0].plot(tempo_rkf45, variaveis_finais_rkf45[:,2]-273.15, "#00AA00", label=f'RKF45 - {temperatura_final_rkf45-273.15:.2f} °C')
axs[1, 0].set_xlabel('Tempo (s)'); axs[1, 0].set_ylabel('Temperatura (°C)')
axs[1, 0].set_xscale('log'); axs[1, 0].legend(); axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].set_title('Evolução da Temperatura')
# Linha vertical em tau_T (opcional)
#axs[1, 0].axvline(tau_T, color='black', linestyle='--', linewidth=2, label=r'$\tau_T$')
axs[1, 0].legend(fontsize=10)

# --- NOVO GRÁFICO: Evolução do passo dt ---
# dt_rkf45 tem comprimento = len(tempo_rkf45)-1, pois cada dt corresponde a um passo.
# Plotamos dt versus o tempo no final do respectivo passo (tempo_rkf45[1:]).
axs[1, 1].loglog(tempo_rkf45[1:], dt_rkf45, 'o-', color='purple', markersize=3, linewidth=1, label='dt usado')
axs[1, 1].set_xlabel('Tempo (s)')
axs[1, 1].set_ylabel('Passo de tempo dt (s)')
axs[1, 1].set_title('Evolução do passo adaptativo (RKF45)')
axs[1, 1].grid(True, alpha=0.3, which='both')
axs[1, 1].legend()

plt.suptitle(f'RKF45 Adaptativo\nTol = {tolerancia}', fontsize=16)
plt.tight_layout()
plt.show()