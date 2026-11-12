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


# Só um passo do rk3
def passo_rk3(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + (dt/2) * k1)
    k3 = f(t + dt, y - dt * k1 + 2 * dt * k2)
    return y + (dt/6) * (k1 + 4*k2 + k3)

nomes_variaveis = ["Massa (m)", "Raio (r)", "Temperatura (T)"]

# rk3 com passo duplo – AGORA RETORNA TAMBÉM A LISTA DE dt
def solver_rk3_adaptativo(f_system, y0, t_final, tol):
    t = 0
    
    # Passo inicial 
    dt = 1e-6

    # Condições iniciais
    y = y0

    # Listas para armazenar a evolução temporal
    t_lista = [t]
    y_lista = [y]
    dt_lista = []               
    
    # Ordem global do rk3 
    p = 3
    
    # Eta < 1       
    eta = 0.9

    dt_min = 1e-10
    dt_max = np.inf
    
    
    while t < t_final:
        # Garante que não ultrapasse o tempo final
        if t + dt > t_final:
            dt = t_final - t
        
        # Solução com um único passo inteiro (dt)
        y_1_passo = passo_rk3(f_system, t, y, dt)
        
        # Solução com dois subpassos (dt/2)
        y_temp = passo_rk3(f_system, t, y, dt/2)
        y_2_passos = passo_rk3(f_system, t + dt/2, y_temp, dt/2)
        
        # Estimativa do erro local 
        diff = y_2_passos - y_1_passo
        # Livro Ascher_Petzold_Computer_Methods_For_Ordi.pdf página 110 do pdf
        erro_est = np.abs(diff) / ((2**p) - 1)  

        # Controle do passo
        erro_relativo = erro_est / np.abs(y_2_passos)
        # Pegamos o maior erro relativo entre m, r e T
        erro_max = np.max(erro_relativo)
        
        # Aceita o passo se erro estiver abaixo da tolerância
        # ou se já estivermos no limite mínimo de passo
        if erro_max < tol or dt < dt_min:
            # Atualiza tempo
            t = t + dt
            
            # Usa a solução mais precisa (dois subpassos)
            y = y_2_passos 
            
            # Guarda o valor do passo que foi usado (dt)
            dt_lista.append(dt)               
            
            # Vai adicionando os valores atuais nas listas 
            t_lista.append(t)
            y_lista.append(y)
            
            # Novo dt para o próximo passo Eq 4.4 
            new_dt = eta * dt * (tol / erro_max)**(1/(p+1))            

            # Limita crescimento e decrescimento do passo
            new_dt = min(new_dt, dt_max)
            new_dt = max(new_dt, dt_min) 

            dt = new_dt        

        # Rejeita o passo 
        else:
            # Se erro for grande, reduz o passo
            dt = eta * dt * (tol / erro_max)**(1/(p+1))
            dt = max(dt, dt_min)

    return np.array(t_lista), np.array(y_lista), np.array(dt_lista)   # <--- NOVO retorno


variaveis_iniciais = np.array([1.2e-15, r_i, T_gota_em_k])

# Tolerância do método adaptativo
tolerancia = 1e-8

# Chamada modificada para receber também dt_lista
tempo_rk3, variaveis_finais_rk3, dt_rk3 = solver_rk3_adaptativo(f_system, variaveis_iniciais, 1e3, tol=tolerancia)

massa_final_rk3 = variaveis_finais_rk3[-1,0]
raio_final_rk3 = variaveis_finais_rk3[-1,1]
temperatura_final_rk3 = variaveis_finais_rk3[-1,2]
print(f"Raio final           : {raio_final_rk3}")
print(f"Temperatura final    : {temperatura_final_rk3}")
print(f"Massa final          : {massa_final_rk3}")
print(f"Número de passos     : {len(tempo_rk3)}")

"""
# Gráficos
plt.rcParams['text.usetex'] = False
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# NÃO deletamos mais o subplot (1,1) – ele será usado para o dt

# Gráfico 1: Massa
axs[0, 0].plot(tempo_rk3, variaveis_finais_rk3[:,0], "#2B00FF", label=f'RK3 Adapt - {massa_final_rk3:.3e} mol')
axs[0, 0].set_xlabel('Tempo (s)'); axs[0, 0].set_ylabel('Massa (mol)')
axs[0, 0].set_xscale('log'); axs[0, 0].legend(); axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].set_title('Evolução da Massa')

# Gráfico 2: Raio
axs[0, 1].plot(tempo_rk3, variaveis_finais_rk3[:,1]*1e6, "#FF0000", label=f'RK3 Adapt - {raio_final_rk3*1e6:.3f} μm')
axs[0, 1].set_xlabel('Tempo (s)'); axs[0, 1].set_ylabel('Raio (μm)')
axs[0, 1].set_xscale('log'); axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].set_title('Evolução do Raio')

# Gráfico 3: Temperatura
axs[1, 0].plot(tempo_rk3, variaveis_finais_rk3[:,2]-273.15, "#00AA00", label=f'RK3 Adapt - {temperatura_final_rk3-273.15:.2f} °C')
axs[1, 0].set_xlabel('Tempo (s)'); axs[1, 0].set_ylabel('Temperatura (°C)')
axs[1, 0].set_xscale('log'); axs[1, 0].legend(); axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].set_title('Evolução da Temperatura')
# Linha vertical em tau_T
#axs[1, 0].axvline(tau_T, color='black', linestyle='--', linewidth=2,label=r'$\tau_T$')
axs[1, 0].legend(fontsize=10)

# --- NOVO GRÁFICO: Evolução do passo dt ---
axs[1, 1].loglog(tempo_rk3[1:], dt_rk3, 'o-', color='purple', markersize=3, linewidth=1, label='dt usado')
axs[1, 1].set_xlabel('Tempo (s)')
axs[1, 1].set_ylabel('Passo de tempo dt (s)')
axs[1, 1].set_title('Evolução do passo adaptativo')
axs[1, 1].grid(True, alpha=0.3, which='both')
axs[1, 1].legend()

plt.suptitle(f'RK3 Adaptativo (Passo Duplo)\nTol = {tolerancia}', fontsize=16)
plt.tight_layout()
plt.show()
"""
