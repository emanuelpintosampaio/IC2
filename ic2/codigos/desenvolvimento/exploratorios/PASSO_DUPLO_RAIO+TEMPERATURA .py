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
C_ar = 0.0154      # Concentração do gás carbônico no ar (não usada)
S = 34             # Salinidade do mar
R_atm = 0.082      # Constante universal dos gases (não usada)
g = 9.81           # Gravidade
v_ar = 1.32e-5     # Viscosidade cinemática do ar
rho_ar = 1.225     # Densidade do ar
H_s = 6            # Altura significativa da onda 
T0 = 273.15
P0 = 1013.25
c_ps = 4000        # Calor específico da spray

e_sat_val = e_sat.calcular_esat(T_a)
rho_ww = m_s.calcular_rho_ww(T_gota)
m_s_val = m_s.calcular_ms(rho_ww, r_i, s)
v_a = rho_s.calcular_v_a(T_gota, m_s_val, M_NaCl, r_i)
m_w = rho_s.calcular_massa_agua(r_i, rho_ww)
rho_s_val = rho_s.calcular_rho_spray(rho_ww, m_s_val, m_w, v_a, M_NaCl)
rho_v_val = rho_v.calcular_rho_v(f, M_H2O, e_sat_val, R, T_a_em_k) 

def f_r_T(t, vars):
    r, T = vars  # vars é um vetor com duas componentes
    
    # dr/dt
    rho_ww_novo = m_s.calcular_rho_ww(T-273.15)
    m_w_novo = rho_s.calcular_massa_agua(r, rho_ww_novo)
    sigma_s_novo = sigma_s.calculate_sigma_s(T-273.15, m_s_val, m_w_novo)
    Phi_s_novo = Phi_s.calcular_phi_s(m_s_val, M_NaCl, m_w_novo)
    D_linha_w_novo = D_linha_w.calculate_Dw_prime(T, R, P, r, M_H2O, T0, P0, alpha_c=0.036, Delta_w=8e-8)
    L_v_novo = L_v.calcular_lv(T-273.15)
    k_linha_a_novo = k_linha_a.calculate_K_linha_a(T-273.15, T, P, r, R, T0, P0, M_a=28.9644e-3, alpha_T=0.7, delta_T=2.16e-7, c_pa=1.006e3)
    rho_vr_novo = rho_vr.calcular_rho_vr(M_H2O,
                                         e_sat_esp.calcular_esat(T-273.15),
                                         code_exp_y.calcular_exp_y(M_H2O, sigma_s_novo, T_a_em_k, rho_w, R,
                                                                    v_ion, Phi_s_novo, m_s_val, r, rho_s_val, M_NaCl),
                                         R, T_a_em_k)
    
    Y = (2 * M_H2O * sigma_s_novo / (R * T_a_em_k * rho_w * r)) - (v_ion * Phi_s_novo * m_s_val * (M_H2O / M_NaCl) / (m_w_novo - m_s_val))
    den1 = rho_s_val * R * T_a_em_k / (D_linha_w_novo * M_H2O * e_sat_val)
    den2 = rho_s_val * L_v_novo / (k_linha_a_novo * T_a_em_k) * (L_v_novo * M_H2O / (R * T_a_em_k) - 1)
    den = den1 + den2
    dr_dt = ((f - 1) - Y) / (r * den)
    
    # dT/dt
    dT_dt = 3 / (rho_s_val * c_ps * r**2) * (k_linha_a_novo * (T_a_em_k - T) + L_v_novo * D_linha_w_novo * (rho_v_val - rho_vr_novo))
    
    return np.array([dr_dt, dT_dt])

# Passo RK3
def passo_rk3(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + (dt/2) * k1)
    k3 = f(t + dt, y - dt * k1 + 2 * dt * k2)
    return y + (dt/6) * (k1 + 4*k2 + k3)

def solver_adaptativo(f_r_T, y0, t_final, tol):
    t = 0

    # Condições iniciais
    y = np.array(y0)    

    # Passo inicial                
    dt = 1e-6

    # Listas para armazenar a evolução temporal
    t_lista = [t]
    y_lista = [y.copy()]        
    dt_lista = []
    
    # Ordem global do rk3 
    p = 3
    
    # Eta < 1       
    eta = 0.9

    dt_min = 1e-13
    dt_max = np.inf

    while t < t_final:
        # Garante que não ultrapasse o tempo final
        if t + dt > t_final:
            dt = t_final - t

        # Solução com um passo dt
        y1 = passo_rk3(f_r_T, t, y, dt)
        #print(y1[0])
        #print(y1[1])

        # Solução com dois passos dt/2
        y_temp = passo_rk3(f_r_T, t, y, dt/2)
        y2 = passo_rk3(f_r_T, t + dt/2, y_temp, dt/2)
        #print(y2[0])
        #print(y2[1])

        # Estimativa do erro local
        diff = y2 - y1
        #print(diff[0])
        #print(diff[1])

        # Livro Ascher_Petzold_Computer_Methods_For_Ordi.pdf página 110 do pdf
        erro_est = np.abs(diff) / (2**p - 1) 

        erro_rel_r = erro_est[0] / (np.abs(y2[0]+ 1e-30))
        #print(f'erro relativo raio = {erro_rel_r}')
        erro_rel_T = erro_est[1] / (np.abs(y2[1]+ 1e-30))
        #print(f'erro relativo temperatura = {erro_rel_T}')

        erro_rel = erro_rel_T

        if erro_rel < tol or dt < dt_min:
            # Atualiza tempo
            t = t + dt
            
            # Usa a solução mais precisa (dois subpassos)
            y = y2

            # Vai adicionando os valores atuais nas listas 
            dt_lista.append(dt)
            t_lista.append(t)
            y_lista.append(y.copy())

            # Novo dt para o próximo passo Eq 4.4 
            new_dt = eta * dt * (tol / erro_rel+ 1e-30)**(1/(p+1))            

            # Limita crescimento e decrescimento do passo
            new_dt = min(new_dt, dt_max)
            new_dt = max(new_dt, dt_min)
            dt = new_dt
        else:
            # Rejeita o passo e reduz dt
            dt = eta * dt * (tol / erro_rel+ 1e-30) ** (1/(p+1))
            dt = max(dt, dt_min)

    return np.array(t_lista), np.array(y_lista), np.array(dt_lista)

# Tolerância do método adaptativo
tolerancia = 1e-8

# Condições Iniciais: apenas raio e temperatura
y0_completo = np.array([r_i, T_gota_em_k])

# Resolve a EDO
t_comp, y_comp, dt_comp = solver_adaptativo(f_r_T, y0_completo, 1e3, tolerancia)
T_comp = y_comp[:, 1]
r_comp = y_comp[:, 0]

print(f"Temperatura final    : {T_comp[-1]} K")
print(f"Raio final           : {r_comp[-1]*1e6} µm")
print(f"Número de passos     : {len(t_comp)}")

"""
# Gráficos
plt.rcParams['text.usetex'] = False
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Temperatura vs tempo (escala log em x)
ax1.semilogx(t_comp, T_comp - T0, 'o-',linewidth=2, label=f'Temperatura - {T_comp[-1]-273.15} °C')
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Temperatura da gota (°C)')
ax1.set_title('Evolução da temperatura')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: Raio vs tempo (escala log em x)
ax2.semilogx(t_comp, r_comp * 1e6, 'o-',linewidth=2, label=f'raio - {r_comp[-1]*1e6} μm')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Raio (µm)')
ax2.set_title('Evolução do raio')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Gráfico 3: Passo dt vs tempo (log-log)
ax3.loglog(t_comp[1:], dt_comp, 'o-', markersize=3, linewidth=2)
ax3.set_xlabel('Tempo (s)')
ax3.set_ylabel('Passo de tempo dt (s)')
ax3.set_title('Evolução do passo adaptativo para temperatura')
ax3.grid(True, alpha=0.3, which='both')
fig.suptitle(f'Passo Duplo - {len(t_comp)} passos')

plt.tight_layout()
plt.show()
"""