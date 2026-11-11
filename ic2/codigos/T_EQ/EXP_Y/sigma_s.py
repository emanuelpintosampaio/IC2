
def calculate_sigma_s(T_gota, m_s,m_w):
    sigma_w = 7.610e-2 - 1.55e-4 * T_gota
    sigma_ss = sigma_w + 2.77e-5 * m_s / m_w 
    return sigma_ss
