def calculate_sigma_s2(T_eq, m_s2,m_ww2):
    sigma_w = 7.610e-2 - 1.55e-4 * T_eq
    sigma_ss = sigma_w + 2.77e-5 * m_s2 / m_ww2 
    return sigma_ss
