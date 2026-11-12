def calcular_phi_s(m_s, M_NaCl, m_w):
    m = m_s / (M_NaCl * m_w)
    phi_s = (0.9270 - 2.164e-2 * m + 3.486e-2 * m**2 - 
             5.956e-3 * m**3 + 3.911e-4 * m**4)
    return phi_s
