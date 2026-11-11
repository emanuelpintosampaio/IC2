def calcular_phi_s2(m_s2, M_NaCl, m_ww2):
    m2 = m_s2 / (M_NaCl * m_ww2)
    phi_s2 = (0.9270 - 2.164e-2 * m2 + 3.486e-2 * m2**2 - 
             5.956e-3 * m2**3 + 3.911e-4 * m2**4)
    return phi_s2
