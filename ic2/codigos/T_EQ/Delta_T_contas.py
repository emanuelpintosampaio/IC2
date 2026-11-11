import math

def calculate_delta_T(T_a_em_k, alpha, beta, b, exp_y, f):
    numerador_2T_a_b = 2 * T_a_em_k + b - 273.15
    denominador_T_a_b = T_a_em_k + b - 273.15
    termo2_A = alpha * (numerador_2T_a_b / denominador_T_a_b)
    termo1_A = alpha**2 / 2
    dentro_A = termo1_A - termo2_A + 1
    A = (beta / (T_a_em_k**2)) * dentro_A * exp_y
    B = 1 + (beta / T_a_em_k) * (alpha - 1) * exp_y
    C = -beta * (f - exp_y)
    discriminante = B**2 - 4 * A * C
    if discriminante < 0:
        return None
    else:
        delta_T1 = (-B + math.sqrt(discriminante)) / (2 * A)
        delta_T2 = (-B - math.sqrt(discriminante)) / (2 * A)
        return (delta_T1, delta_T2)

