"""
Implementação do método Runge-Kutta de 4ª ordem com controle adaptativo de passo
usando a técnica de step doubling (passo duplo).

Baseado em: Numerical Recipes, Chapter 17.2 - Adaptive Stepsize Control for Runge-Kutta
"""

import numpy as np


def rk4_step(f, t, y, h):
    """
    Executa um único passo do método RK4 clássico.
    
    Parâmetros:
    -----------
    f : callable
        Função que define o sistema de EDOs: dy/dt = f(t, y)
    t : float
        Tempo atual
    y : ndarray
        Vetor de estado atual
    h : float
        Tamanho do passo
        
    Retorna:
    --------
    y_new : ndarray
        Novo vetor de estado após o passo h
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    
    y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return y_new


def rk4_step_adaptativo(f, t, y, h, tol=1e-6):
    """
    Executa um passo RK4 com controle adaptativo usando step doubling.
    
    A técnica consiste em:
    1. Fazer um passo completo de tamanho h
    2. Fazer dois passos de tamanho h/2
    3. Comparar os resultados para estimar o erro
    4. Ajustar o passo se necessário
    
    Parâmetros:
    -----------
    f : callable
        Função que define o sistema de EDOs: dy/dt = f(t, y)
    t : float
        Tempo atual
    y : ndarray
        Vetor de estado atual
    h : float
        Tamanho inicial do passo
    tol : float
        Tolerância desejada para o erro de truncamento
        
    Retorna:
    --------
    y_new : ndarray
        Novo vetor de estado
    t_new : float
        Novo tempo
    h_new : float
        Tamanho do passo para a próxima iteração
    erro : float
        Estimativa do erro de truncamento
    """
    # Fator de segurança para ajuste do passo
    SAFETY = 0.9
    # Fator de crescimento máximo do passo
    PGROW = -0.2
    # Fator de redução máximo do passo
    PSHRINK = -0.25
    # Erro máximo permitido (múltiplo da tolerância)
    ERRCON = 1.89e-4
    
    # Loop até encontrar um passo aceitável
    while True:
        # Passo 1: Um passo completo de tamanho h
        y1 = rk4_step(f, t, y, h)
        
        # Passo 2: Dois passos de tamanho h/2
        y_meio = rk4_step(f, t, y, h/2)
        y2 = rk4_step(f, t + h/2, y_meio, h/2)
        
        # Passo 3: Estimar o erro
        # Para RK4, o erro local de truncamento é O(h^5)
        # A diferença entre y1 e y2 é aproximadamente 15*erro_local
        # (porque erro com h é 2^5 vezes maior que erro com h/2)
        Delta = y2 - y1
        erro = np.max(np.abs(Delta)) / 15.0
        
        # Calcular escala para determinar erro relativo
        y_escala = np.maximum(np.abs(y), np.abs(y2))
        y_escala = np.maximum(y_escala, 1e-30)  # Evitar divisão por zero
        
        # Erro relativo
        erro_rel = np.max(np.abs(Delta) / y_escala) / 15.0
        
        # Passo 4: Decidir se aceita ou rejeita o passo
        if erro_rel <= tol:
            # Passo aceito!
            # Usar a solução mais precisa (y2) com correção de Richardson
            y_new = y2 + Delta / 15.0
            t_new = t + h
            
            # Calcular novo tamanho de passo
            if erro_rel > ERRCON:
                h_new = SAFETY * h * (tol / erro_rel)**PGROW
            else:
                # Se erro muito pequeno, pode crescer o passo significativamente
                h_new = 5.0 * h
                
            return y_new, t_new, h_new, erro_rel
        else:
            # Passo rejeitado - reduzir tamanho do passo
            h_temp = SAFETY * h * (tol / erro_rel)**PSHRINK
            # Não reduzir mais que um fator de 10
            h = max(h_temp, 0.1 * h)
            
            # Verificar se passo não ficou muito pequeno
            if abs(h) < 1e-20:
                raise RuntimeError("Passo muito pequeno no RK4 adaptativo. Possível stiffness no sistema.")


def integrar_rk4_adaptativo(f, y0, t_span, h_inicial=0.01, tol=1e-6, 
                             h_min=1e-10, h_max=1.0, max_passos=1000000):
    """
    Integra um sistema de EDOs usando RK4 com passo adaptativo.
    
    Parâmetros:
    -----------
    f : callable
        Função que define o sistema de EDOs: dy/dt = f(t, y)
    y0 : array_like
        Condições iniciais
    t_span : tuple
        Tupla (t_inicial, t_final) com o intervalo de integração
    h_inicial : float
        Tamanho inicial do passo
    tol : float
        Tolerância para o erro
    h_min : float
        Tamanho mínimo permitido para o passo
    h_max : float
        Tamanho máximo permitido para o passo
    max_passos : int
        Número máximo de passos permitidos
        
    Retorna:
    --------
    dict com:
        't' : ndarray - Vetor de tempos
        'y' : ndarray - Matriz com as soluções (cada linha é um tempo)
        'h_history' : ndarray - Histórico dos tamanhos de passo
        'erro_history' : ndarray - Histórico dos erros
        'n_passos' : int - Número de passos realizados
        'n_rejeicoes' : int - Número de passos rejeitados
    """
    t_inicial, t_final = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Listas para armazenar resultados
    t_list = [t_inicial]
    y_list = [y0.copy()]
    h_history = []
    erro_history = []
    
    # Inicialização
    t = t_inicial
    y = y0.copy()
    h = h_inicial
    n_passos = 0
    n_rejeicoes = 0
    
    print(f"\nIniciando integração RK4 adaptativo:")
    print(f"  t_inicial = {t_inicial:.3e}, t_final = {t_final:.3e}")
    print(f"  h_inicial = {h_inicial:.3e}, tolerância = {tol:.3e}")
    print(f"  Dimensão do sistema: {len(y0)}")
    print("-" * 60)
    
    # Loop de integração
    while t < t_final and n_passos < max_passos:
        # Ajustar último passo para não ultrapassar t_final
        if t + h > t_final:
            h = t_final - t
        
        # Garantir que h está dentro dos limites
        h = np.clip(h, h_min, h_max)
        
        # Tentar dar um passo
        try:
            y_new, t_new, h_new, erro = rk4_step_adaptativo(f, t, y, h, tol)
            
            # Passo aceito
            t = t_new
            y = y_new
            
            # Armazenar resultados
            t_list.append(t)
            y_list.append(y.copy())
            h_history.append(h)
            erro_history.append(erro)
            
            # Atualizar passo para próxima iteração
            h = h_new
            n_passos += 1
            
            # Imprimir progresso
            if n_passos % 1000 == 0:
                progresso = (t - t_inicial) / (t_final - t_inicial) * 100
                print(f"  Passo {n_passos}: t = {t:.3e} ({progresso:.1f}%), h = {h:.3e}, erro = {erro:.3e}")
                
        except RuntimeError as e:
            print(f"\nErro na integração: {e}")
            break
    
    print("-" * 60)
    print(f"Integração concluída:")
    print(f"  Número de passos: {n_passos}")
    print(f"  Tempo final: {t:.6e}")
    print(f"  Passo médio: {np.mean(h_history):.3e}")
    print(f"  Passo mínimo: {np.min(h_history):.3e}")
    print(f"  Passo máximo: {np.max(h_history):.3e}")
    print(f"  Erro médio: {np.mean(erro_history):.3e}")
    print(f"  Erro máximo: {np.max(erro_history):.3e}")
    
    # Converter listas para arrays
    t_array = np.array(t_list)
    y_array = np.array(y_list)
    h_array = np.array(h_history)
    erro_array = np.array(erro_history)
    
    return {
        't': t_array,
        'y': y_array,
        'h_history': h_array,
        'erro_history': erro_array,
        'n_passos': n_passos,
        'n_rejeicoes': n_rejeicoes
    }


def interpolate_solution(resultado, t_desejados):
    """
    Interpola a solução do RK4 adaptativo para tempos desejados.
    
    Parâmetros:
    -----------
    resultado : dict
        Dicionário retornado por integrar_rk4_adaptativo
    t_desejados : array_like
        Tempos onde se deseja a solução
        
    Retorna:
    --------
    y_interp : ndarray
        Soluções interpoladas nos tempos desejados
    """
    from scipy.interpolate import interp1d
    
    t = resultado['t']
    y = resultado['y']
    
    # Criar interpolador para cada variável
    n_vars = y.shape[1]
    y_interp = np.zeros((len(t_desejados), n_vars))
    
    for i in range(n_vars):
        interpolador = interp1d(t, y[:, i], kind='cubic', 
                               fill_value='extrapolate')
        y_interp[:, i] = interpolador(t_desejados)
    
    return y_interp
