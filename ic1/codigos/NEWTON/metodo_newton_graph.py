def Newton(f, df, x0, epsilon, max_iter):
    """
    Método de Newton que retorna (raiz, lista_de_iteracoes).
    Compatível com graph_req.py.
    """
    if max_iter <= 0:
        raise ValueError("Número de iterações deve ser maior que zero.")

    x = x0
    iteracoes = [x0]

    for k in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < epsilon:
            return x, iteracoes

        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivada zero em x = %.8e" % x)

        x_new = x - fx / dfx
        iteracoes.append(x_new)

        if abs(x_new - x) < epsilon:
            return x_new, iteracoes

        x = x_new

    raise ValueError(
        "Não convergiu após %d iterações (último x: %.8e)" % (max_iter, x)
    )
