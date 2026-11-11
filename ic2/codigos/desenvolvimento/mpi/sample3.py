import concurrent.futures
import time

inicio = time.perf_counter()

def calcula_soma(numero):
    print('Iniciando a função...')
    soma = 0
    for i in range(numero) :
        soma = soma + 1
    return 'calculo finalizado'

with concurrent.futures.ProcessPoolExecutor() as executer:
    cod1 = executer.submit(calcula_soma,500_000_000)
    cod2 = executer.submit(calcula_soma,500_000_000)

    print(cod1.result())
    print(cod2.result())

    fim = time.perf_counter()
    total = round(fim - inicio, 2)
    print(f'Tempo de execução: {total}')
