import time


inicio = time.perf_counter()

def calcula_soma():
    print('Iniciando a função...')
    soma = 0
    for i in range(50_000_000) :
        soma = soma + 1
    print('Calculo finalizado')

calcula_soma()
calcula_soma()

fim = time.perf_counter()
total = round(fim - inicio, 2)
print(f'Tempo de execução: {total}')
