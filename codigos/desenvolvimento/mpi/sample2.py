import time
import multiprocessing


inicio = time.perf_counter()

def calcula_soma():
    print('Iniciando a função...')
    soma = 0
    for i in range(50_000_000) :
        soma = soma + 1
    print('Calculo finalizado')

code1 = multiprocessing.Process(target=calcula_soma)
code2 = multiprocessing.Process(target=calcula_soma)

code1.start()
code2.start()

code1.join()
code2.join()

fim = time.perf_counter()
total = round(fim - inicio, 2)
print(f'Tempo de execução: {total}')
