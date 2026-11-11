# IC2: Aplicação do método de Runge-Kutta com controle por subpassos em uma equação de difusão de massa em superfície esférica

Edital Piic 2025/2026 · UFES · Orientador: Iury Angelo Gonçalves · Estudante: Emanuel Pinto Sampaio

## Modelo

Sistema completo de EDOs acopladas para o raio r(t), a temperatura T(t) e a massa de CO₂ m(t) de uma gotícula de espuma marinha, resolvido pelo método de Runge-Kutta de 3ª ordem **SSPRK3**. Como τ_T ≪ τ_r, a temperatura é o subsistema rápido e o raio e a massa formam o subsistema lento. No **controle por subpassos**, a cada passo macroscópico Δt do subsistema lento, T(t) avança M subpassos de δt = Δt/M com r(t) e m(t) congelados. O modelo completo é comparado com o modelo simplificado da IC1 (r e T por decaimento exponencial, só m integrada).

| Experimento | Modelo | Passo de tempo | Script (`codigos/`) |
|---|---|---|---|
| ECONTROL | completo (r, T, m) | fixo, Δt = 10⁻⁴ s | `IC2_completo_RK3_dt_fixo_{30,500}.py` |
| E1 | completo (r, T, m) | subpassos: Δt = 10⁻³ s, M = 10 | `IC2_completo_RK3_subpassos_{30,500}.py` |
| E2 | simplificado (só m) | fixo, Δt = 10⁻⁴ s | `IC2_simplificacao_RK3_dt_fixo_{30,500}.py` |
| E3 | simplificado (só m) | subpassos | `IC2_simplificacao_RK3_subpassos_{30,500}.py` |
| comparação | os quatro juntos | | `IC2_RK3_all_{30,500}.py` |
| MPI | E1 com 2 processos (T em um, r e m no outro) | subpassos | `IC2_completo_RK3_MPI_subpassos.py` |

O sufixo `_30` ou `_500` é o raio inicial em µm. Os arquivos `*_passo_adaptativo_PID_*` são uma terceira estratégia (passo adaptativo com controle PID), testada mas fora da tabela do relatório.

### Conferência com o relatório

Rodando os scripts `_500` desta pasta, as massas finais de CO₂ são idênticas às mostradas no relatório:

| Experimento | Massa final (mol), r_i = 500 µm |
|---|---|
| ECONTROL | 7.957903319952931e-12 |
| E1 | 7.957844232570539e-12 |
| E2 | 7.953202982460029e-12 |
| E3 | 7.95314107953467e-12 |

## Pastas

- `relatorio/`: relatório final em PDF.
- `figuras_relatorio/`: figuras e tabelas do relatório.
  - `30_Tabela1.png`, `500_Tabela1.png`: tempos de execução dos experimentos (Tabela 1).
  - `Tabela2.png`, `teste_mpi.png`: comparação entre a versão sequencial e a versão com MPI.
  - `30_simu.png`, `500_simu.png`: modelo completo × aproximações exponenciais.
  - `econtrolprova.png`: instabilidade numérica (overflow) do sistema completo com passo grande.
  - `rk3.pdf`: representação geométrica do método RK3.
- `codigos/`: os scripts da tabela acima e os módulos físicos (`NEWTON/`, `T_EQ/`, `TAU_R/`, `TAU_T/`, `DGH/`, `CONSTANTES`). Os gráficos são gravados como `grafico_*.png` na pasta de onde o script é executado.
- `codigos/desenvolvimento/`: versões feitas durante o desenvolvimento.
  - `IC2_RK3_all.py`, `E1_MPI.py`, `IC2_simplificacao_RK3_passo_adaptativo_PID.py`: versões editadas depois do relatório (ver Observações).
  - `subsistemas/`: só r, só T, ou r e T, com Euler e RK3, em passo fixo, subpassos e passo adaptativo.
  - `mpi/`: outras versões com MPI e testes de paralelismo (`sample1-3.py` usam `multiprocessing`; `a.py` roda vários raios em sequência).
  - `versoes_antigas/`: versões antigas dos scripts de subsistemas (arquivos que começam com `- `).
  - `exploratorios/`: o caminho até o RK3 com subpassos (passo duplo, RKF45, sistemas com Euler, RK3 e RK4).
  - `resultados/`: figuras e o CSV gerados durante o desenvolvimento.

## Como rodar no computador

```bash
pip install numpy matplotlib
cd ic2/codigos
python IC2_completo_RK3_subpassos_500.py              # E1 com r_i = 500 µm
python IC2_RK3_all_30.py                              # os quatro experimentos com r_i = 30 µm

# MPI (precisa de mpi4py e de OpenMPI ou MPICH); exige exatamente 2 processos
mpiexec -n 2 python IC2_completo_RK3_MPI_subpassos.py

# scripts de desenvolvimento/: rodar de dentro de codigos/
PYTHONPATH=. python desenvolvimento/IC2_RK3_all.py
```

## Menu das simulações no Google Colab

Você pode rodar este projeto diretamente no Google Colab, sem instalar nada localmente.

### Passos:

1° Acesse: [https://colab.research.google.com/](https://colab.research.google.com/)

2° Clique em **"Novo notebook"**

3° Clique em **"Conectar"**

4° Cole o código abaixo em uma célula e execute:

```python
import os
import time
from IPython.display import Image, display, clear_output

%cd /content
if not os.path.exists('IC'):
    !git clone https://github.com/emanuelpintosampaio/IC.git
%cd IC/ic2/codigos
!pip install -q numpy matplotlib

SIMULACOES = {
    "ECONTROL": {
        "descricao": "EDO completo | passo fixo",
        "script": "IC2_completo_RK3_dt_fixo_{raio}.py",
        "grafico": "grafico_dt_fixo_completo_{raio}.png",
    },
    "E1": {
        "descricao": "EDO completo | controle por subpassos",
        "script": "IC2_completo_RK3_subpassos_{raio}.py",
        "grafico": "grafico_subpassos_completo_{raio}.png",
    },
    "E2": {
        "descricao": "EDO simples | passo fixo",
        "script":    "IC2_simplificacao_RK3_dt_fixo_{raio}.py",
        "grafico":   "grafico_dt_fixo_simplificacao_{raio}.png",
    },
    "E3": {
        "descricao": "EDO simples | controle por subpassos",
        "script": "IC2_simplificacao_RK3_subpassos_{raio}.py",
        "grafico": "grafico_subpassos_simplificacao_{raio}.png",
    }}

OPCOES_RAIO = {"1": "30", "2": "500"}

while True:
    # Seleção de raio
    print("\nESCOLHA O RAIO DA GOTA PARA A SIMULAÇÃO")
    for k, v in OPCOES_RAIO.items():
        print(f"{k} - {v} µm{' ' * (41 - len(v))}")

    opcao_raio = input("Opção: ").strip()
    if opcao_raio not in OPCOES_RAIO:
        print("Opção de raio inválida! Tente novamente.")
        continue
    raio = OPCOES_RAIO[opcao_raio]
    time.sleep(0.5)

    # Seleção da simulação 
    print("\nESCOLHA O TIPO DE SIMULAÇÃO")
    opcoes_sim = list(SIMULACOES.keys())
    for i, nome in enumerate(opcoes_sim, 1):
        desc = SIMULACOES[nome]["descricao"]
        linha = f"{i} - {nome}: {desc}"
        print(f"{linha:<58}")
    print(f"{len(opcoes_sim)+1} - Comparação: todas as simulações{' ' * 25}")

    opcao_sim = input("Opção: ").strip()

    graficos_para_exibir = []

    if opcao_sim.isdigit() and 1 <= int(opcao_sim) <= len(opcoes_sim):
        nome_sim = opcoes_sim[int(opcao_sim) - 1]
        sim = SIMULACOES[nome_sim]
        script  = sim["script"].format(raio=raio)
        grafico = sim["grafico"].format(raio=raio)

        print(f"\nExecutando Simulação {nome_sim} — {sim['descricao']} | Raio: {raio} µm\n")
        os.system(f"python {script}")
        graficos_para_exibir.append(grafico)

    elif opcao_sim == str(len(opcoes_sim) + 1):
        print(f"\nExecutando comparação completa | Raio: {raio} µm\n")
        os.system(f"python IC2_RK3_all_{raio}.py")
        graficos_para_exibir = [sim["grafico"].format(raio=raio) for sim in SIMULACOES.values()] + [f"grafico_all_{raio}.png"]

    else:
        print("Opção inválida! Tente novamente.")
        continue

    # Gráficos 
    print("\nBUSCANDO GRÁFICOS GERADOS...")
    exibidos = 0
    for arq in graficos_para_exibir:
        if os.path.exists(arq):
            print(f"Exibindo: {arq}")
            display(Image(arq))
            exibidos += 1
    if exibidos == 0:
        print("Nenhum gráfico encontrado para exibir.")
    else:
        time.sleep(3)

    # Continuar? 
    if input("\nDeseja fazer outra simulação? (s/n): ").strip().lower() != 's':
        print("\nEncerrando o programa...")
        break
    
```


## Observações

- As versões em `desenvolvimento/` (`IC2_RK3_all.py`, `E1_MPI.py`) foram editadas depois do relatório: `IC2_RK3_all.py`, por exemplo, usa `M_H2O` no lugar de `M_NaCl` no cálculo de Φ_s e chega a massas finais um pouco diferentes (7.957652476067248e-12 no ECONTROL a 500 µm). Os números do relatório são os dos scripts da raiz de `codigos/`.
- Vários scripts de `desenvolvimento/` importam `from GRAFICO.DGH import ...`; a pasta `codigos/GRAFICO/DGH/` é uma cópia de `DGH/` para que eles funcionem sem alteração.
