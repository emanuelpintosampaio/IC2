# Iniciação Científica: difusão de CO₂ em gotículas de espuma marinha

Códigos e relatórios das duas iniciações científicas de **Emanuel Pinto Sampaio** na Universidade Federal do Espírito Santo (UFES), orientadas pelo prof. **Iury Angelo Gonçalves**, no projeto *Um estudo climatológico do efeito das gotículas espumas nas trocas de gás CO₂ na interface oceano-atmosfera*.

| | [IC1](ic1/) (Piic 2024/2025) | [IC2](ic2/) (Piic 2025/2026) |
|---|---|---|
| Subprojeto | Métodos matemáticos para resolver uma equação de difusão de massa em superfície esférica | Aplicação do método de Runge-Kutta com controle por subpassos em uma equação de difusão de massa em superfície esférica |
| Modelo | simplificado: r(t) e T(t) por decaimento exponencial; só a massa de CO₂ m(t) é integrada | sistema completo de EDOs acopladas para r, T e m, comparado com o modelo simplificado |
| Método numérico | Euler implícito (Euler Backward) | Runge-Kutta de 3ª ordem (SSPRK3) com controle por subpassos; versão paralela com MPI |
| Experimentos | E-CONTROL, E-R, E-T e E-RT; r_i = 30, 100, 250 e 500 µm | ECONTROL, E1, E2 e E3; r_i = 30 e 500 µm |

## Estrutura

```
ic1/  e  ic2/
├── relatorio/           relatório final (PDF)
├── figuras_relatorio/   figuras e tabelas que aparecem no relatório
└── codigos/             scripts Python e módulos físicos (NEWTON, T_EQ, TAU_R, TAU_T, DGH…)
```

Cada pasta `codigos/` é independente: os módulos ficam ao lado dos scripts, então basta entrar na pasta e rodar.

## Como rodar

Requisitos: Python 3 com `numpy` e `matplotlib`.

```bash
pip install numpy matplotlib

cd ic2/codigos
python IC2_completo_RK3_dt_fixo_500.py          # scripts da raiz de codigos/ rodam direto

PYTHONPATH=. python desenvolvimento/mpi/a.py    # scripts em subpastas: rodar de dentro de codigos/ com PYTHONPATH=.
```

A IC2 também pode ser executada no Google Colab por um menu interativo; veja [ic2/README.md](ic2/README.md).
Os scripts MPI da IC2 precisam de `mpi4py` e de uma implementação MPI (OpenMPI ou MPICH).

As referências bibliográficas completas estão nos relatórios, na pasta `relatorio/` de cada IC.
