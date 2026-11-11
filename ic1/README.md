# IC1: Métodos matemáticos para resolver uma equação de difusão de massa em superfície esférica

Edital Piic 2024/2025 · UFES · Orientador: Iury Angelo Gonçalves · Estudante: Emanuel Pinto Sampaio

## Modelo

Difusão de CO₂ através da superfície de uma gotícula de espuma marinha. O raio e a temperatura da gotícula seguem as soluções analíticas de decaimento exponencial (Andreas, 1989; 2005):

```
r(t) = r_eq + (r_i − r_eq)·exp(−t/τ_r)        T(t) = T_eq + (T_i − T_eq)·exp(−t/τ_T)
```

Só a massa de CO₂ m(t) é integrada numericamente, pelo método de **Euler implícito (Euler Backward)**, até o tempo de residência da gotícula na atmosfera, τ_f. O raio de equilíbrio r_eq é a raiz de ζ(r), encontrada pelo método de Newton.

| Experimento | r(t) | T(t) |
|---|---|---|
| E-CONTROL | constante | constante |
| E-R | varia | constante |
| E-T | constante | varia |
| E-RT | varia | varia |

As simulações do relatório usam H_s = 6 m e r_i = 30, 100, 250 e 500 µm.

**Parâmetros:** os scripts principais (`main_code.py`, `2)-EULER_IMPLICITO(BACKWARD)_SIMPLIFICAÇÕES.py`) estão como foram usados no relatório, com T_ar = 290 K, T_mar = 285 K, f = 0,8 e Φ_s calculado com `M_NaCl`. A versão de referência, com os parâmetros corretos (T_ar = 18 °C, T_mar = 293,15 K, T_gota = 19,8 °C, f = 0,9 e Φ_s com `M_H2O`), é a do TCC: veja [`../ic2/referencia_tcc/`](../ic2/referencia_tcc/) e o repositório [emanuelpintosampaio/TCC](https://github.com/emanuelpintosampaio/TCC).

## Pastas

- `relatorio/`: relatório final em PDF.
- `figuras_relatorio/`: figuras do relatório (simulações para 30, 100, 250 e 500 µm; gráficos de ζ(r) e do método de Newton).
- `codigos/`: scripts e módulos, descritos abaixo.

## Códigos

| Arquivo | Descrição |
|---|---|
| `main_code.py` | Código principal. Menu com os quatro experimentos (1 = E-RT, 2 = E-R, 3 = E-T, 4 = E-CONTROL), integrados por Euler implícito |
| `tau_f.py` | Velocidade terminal U_f (pelo método de Newton) e tempo de residência τ_f |
| `graph_req.py` | Gráfico de ζ(r) e do raio de equilíbrio r_eq |
| `1)-EULER_EXPLICITO(FORWARD)_SIMPLIFICAÇÕES.py`, `2)-EULER_IMPLICITO(BACKWARD)_SIMPLIFICAÇÕES.py`, `3)-RUNGE-KUTTA-4_SIMPLIFICAÇÕES.py` | Modelo simplificado resolvido por Euler explícito, Euler implícito e RK4 |
| `SIMPLE_EX_EULER.py`, `SIMPLE_IM_EULER.py`, `SIMPLE_RK3.py`, `SIMPLE_RK4.py` | O mesmo modelo, um método por arquivo |
| `IC1_GRAFICOS.py` | Gráfico comparando E-CONTROL, E-R, E-T e E-RT |
| `EULER_SIMPLES.py`, `EULER_ANDREAS.py`, `euler_andreas_1.py`, `EULER-PREDITOR-CORRETOR.py`, `RK4_SIMPLES.py`, `RK4_SIMPLES_sem_passo_adaptativo.py` | Variações com os parâmetros de Andreas e o método preditor-corretor |
| `euler_backward_r.py`, `edo_python.py`, `edo_fator_integrante.py` | Primeiros testes da EDO da massa (Euler implícito, solver do Python e solução por fator integrante) |
| `GRAFICO/` | Gráficos comparativos dos métodos (`base.py`, `graph_metd_euler.py`, `graph_metd_rk.py`) |
| `exploratorios/` | Testes e versões intermediárias: RK4 com passo adaptativo e primeiras versões do sistema completo |
| `NEWTON/`, `T_EQ/`, `TAU_R/`, `TAU_T/`, `DGH/`, `U_f/`, `CONSTANTES` | Módulos físicos: método de Newton, temperatura de equilíbrio, constantes de tempo τ_r e τ_T, D_g* e H*, velocidade terminal e constantes |

## Como rodar

```bash
cd ic1/codigos
python tau_f.py                                  # teste rápido (menos de 1 s)
python main_code.py                              # menu interativo
PYTHONPATH=. python exploratorios/testess.py     # scripts de subpastas
```

## Observações

- `rk4.py` não existe mais; por isso `IC1_GRAFICOS.py`, `EULER_SIMPLES.py`, `EULER-PREDITOR-CORRETOR.py`, `RK4_SIMPLES.py`, `RK4_SIMPLES_sem_passo_adaptativo.py` e, em `exploratorios/`, `6)-RK4_SISTEMA.py`, `7)-RK4_SISTEMA_PASSO_ADAPTATIVO.py` e `sistema_teste.py` (que fazem `from rk4 import ...`) não rodam.
- `graph_req.py`, `SIMPLE_EX_EULER.py`, `SIMPLE_IM_EULER.py`, `SIMPLE_RK3.py`, `EULER-PREDITOR-CORRETOR.py` e `RK4_SIMPLES_sem_passo_adaptativo.py` usam `text.usetex` e precisam de LaTeX instalado.
- `exploratorios/sistema_rk4_adaptativo.py`, `comparacao_fixo_vs_adaptativo.py` e `teste_validacao.py` salvam figuras num caminho absoluto que não existe nesta máquina (`/home/claude/...`).
- Vários scripts antigos importam `from GRAFICO.DGH import ...`; a pasta `GRAFICO/DGH/` é uma cópia de `DGH/` para que eles funcionem sem alteração.
