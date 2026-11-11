# Referência: códigos do TCC com os parâmetros corretos

Estes são os códigos do TCC ([github.com/emanuelpintosampaio/TCC](https://github.com/emanuelpintosampaio/TCC)) que fazem os mesmos experimentos da IC2 com o SSPRK3, mas com os **parâmetros corretos**. Eles foram copiados sem nenhuma alteração.

O algoritmo é o mesmo dos scripts da IC2 em `../codigos/`; só mudam os parâmetros abaixo.

| Parâmetro | IC2 (`../codigos/`) | TCC (esta pasta) |
|---|---|---|
| Temperatura do ar, `T_a` | 16,85 °C | 18 °C |
| Temperatura do mar, `T_mar_em_k` | 285 K | 293,15 K |
| Temperatura inicial da gota, `T_gota` | 11,85 °C | 19,8 °C |
| Umidade relativa fracionária, `f` | 0,8 | 0,9 |
| Coeficiente osmótico, `Φ_s` | `calcular_phi_s(m_s_val, M_NaCl, m_w)` | `calcular_phi_s(m_s_val, M_H2O, m_w)` |

## Equivalência com os experimentos da IC2

| IC2 | TCC | Arquivo | Passo | Origem |
|---|---|---|---|---|
| ECONTROL | SCONTROL | `SCONTROL_30.py`, `SCONTROL_500.py` | fixo, Δt = 10⁻⁴ s | repositório TCC (`CODE/`) |
| E1 | S1 | `S1.py` (30 µm) | subpassos, Δt = 10⁻³ s, M = 10 | versão do TCC no computador do autor (não publicada) |
| E2 | S3 | `S3.py` (30 µm) | fixo, Δt = 10⁻⁴ s | idem |
| E3 | S4 | `S4.py` (30 µm) | subpassos, Δt = 10⁻³ s, M = 10 | idem |

No TCC, o SCONTROL é a solução de referência usada para calcular o erro dos experimentos com Euler Explícito. Os módulos (`DGH/`, `NEWTON/`, `TAU_R/`, `TAU_T/`, `T_EQ/`) também são os do repositório TCC.

## Resultados

Massa final de CO₂ (mol):

| Experimento | r_i = 30 µm | r_i = 500 µm |
|---|---|---|
| SCONTROL | 6.790447587743291e-16 | 6.560494261756399e-12 |
| S1 | 6.79044890729124e-16 | |
| S3 | 5.865095392449541e-16 | |
| S4 | 5.865095392306654e-16 | |

Conferência: usando o `SCONTROL_30.py` e o `SCONTROL_500.py` como referência, o ECONTROL do TCC (Euler Explícito) tem erro relativo de 4,94 × 10⁻⁷ (30 µm) e 1,23 × 10⁻⁷ (500 µm), os mesmos valores da tabela de erros do TCC.

## Como rodar

```bash
cd ic2/referencia_tcc
python SCONTROL_500.py      # grava SCONTROL_500.png
python S1.py                # abre o gráfico na tela
```

Em `S1.py`, `S3.py` e `S4.py`, o raio inicial é fixo em 30 µm (`r_i` no topo do arquivo).
