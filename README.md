# Avaliação de Eficiência e Escalabilidade de Programas Paralelos em CUDA, PolyHok e Numba

Repositório de código e dados do Trabalho de Conclusão de Curso de **Daniel Lisboa Pereira**
(Bacharelado em Ciência da Computação — UFPel), orientado pelo **Prof. Dr. André Rauber Du Bois**.

O trabalho avalia comparativamente a eficiência e a escalabilidade de cinco *benchmarks*
de GPU implementados em três abordagens — **CUDA** (linha de base), **PolyHok** e **Numba** —,
medindo o tempo de execução em cinco tamanhos de entrada por *benchmark* e analisando os
resultados estatisticamente.

## Estrutura do repositório

| Pasta        | Conteúdo |
|--------------|----------|
| `numba/`     | Implementações dos cinco *benchmarks* em Numba (`*_execs.py`). |
| `cuda/`      | Versões em CUDA (`*_execs.cu`), adaptadas neste trabalho para assegurar a paridade com as demais. |
| `polyhok/`   | Implementações de referência em PolyHok (`*.ex`). **Material de terceiro** — ver Créditos. |
| `scripts/`   | `run_interleaved.py` (coleta) e `analise_estatistica.py` (análise). |
| `dados/`     | `resultados.csv` (medições brutas) e os resumos derivados da análise. |

## Benchmarks e tamanhos

| Benchmark              | Perfil                | Tamanhos |
|------------------------|-----------------------|----------|
| Multiplicação de Matrizes | Compute-bound      | 2000, 2500, 3000, 3500, 4000 |
| SAXPY                  | Memory-bound          | 50M, 75M, 100M, 125M, 150M |
| N-Bodies               | Compute-bound, O(N²)  | 15000, 20000, 25000, 30000, 35000 |
| Conjunto de Julia      | Compute-bound         | 500, 1000, 1500, 2000, 2500 |
| Nearest Neighbor       | Memory-bound          | 20M, 40M, 60M, 80M, 100M |

## Dados

- **`dados/resultados.csv`** — dado primário do trabalho: 2250 medições no formato
  `ferramenta,benchmark,tamanho,rodada,tempo_ms` (5 *benchmarks* × 5 tamanhos × 3 ferramentas × 30 repetições).
- **`dados/resumo_normalidade.csv`** e **`dados/resumo_comparacoes.csv`** — gerados pela
  análise a partir do arquivo acima (estatística descritiva, testes de normalidade e comparações).

## Como reproduzir

### 1. Coleta dos tempos

O `run_interleaved.py` executa todos os casos de forma intercalada (rodadas em ordem aleatória,
um processo novo por medição, sem *warmup* — medindo o custo a frio com compilação JIT). O tempo
é lido da saída de cada programa (medição interna à ROI), não do relógio do processo.

```bash
python3 scripts/run_interleaved.py --rounds 30 --seed 12345
# gera dados/resultados.csv
```

Requisitos: `nvcc` no PATH (CUDA), Python com `numba` (Numba) e a cadeia Elixir + PolyHok (`.ex`).
No ambiente original, o PolyHok foi executado sob WSL2 e o orquestrador o invoca por um *wrapper*;
os caminhos no início do script refletem a máquina de testes e podem precisar de ajuste ao ambiente local.

### 2. Análise estatística

```bash
python3 scripts/analise_estatistica.py dados/resultados.csv
# gera resumo_normalidade.csv, resumo_comparacoes.csv e a pasta figuras/
```

Requisitos: `numpy`, `pandas`, `scipy`, `matplotlib`.

```bash
pip install numpy pandas scipy matplotlib
```

A análise verifica a normalidade de cada caso (Shapiro-Wilk e Anderson-Darling) e seleciona o
teste de forma adaptativa: ANOVA + *t* de Welch quando os grupos são normais e homocedásticos,
ou Kruskal-Wallis + Mann-Whitney (com correção de Holm) caso contrário. Os tempos são reportados
por mediana e intervalo interquartil (IQR), com a razão de cada abordagem em relação ao CUDA.

## Ambiente experimental

| Componente | Especificação |
|------------|---------------|
| GPU        | NVIDIA GeForce GTX 970, 4 GB GDDR5 |
| CPU        | Intel Core i5-12400F |
| RAM        | 16 GB |
| Driver NVIDIA | 577.00 |
| SO (CUDA, Numba) | Windows 10 |
| SO (PolyHok)     | WSL2 — Ubuntu 20.04.6 LTS |
| CUDA Toolkit | 11.8 (V11.8.89) |
| Python / Numba | 3.13.5 / 0.61.2 |
| Elixir / Erlang OTP | 1.17.3 / OTP 27 (erts 15.2.3) |
| PolyHok | 0.1.0 |

## Créditos

As implementações em PolyHok (`polyhok/*.ex`) e as versões originais em CUDA que serviram de
base às adaptações deste trabalho provêm do repositório oficial da linguagem PolyHok, de autoria
de **André Rauber Du Bois** e **Gerson Geraldo H. Cavalheiro**:

> DU BOIS, A. R.; CAVALHEIRO, G. *Polymorphic Higher-Order GPU Kernels*. In: XXIV Brazilian
> Symposium on Programming Languages (SBLP 2025), Pelotas, RS, Brasil, 2025.
