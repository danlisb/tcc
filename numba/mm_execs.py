"""
Multiplicação de Matrizes - Numba CUDA
Computa C = A × B para matrizes quadradas N×N

Uso: python mm_execs.py <tamanho_N> [repeticoes]
Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms

Equivalente ao mm_execs.cu / mm.ex

ALINHAMENTO COM O POLYHOK (referência):
  - Arrays float32 (PolyHok: {:f,32})
  - ACUMULADOR EM FLOAT32: corrigido o bug do acumulador inteiro ("int sum = 0")
    herdado da geração de código do PolyHok, que truncava os produtos parciais
    a cada iteração. A soma passa a ser acumulada em float32.
  - Alocação na GPU DENTRO do timer (PolyHok cronometra os new_gnx do gpufor).
  - SEM warmup: o JIT do kernel ocorre dentro da região cronometrada.
"""

import sys
import time
import warnings
import numpy as np
from numba import cuda
from numba import float32 as nb_float32

warnings.filterwarnings("ignore", category=Warning)

N_RUNS = 30


# ---------------------------------------------------------------------------
# Kernel — acumulador em float32 (produto interno correto)
# ---------------------------------------------------------------------------
@cuda.jit
def matmul_kernel(a, b, c, size):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < size and col < size:
        acc = nb_float32(0.0)   # acumulador em precisão simples (float32)
        for i in range(size):
            acc += a[row * size + i] * b[i * size + col]
        c[row * size + col] = acc


# ---------------------------------------------------------------------------
# Função de execução — alocação GPU DENTRO do timer (igual aos new_gnx do gpufor)
# ---------------------------------------------------------------------------
def run_matmul(m, a, b, c_host, dim_grid, dim_block):
    t0 = time.perf_counter()

    d_a = cuda.to_device(a)                       # new_gnx(arr1): aloca + H->D
    d_b = cuda.to_device(b)                       # new_gnx(arr2): aloca + H->D
    d_c = cuda.device_array(m * m, dtype=np.float32)  # new_gnx(result): só aloca

    matmul_kernel[dim_grid, dim_block](d_a, d_b, d_c, m)
    cuda.synchronize()

    d_c.copy_to_host(c_host)                      # get_gnx: D->H

    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python mm_execs.py <tamanho_N> [repeticoes]")
        sys.exit(1)

    m      = int(sys.argv[1])
    n_runs = int(sys.argv[2]) if len(sys.argv) >= 3 else N_RUNS

    rng = np.random.default_rng()
    a = rng.integers(0, 1000, size=m * m, dtype=np.int32).astype(np.float32)
    b = rng.integers(0, 1000, size=m * m, dtype=np.int32).astype(np.float32)
    c_host = np.zeros(m * m, dtype=np.float32)

    block_size = 16
    grid_rows  = (m + block_size - 1) // block_size
    grid_cols  = (m + block_size - 1) // block_size
    dim_block  = (block_size, block_size)
    dim_grid   = (grid_cols, grid_rows)

    tempos = []
    for _ in range(n_runs):
        tempos.append(run_matmul(m, a, b, c_host, dim_grid, dim_block))

    media  = np.mean(tempos)
    desvio = np.std(tempos)
    minimo = np.min(tempos)
    maximo = np.max(tempos)

    header = f"{'Ferramenta':<12} {'Tamanho':<10} {'media_ms':<12} {'desvio_ms':<12} {'min_ms':<12} {'max_ms':<12}"
    print(header)
    print(f"{'Numba':<12} {m:<10} {media:<12.2f} {desvio:<12.2f} {minimo:<12.2f} {maximo:<12.2f}")


if __name__ == "__main__":
    main()
