"""
SAXPY - Numba CUDA
Operação: result[i] = 2 * a[i] + b[i]

Uso: python saxpy_execs.py <tamanho> [repeticoes]
Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms

Equivalente ao saxpy_execs.cu / saxpy_rts.ex

ALINHAMENTO COM O POLYHOK (referência):
  - Tipo float32 (PolyHok: {:f,32})
  - Alocação na GPU DENTRO do timer (PolyHok cronometra new_gnx).
    Inputs: cuda.to_device  -> equivale a new_gnx(vet)  (aloca + H->D)
    Output: cuda.device_array -> equivale a new_gnx(1,n,{:f,32}) (só aloca)
  - SEM warmup: a 1ª (e única, quando repeticoes=1) execução compila o kernel
    via JIT dentro da região cronometrada. Para que o JIT entre em TODAS as
    medições, rode 1 medição por processo (use run_interleaved.py com repeticoes=1).
"""

import sys
import time
import warnings
import numpy as np
from numba import cuda, float32 as nb_float32

warnings.filterwarnings("ignore", category=Warning)

N_RUNS = 30


# ---------------------------------------------------------------------------
# Kernel — float32, stride loop. 2*a+b mantido em f32 (PolyHok: 2*a+b com 2 int)
# ---------------------------------------------------------------------------
@cuda.jit
def saxpy_kernel(a, b, result, size):
    index  = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    i = index
    while i < size:
        result[i] = nb_float32(2.0) * a[i] + b[i]
        i += stride


# ---------------------------------------------------------------------------
# Função de execução — alocação GPU DENTRO do timer (igual ao new_gnx do PolyHok)
# ---------------------------------------------------------------------------
def run_saxpy(size, host_a, host_b, host_result, threads_per_block, number_of_blocks):
    t0 = time.perf_counter()

    # new_gnx(vet1) / new_gnx(vet2): aloca + copia H->D
    dev_a = cuda.to_device(host_a)
    dev_b = cuda.to_device(host_b)
    # new_gnx(1,n,{:f,32}): só aloca
    dev_result = cuda.device_array(size, dtype=np.float32)

    # spawn — 1ª chamada compila o kernel (JIT) dentro do timer
    saxpy_kernel[number_of_blocks, threads_per_block](dev_a, dev_b, dev_result, size)

    # get_gnx: D->H (bloqueante)
    dev_result.copy_to_host(host_result)

    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python saxpy_execs.py <tamanho> [repeticoes]")
        sys.exit(1)

    size   = int(sys.argv[1])
    n_runs = int(sys.argv[2]) if len(sys.argv) >= 3 else N_RUNS

    # Dados no host (float32). Geração FORA do timer (igual ao PolyHok, antes de prev)
    host_a      = np.arange(1, size + 1, dtype=np.float32)
    host_b      = np.arange(1, size + 1, dtype=np.float32)
    host_result = np.zeros(size, dtype=np.float32)

    threads_per_block = 128
    number_of_blocks  = (size + threads_per_block - 1) // threads_per_block

    tempos = []
    for _ in range(n_runs):
        tempos.append(run_saxpy(size, host_a, host_b, host_result,
                                threads_per_block, number_of_blocks))

    media  = np.mean(tempos)
    desvio = np.std(tempos)
    minimo = np.min(tempos)
    maximo = np.max(tempos)

    header = f"{'Ferramenta':<12} {'Tamanho':<10} {'media_ms':<12} {'desvio_ms':<12} {'min_ms':<12} {'max_ms':<12}"
    print(header)
    print(f"{'Numba':<12} {size:<10} {media:<12.2f} {desvio:<12.2f} {minimo:<12.2f} {maximo:<12.2f}")


if __name__ == "__main__":
    main()
