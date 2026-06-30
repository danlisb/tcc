"""
Julia Set - Numba CUDA
Gera o conjunto de Julia para uma grade DIM×DIM

Cada pixel (x,y) produz 4 valores int32: [R, G, B, A]
Buffer flat: pixel (x,y) → buf[(x + y*DIM)*4 .. (x + y*DIM)*4 + 3]

Uso: python julia_execs.py <DIM> [repeticoes]
Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms

Equivalente ao julia.cu / julia.ex
"""

import sys
import time
import warnings
import numpy as np
from numba import cuda, float32 as nb_float32, int32 as nb_int32

warnings.filterwarnings("ignore", category=Warning)

N_RUNS = 30


# ---------------------------------------------------------------------------
# Device function: julia(x, y, dim) → 0 ou 1
# Idêntica ao __device__ int julia() do julia.cu — todos os tipos float32
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def julia(x, y, dim):
    scale = nb_float32(0.1)
    jx    = nb_float32(scale * (dim - x) / dim)
    jy    = nb_float32(scale * (dim - y) / dim)

    cr = nb_float32(-0.8)
    ci = nb_float32(0.156)
    ar = jx
    ai = jy

    for i in range(200):
        nar = nb_float32((ar * ar - ai * ai) + cr)
        nai = nb_float32((ai * ar + ar * ai) + ci)

        if (nar * nar + nai * nai) > nb_float32(1.0e3):
            return nb_int32(0)

        ar = nar
        ai = nai

    return nb_int32(1)


# ---------------------------------------------------------------------------
# Device function: julia_function(ptr, x, y, dim)
# Idêntica ao __device__ void julia_function() do julia.cu
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def julia_function(ptr, x, y, dim):
    offset     = x + y * dim
    juliaValue = julia(x, y, dim)

    ptr[offset * 4 + 0] = 255 * juliaValue
    ptr[offset * 4 + 1] = 0
    ptr[offset * 4 + 2] = 0
    ptr[offset * 4 + 3] = 255


# ---------------------------------------------------------------------------
# Kernel: mapgen2D_xy_1para_noret_ker(resp, arg1, size)
# Idêntico ao __global__ mapgen2D_xy_1para_noret_ker do julia.cu
# Grid: (DIM, DIM) blocos de 1 thread — igual ao <<<grid, 1>>> do CUDA
# ---------------------------------------------------------------------------
@cuda.jit
def julia_kernel(resp, arg1, size):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x < size and y < size:
        julia_function(resp, x, y, arg1)


# ---------------------------------------------------------------------------
# Função de execução
# Nota: no CUDA original cudaMalloc está DENTRO do timer (após cudaEventRecord),
# portanto d_pixelbuffer é alocado dentro desta função, igual ao original.
# ---------------------------------------------------------------------------
def run_julia(dim, h_pixelbuffer, grid_dim, block_dim):
    t0 = time.perf_counter()

    # cudaMalloc dentro do timer — igual ao julia.cu (linha 148 após linha 142)
    d_pixelbuffer = cuda.device_array(dim * dim * 4, dtype=np.int32)

    julia_kernel[grid_dim, block_dim](d_pixelbuffer, dim, dim)

    # D→H — equivalente ao cudaMemcpy D→H do CUDA
    d_pixelbuffer.copy_to_host(h_pixelbuffer)

    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python julia_execs.py <DIM> [repeticoes]")
        sys.exit(1)

    dim    = int(sys.argv[1])
    n_runs = int(sys.argv[2]) if len(sys.argv) >= 3 else N_RUNS

    # Buffer host — alocado uma vez fora do loop (igual ao malloc do CUDA)
    h_pixelbuffer = np.zeros(dim * dim * 4, dtype=np.int32)

    # Grid (DIM, DIM) com 1 thread por bloco — igual ao <<<dim3(DIM,DIM), 1>>> do CUDA
    grid_dim  = (dim, dim)
    block_dim = (1, 1)

    # Medições reais (inclui JIT na 1ª iteração)
    tempos = []
    for _ in range(n_runs):
        tempos.append(run_julia(dim, h_pixelbuffer, grid_dim, block_dim))

    media  = np.mean(tempos)
    desvio = np.std(tempos)
    minimo = np.min(tempos)
    maximo = np.max(tempos)

    header = f"{'Ferramenta':<12} {'Tamanho':<10} {'media_ms':<12} {'desvio_ms':<12} {'min_ms':<12} {'max_ms':<12}"
    print(header)
    print(f"{'Numba':<12} {dim:<10} {media:<12.2f} {desvio:<12.2f} {minimo:<12.2f} {maximo:<12.2f}")


if __name__ == "__main__":
    main()
