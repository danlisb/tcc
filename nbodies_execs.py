"""
N-Bodies - Numba CUDA
Simulação gravitacional de N corpos (1 passo de tempo)

Cada corpo: 6 doubles [x, y, z, vx, vy, vz]. Layout flat: corpo i -> buf[6*i .. 6*i+5]

Uso: python nbodies_execs.py <n_corpos> [repeticoes]
Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms

Equivalente ao nbodies_execs.cu / nbodies.ex

ALINHAMENTO COM O POLYHOK (referência):
  - buf float64 (PolyHok: {:f,64}); forças fx,fy,fz e inv_dist* em float32
    (igual ao "float fx=0.0; float invDist=..." do CUDA)
  - Alocação na GPU DENTRO do timer: d_buf = cuda.to_device(h_buf)
    equivale a new_gnx(h_buf) (aloca + H->D), cronometrado pelo PolyHok.
  - h_buf (host) gerado FORA do timer (igual ao PolyHok, antes de prev).
  - cuda.synchronize() apenas após o kernel 1 (igual ao .cu).
  - SEM warmup: o JIT dos kernels ocorre dentro da região cronometrada.
"""

import sys
import warnings
import time
import numpy as np
from numba import cuda, float32 as nb_float32

warnings.filterwarnings("ignore", category=Warning)

N_RUNS    = 30
SOFTENING = nb_float32(1.0e-9)
DT        = nb_float32(0.01)


# ---------------------------------------------------------------------------
# Kernel 1 — forças + velocidades
# ---------------------------------------------------------------------------
@cuda.jit
def nbodies_force_kernel(buf, n):
    global_id = (cuda.blockDim.x * (cuda.gridDim.x * cuda.blockIdx.y + cuda.blockIdx.x)
                 + cuda.threadIdx.x)
    if global_id >= n:
        return

    p  = global_id * 6
    px = buf[p    ]
    py = buf[p + 1]
    pz = buf[p + 2]

    fx = nb_float32(0.0)
    fy = nb_float32(0.0)
    fz = nb_float32(0.0)

    for j in range(n):
        c = j * 6
        dx = buf[c    ] - px          # float64
        dy = buf[c + 1] - py
        dz = buf[c + 2] - pz

        dist_sqr  = dx*dx + dy*dy + dz*dz + SOFTENING                            # float64
        inv_dist  = nb_float32(1.0) / nb_float32(cuda.libdevice.sqrt(dist_sqr))  # float32
        inv_dist3 = inv_dist * inv_dist * inv_dist                                # float32

        fx += dx * inv_dist3
        fy += dy * inv_dist3
        fz += dz * inv_dist3

    buf[p + 3] += DT * fx
    buf[p + 4] += DT * fy
    buf[p + 5] += DT * fz


# ---------------------------------------------------------------------------
# Kernel 2 — integração das posições
# ---------------------------------------------------------------------------
@cuda.jit
def nbodies_integrate_kernel(buf, n):
    global_id = (cuda.blockDim.x * (cuda.gridDim.x * cuda.blockIdx.y + cuda.blockIdx.x)
                 + cuda.threadIdx.x)
    if global_id >= n:
        return

    p = global_id * 6
    buf[p    ] += buf[p + 3] * DT
    buf[p + 1] += buf[p + 4] * DT
    buf[p + 2] += buf[p + 5] * DT


# ---------------------------------------------------------------------------
# Executa uma rodada completa e devolve o tempo em ms
#   h_buf (host)  -> FORA do timer  (igual ao PolyHok, antes de prev)
#   to_device     -> DENTRO do timer (new_gnx: aloca + H->D)
#   k1, sync, k2, D->H -> DENTRO do timer
# ---------------------------------------------------------------------------
def run_nbodies(n_bodies, n_blocks, block_size):
    # host gerado fora do timer
    h_buf = np.random.rand(n_bodies * 6).astype(np.float64)

    t0 = time.perf_counter()

    d_buf = cuda.to_device(h_buf)         # new_gnx(h_buf): aloca + H->D (dentro do timer)

    nbodies_force_kernel[n_blocks, block_size](d_buf, n_bodies)
    cuda.synchronize()

    nbodies_integrate_kernel[n_blocks, block_size](d_buf, n_bodies)

    d_buf.copy_to_host(h_buf)             # get_gnx: D->H

    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python nbodies_execs.py <n_corpos> [repeticoes]")
        sys.exit(1)

    n_bodies   = int(sys.argv[1])
    n_runs     = int(sys.argv[2]) if len(sys.argv) >= 3 else N_RUNS
    block_size = 128
    n_blocks   = (n_bodies + block_size - 1) // block_size

    tempos = [run_nbodies(n_bodies, n_blocks, block_size) for _ in range(n_runs)]

    media  = np.mean(tempos)
    desvio = np.std(tempos)
    minimo = np.min(tempos)
    maximo = np.max(tempos)

    header = (f"{'Ferramenta':<12} {'Tamanho':<10} {'media_ms':<12} "
              f"{'desvio_ms':<12} {'min_ms':<12} {'max_ms':<12}")
    print(header)
    print(f"{'Numba':<12} {n_bodies:<10} {media:<12.2f} {desvio:<12.2f} "
          f"{minimo:<12.2f} {maximo:<12.2f}")


if __name__ == "__main__":
    main()
