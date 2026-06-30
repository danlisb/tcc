"""
Nearest Neighbor - Numba CUDA
Encontra a menor distância euclidiana entre um ponto de referência (0,0)
e um conjunto de N localizações (lat, lng)

Layout flat: location i → buf[2*i] = lat, buf[2*i+1] = lng

Uso: python nearest_neighbor_execs.py <numRecords> [repeticoes]
Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms

Equivalente ao nearest_neighbor.cu / nearest_neighbor.ex

NOTA SOBRE TIPOS:
  - CUDA original   : float32
  - PolyHok (.ex)   : float64  (gen_data_set_nx_double → {:f,64})
  - Este arquivo    : float64  ← corrigido para paridade com o PolyHok

NOTA SOBRE JIT:
  O tempo de compilação JIT da 1ª execução está INCLUÍDO nas medições,
  de forma análoga ao tempo de compilação do PolyHok (compilação em runtime).
  Não há warmup separado.
"""

import sys
import time
import warnings
import math
import numpy as np
from numba import cuda, float64 as nb_float64

warnings.filterwarnings("ignore", category=Warning)

N_RUNS = 30


# ---------------------------------------------------------------------------
# Device function: euclid(d_locations, offset, lat, lng)
# Equivalente ao __device__ float euclid(float *d_locations, float lat, float lng)
# do CUDA — recebe array flat + offset (substitui aritmética de ponteiro C)
# Tipo: float64 para paridade com o PolyHok (gen_data_set_nx_double)
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def euclid(d_locations, offset, lat, lng):
    dlat = lat - d_locations[offset]
    dlng = lng - d_locations[offset + 1]
    return math.sqrt(dlat * dlat + dlng * dlng)


# ---------------------------------------------------------------------------
# Device function: menor(x, y) → mínimo entre x e y
# Equivalente ao __device__ float menor(float x, float y) do CUDA
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def menor(x, y):
    if x < y:
        return x
    return y


# ---------------------------------------------------------------------------
# Kernel 1: map_step_2para_1resp_kernel
# Equivalente ao __global__ map_step_2para_1resp_kernel do CUDA
# Lança: <<<numRecords, 1>>>  (1 thread por bloco, fiel ao CUDA original)
# ---------------------------------------------------------------------------
@cuda.jit
def map_step_2para_1resp_kernel(d_array, d_result, step, par1, par2, size):
    global_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    id_       = step * global_id
    if global_id < size:
        d_result[global_id] = euclid(d_array, id_, par1, par2)


# ---------------------------------------------------------------------------
# Kernel 2: reduce_kernel
# Equivalente ao __global__ reduce_kernel do CUDA
# Usa shared memory de 256 posições + atomic min (equivalente ao CAS do CUDA)
# Lança: <<<blocksPerGrid, 256>>>
# ---------------------------------------------------------------------------
@cuda.jit
def reduce_kernel(a, ref4, n):
    cache = cuda.shared.array(256, dtype=nb_float64)

    tid         = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    cache_index = cuda.threadIdx.x

    # Fase 1: cada thread reduz localmente, igual ao while(tid < n) do CUDA
    temp = ref4[0]
    while tid < n:
        temp = menor(a[tid], temp)
        tid += cuda.blockDim.x * cuda.gridDim.x

    cache[cache_index] = temp
    cuda.syncthreads()

    # Fase 2: redução em shared memory (árvore binária), igual ao CUDA
    i = cuda.blockDim.x // 2
    while i != 0:
        if cache_index < i:
            cache[cache_index] = menor(cache[cache_index + i], cache[cache_index])
        cuda.syncthreads()
        i //= 2

    # Fase 3: escreve resultado do bloco com atomic min
    # Equivalente ao atomic_cas do CUDA — cuda.atomic.min é funcionalmente
    # idêntico: garante atualização atômica do mínimo global em ref4[0]
    if cache_index == 0:
        cuda.atomic.min(ref4, 0, cache[0])


# ---------------------------------------------------------------------------
# Geração de dados — equivalente ao gen_data_set_nx_double() do PolyHok
# (float64, igual ao Elixir: gen_data_set_nx_double → {:f, 64})
# lat: [7+rand(0..63)] + rand/RAND_MAX
# lng: [rand(0..357)]  + rand/RAND_MAX
# ---------------------------------------------------------------------------
def load_data(num_records):
    rng = np.random.default_rng()
    lat = (7 + rng.integers(0, 63, size=num_records)).astype(np.float64) \
          + rng.random(num_records).astype(np.float64)
    lng = rng.integers(0, 358, size=num_records).astype(np.float64) \
          + rng.random(num_records).astype(np.float64)
    # layout intercalado: [lat0, lng0, lat1, lng1, ...]
    locations = np.empty(num_records * 2, dtype=np.float64)
    locations[0::2] = lat
    locations[1::2] = lng
    return locations


# ---------------------------------------------------------------------------
# Função de execução
# O timer inicia ANTES das alocações GPU (cudaMalloc está dentro do timer
# no CUDA original, após cudaEventRecord linha 130).
# O tempo de JIT da 1ª execução é capturado naturalmente na 1ª iteração.
# ---------------------------------------------------------------------------
def run_nn(num_records, locations):
    threads_per_block = 256
    blocks_per_grid   = (num_records + threads_per_block - 1) // threads_per_block

    t0 = time.perf_counter()

    # cudaMalloc + cudaMemcpy dentro do timer — fiel ao CUDA original
    d_locations = cuda.to_device(locations)
    d_distances = cuda.device_array(num_records, dtype=np.float64)

    # Kernel 1: calcula distâncias euclidianas
    # <<<numRecords, 1>>> — fiel ao CUDA (linha 148 do .cu)
    map_step_2para_1resp_kernel[num_records, 1](
        d_locations, d_distances, 2,
        nb_float64(0.0), nb_float64(0.0),
        num_records
    )
    cuda.synchronize()

    # Inicializa resp com 50000 — fiel ao CUDA (linha 158 do .cu)
    resp   = np.array([50000.0], dtype=np.float64)
    d_resp = cuda.to_device(resp)

    # Kernel 2: redução para mínimo
    # <<<blocksPerGrid, 256>>> — fiel ao CUDA (linha 162 do .cu)
    reduce_kernel[blocks_per_grid, threads_per_block](d_distances, d_resp, num_records)
    cuda.synchronize()

    # D→H — fiel ao cudaMemcpy D→H (linha 166 do .cu)
    d_resp.copy_to_host(resp)

    # cudaFree antes do stop do timer — fiel ao CUDA (linhas 172-173 do .cu)
    del d_locations
    del d_distances

    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python nearest_neighbor_execs.py <numRecords> [repeticoes]")
        sys.exit(1)

    num_records = int(sys.argv[1])
    n_runs      = int(sys.argv[2]) if len(sys.argv) >= 3 else N_RUNS

    # Dados gerados no host uma vez, fora do timer
    # (equivalente ao gen_data_set_nx_double antes do prev = System.monotonic_time()
    # no PolyHok, linha 182 do .ex)
    locations = load_data(num_records)

    # Medições — JIT incluído na 1ª iteração, sem warmup separado
    tempos = []
    for _ in range(n_runs):
        tempos.append(run_nn(num_records, locations))

    media  = np.mean(tempos)
    desvio = np.std(tempos)
    minimo = np.min(tempos)
    maximo = np.max(tempos)

    header = (f"{'Ferramenta':<12} {'Tamanho':<10} {'media_ms':<12} "
              f"{'desvio_ms':<12} {'min_ms':<12} {'max_ms':<12}")
    print(header)
    print(f"{'Numba':<12} {num_records:<10} {media:<12.2f} "
          f"{desvio:<12.2f} {minimo:<12.2f} {maximo:<12.2f}")


if __name__ == "__main__":
    main()