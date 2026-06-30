/*
 * SAXPY - CUDA
 * Operação: result[i] = 2 * a[i] + b[i]
 *
 * Uso: ./saxpy_execs <tamanho> [repeticoes]
 * Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms
 *
 * Compila: nvcc -O2 -o saxpy_execs saxpy_execs.cu -lm
 *
 * ALINHAMENTO COM O POLYHOK (referência):
 *   - float (f32), igual ao {:f,32} do PolyHok          [ALTERADO: era double]
 *   - cudaMalloc DENTRO do timer (PolyHok cronometra new_gnx)  [ALTERADO]
 *   - host malloc/preenchimento FORA do timer (igual ao PolyHok, antes de prev)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_RUNS 30

/* -------------------------------------------------------------------------- */
__global__ void comprehension(float *a, float *b, float *result, int size)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int id = index; id < size; id += stride)
        result[id] = (2 * a[id]) + b[id];
}

/* --------------------------------------------------------------------------
 * Executa uma rodada completa e devolve o tempo em ms
 * Timer cobre: cudaMalloc + H->D + kernel + D->H   (igual ao new_gnx/spawn/get_gnx)
 * -------------------------------------------------------------------------- */
float run_saxpy(int size, int threadsPerBlock, int numberOfBlocks)
{
    int bytes = size * sizeof(float);

    /* host FORA do timer (igual ao PolyHok, antes de prev) */
    float *host_a      = (float *)malloc(bytes);
    float *host_b      = (float *)malloc(bytes);
    float *host_result = (float *)malloc(bytes);

    for (int i = 0; i < size; i++) {
        host_a[i] = i + 1;
        host_b[i] = i + 1;
    }

    float *dev_a, *dev_b, *dev_result;

    float elapsed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* cudaMalloc DENTRO do timer (igual ao new_gnx do PolyHok) */
    cudaMalloc((void **)&dev_a,      bytes);
    cudaMalloc((void **)&dev_b,      bytes);
    cudaMalloc((void **)&dev_result, bytes);

    cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice);

    comprehension<<<numberOfBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_result, size);

    cudaMemcpy(host_result, dev_result, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    free(host_a);
    free(host_b);
    free(host_result);

    return elapsed;
}

/* -------------------------------------------------------------------------- */
int main(int argc, char const *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <tamanho> [repeticoes]\n", argv[0]);
        return 1;
    }

    int size   = atoi(argv[1]);
    int n_runs = (argc >= 3) ? atoi(argv[2]) : N_RUNS;

    int threadsPerBlock = 128;
    int numberOfBlocks  = (size + threadsPerBlock - 1) / threadsPerBlock;

    float *tempos = (float *)malloc(n_runs * sizeof(float));

    for (int r = 0; r < n_runs; r++)
        tempos[r] = run_saxpy(size, threadsPerBlock, numberOfBlocks);

    double soma = 0.0;
    float  minimo = tempos[0], maximo = tempos[0];
    for (int r = 0; r < n_runs; r++) {
        soma   += tempos[r];
        if (tempos[r] < minimo) minimo = tempos[r];
        if (tempos[r] > maximo) maximo = tempos[r];
    }
    double media = soma / n_runs;
    double var = 0.0;
    for (int r = 0; r < n_runs; r++) { double d = tempos[r] - media; var += d * d; }
    double desvio = sqrt(var / n_runs);

    free(tempos);

    printf("%-12s %-10s %-12s %-12s %-12s %-12s\n",
           "Ferramenta", "Tamanho", "media_ms", "desvio_ms", "min_ms", "max_ms");
    printf("%-12s %-10d %-12.2f %-12.2f %-12.2f %-12.2f\n",
           "CUDA", size, media, desvio, (double)minimo, (double)maximo);

    return 0;
}
