/*
 * Multiplicação de Matrizes - CUDA
 * Computa C = A × B para matrizes quadradas N×N
 *
 * Uso: ./mm_execs <tamanho_N> [repeticoes]
 * Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms
 *
 * Compila: nvcc -O2 -o mm_execs mm_execs.cu -lm
 *
 * ALINHAMENTO COM O POLYHOK (referência):
 *   - Arrays float (f32) e acumulador "int sum" (gerado pelo PolyHok) — mantidos
 *   - cudaMalloc DENTRO do timer (PolyHok cronometra os new_gnx do gpufor)  [ALTERADO]
 *   - Correção do laço de inicialização: i de 0 a m*m-1                      [ALTERADO]
 *     (o original ia de 1 a m*m: deixava a[0] sem inicializar e escrevia em
 *      a[m*m], uma posição fora do array)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_RUNS 30

/* -------------------------------------------------------------------------- */
__device__
float anon_ajh07a72e0(float *mat1, float *mat2, int m, int x, int y)
{
    int sum = 0;   /* acumulador inteiro, igual ao gerado pelo PolyHok */
    for (int i = 0; i < m; i += 1)
        sum = (sum + (mat1[((x * m) + i)] * mat2[((i * m) + y)]));
    return (sum);
}

extern "C" __global__ void map2xy2D_kernel(float *arr1, float *arr2, int par, float *resp, int size)
{
    int row = ((blockIdx.y * blockDim.y) + threadIdx.y);
    int col = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if (((col < size) && (row < size)))
        resp[((row * size) + col)] = anon_ajh07a72e0(arr1, arr2, par, row, col);
}

/* --------------------------------------------------------------------------
 * Timer cobre: cudaMalloc + H->D + kernel + sync + D->H
 * -------------------------------------------------------------------------- */
float run_matmul(int m, dim3 dimGrid, dim3 dimBlock)
{
    /* host FORA do timer */
    float *a = (float *)malloc(m * m * sizeof(float));
    float *b = (float *)malloc(m * m * sizeof(float));
    float *c = (float *)malloc(m * m * sizeof(float));

    srand((unsigned)time(NULL));
    for (int i = 0; i < m * m; i++) { a[i] = rand() % 1000; }
    for (int i = 0; i < m * m; i++) { b[i] = rand() % 1000; }

    float *d_a, *d_b, *d_c;

    float elapsed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* cudaMalloc DENTRO do timer (igual aos new_gnx do gpufor no PolyHok) */
    cudaMalloc((void **)&d_a, sizeof(float) * m * m);
    cudaMalloc((void **)&d_b, sizeof(float) * m * m);
    cudaMalloc((void **)&d_c, sizeof(float) * m * m);

    cudaMemcpy(d_a, a, sizeof(float) * m * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * m * m, cudaMemcpyHostToDevice);

    map2xy2D_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, m, d_c, m);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, sizeof(float) * m * m, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return elapsed;
}

/* -------------------------------------------------------------------------- */
int main(int argc, char const *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <tamanho_N> [repeticoes]\n", argv[0]);
        return 1;
    }

    int m      = atoi(argv[1]);
    int n_runs = (argc >= 3) ? atoi(argv[2]) : N_RUNS;

    int block_size = 16;
    int grid_rows  = (m + block_size - 1) / block_size;
    int grid_cols  = (m + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    float *tempos = (float *)malloc(n_runs * sizeof(float));

    for (int r = 0; r < n_runs; r++)
        tempos[r] = run_matmul(m, dimGrid, dimBlock);

    double soma = 0.0;
    float  minimo = tempos[0], maximo = tempos[0];
    for (int r = 0; r < n_runs; r++) {
        soma += tempos[r];
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
           "CUDA", m, media, desvio, (double)minimo, (double)maximo);

    return 0;
}
