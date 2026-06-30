/*
 * N-Bodies - CUDA
 * Simulação gravitacional de N corpos (1 passo de tempo)
 *
 * Cada corpo: [x, y, z, vx, vy, vz] — layout flat, 6 doubles por corpo
 *
 * Uso: ./nbodies_execs <n_corpos> [repeticoes]
 * Saída: Ferramenta  Tamanho  media_ms  desvio_ms  min_ms  max_ms
 *
 * Compila: nvcc -O2 -o nbodies_execs nbodies_execs.cu -lm
 *
 * ALINHAMENTO COM O POLYHOK (referência):
 *   - buf double (f64); forças em float (f32) — mantidos
 *   - cudaMalloc(d_buf) DENTRO do timer (PolyHok cronometra new_gnx)  [ALTERADO]
 *   - h_buf + randomizeBodies FORA do timer (igual ao PolyHok, antes de prev)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_RUNS 30

/* -------------------------------------------------------------------------- */
__device__
void gpu_nBodies(double *p, double *c, int n)
{
    float softening = 1.0e-9;
    float dt = 0.01;
    float fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < n; j++) {
        double dx = (c[(6 * j)]         - p[0]);
        double dy = (c[((6 * j) + 1)]   - p[1]);
        double dz = (c[((6 * j) + 2)]   - p[2]);
        double distSqr = dx*dx + dy*dy + dz*dz + softening;
        float invDist  = (1.0 / sqrt(distSqr));
        float invDist3 = invDist * invDist * invDist;
        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }

    p[3] = (p[3] + (dt * fx));
    p[4] = (p[4] + (dt * fy));
    p[5] = (p[5] + (dt * fz));
}

__device__
void gpu_integrate(double *p, float dt, int n)
{
    p[0] = (p[0] + (p[3] * dt));
    p[1] = (p[1] + (p[4] * dt));
    p[2] = (p[2] + (p[5] * dt));
}

__global__ void map1(double *d_array, int step, double *par1, int par2, int size)
{
    int globalId = ((blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x)) + threadIdx.x);
    int id       = (step * globalId);
    if (globalId < size)
        gpu_nBodies((d_array + id), par1, par2);
}

__global__ void map2(double *d_array, int step, float par1, int par2, int size)
{
    int globalId = ((blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x)) + threadIdx.x);
    int id       = (step * globalId);
    if (globalId < size)
        gpu_integrate((d_array + id), par1, par2);
}

/* -------------------------------------------------------------------------- */
void randomizeBodies(double *data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = (double)(rand() / (float)RAND_MAX);
}

/* --------------------------------------------------------------------------
 * Timer cobre: cudaMalloc(d_buf) + H->D + map1 + sync + map2 + D->H
 * -------------------------------------------------------------------------- */
float run_nbodies(int nBodies, int block_size, int nBlocks)
{
    int bytes  = nBodies * sizeof(double) * 6;

    /* host FORA do timer (igual ao PolyHok, antes de prev) */
    double *h_buf  = (double *)malloc(bytes);
    double *d_resp = (double *)malloc(bytes);
    randomizeBodies(h_buf, 6 * nBodies);

    double *d_buf;

    float elapsed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* cudaMalloc DENTRO do timer (igual ao new_gnx do PolyHok) */
    cudaMalloc(&d_buf, bytes);

    cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice);

    map1<<<nBlocks, block_size>>>(d_buf, 6, d_buf, nBodies, nBodies);
    cudaDeviceSynchronize();

    map2<<<nBlocks, block_size>>>(d_buf, 6, 0.01, nBodies, nBodies);

    cudaMemcpy(d_resp, d_buf, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_buf);
    free(h_buf);
    free(d_resp);

    return elapsed;
}

/* -------------------------------------------------------------------------- */
int main(int argc, char const *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <n_corpos> [repeticoes]\n", argv[0]);
        return 1;
    }

    int nBodies = atoi(argv[1]);
    int n_runs  = (argc >= 3) ? atoi(argv[2]) : N_RUNS;

    int block_size = 128;
    int nBlocks    = (nBodies + block_size - 1) / block_size;

    float *tempos = (float *)malloc(n_runs * sizeof(float));

    for (int r = 0; r < n_runs; r++)
        tempos[r] = run_nbodies(nBodies, block_size, nBlocks);

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
           "CUDA", nBodies, media, desvio, (double)minimo, (double)maximo);

    return 0;
}
