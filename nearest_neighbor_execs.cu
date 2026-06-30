/*
 * nn.cu
 * Nearest Neighbor
 * Modified by André Du Bois: changed depracated api, creating data set in memory. clean up code not used
 *
 * ALINHAMENTO COM O POLYHOK (referência):
 *   - double (f64), igual ao gen_data_set_nx_double -> {:f,64} do PolyHok  [ALTERADO: era float]
 *     atomicCAS de double feito via reinterpretação em unsigned long long.
 *   - cudaMalloc já estava DENTRO do timer (mantido).
 *
 * Compila: nvcc -O2 -o nearest_neighbor_execs nearest_neighbor_execs.cu -lm
 */

#include <stdio.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define N_RUNS 30

__device__ static double atomic_cas(double* address, double oldv, double newv)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    return __longlong_as_double(
        atomicCAS(address_as_ull,
                  __double_as_longlong(oldv),
                  __double_as_longlong(newv)));
}

__device__
double euclid(double *d_locations, double lat, double lng)
{
return (sqrt((((lat - d_locations[0]) * (lat - d_locations[0])) + ((lng - d_locations[1]) * (lng - d_locations[1])))));
}

extern "C" __global__ void map_step_2para_1resp_kernel(double *d_array, double *d_result, int step, double par1, double par2, int size)
{
	int globalId = (threadIdx.x + (blockIdx.x * blockDim.x));
	int id = (step * globalId);
if((globalId < size))
{
	d_result[globalId] = euclid((d_array + id), par1, par2);
}

}

__device__
double menor(double x, double y)
{
if((x < y))
{
return (x);
}
else{
return (y);
}

}

extern "C" __global__ void reduce_kernel(double *a, double *ref4, int n)
{
__shared__ double cache[256];
	int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
	int cacheIndex = threadIdx.x;
	double temp = ref4[0];
while((tid < n)){
	temp = menor(a[tid], temp);
	tid = ((blockDim.x * gridDim.x) + tid);
}
	cache[cacheIndex] = temp;
__syncthreads();
	int i = (blockDim.x / 2);
while((i != 0)){
if((cacheIndex < i))
{
	cache[cacheIndex] = menor(cache[(cacheIndex + i)], cache[cacheIndex]);
}

__syncthreads();
	i = (i / 2);
}
if((cacheIndex == 0))
{
	double current_value = ref4[0];
while((! (current_value == atomic_cas(ref4, current_value, menor(cache[0], current_value))))){
	current_value = ref4[0];
}
}

}

void loadData(double *locations, int size);

/* --------------------------------------------------------------------------
 * Timer cobre: cudaMalloc + H->D + kernel1 + sync + reduce + sync + D->H
 * -------------------------------------------------------------------------- */
float run_nn(double *locations, int numRecords)
{
    double *distances;
    double *d_locations;
    double *d_distances;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    distances = (double *)malloc(sizeof(double) * numRecords);
    cudaMalloc((void **) &d_locations, sizeof(double) * 2 * numRecords);
    cudaMalloc((void **) &d_distances, sizeof(double) * numRecords);

    cudaMemcpy(d_locations, &locations[0], sizeof(double) * 2 * numRecords, cudaMemcpyHostToDevice);

    map_step_2para_1resp_kernel<<< numRecords, 1 >>>(d_locations, d_distances, 2, 0.0, 0.0, numRecords);

    cudaDeviceSynchronize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (numRecords + threadsPerBlock - 1) / threadsPerBlock;

    double *resp, *d_resp;
    resp = (double *)malloc(sizeof(double));
    resp[0] = 50000;
    cudaMalloc((void **) &d_resp, sizeof(double));
    cudaMemcpy(d_resp, resp, sizeof(double), cudaMemcpyHostToDevice);

    reduce_kernel<<< blocksPerGrid, threadsPerBlock >>>(d_distances, d_resp, numRecords);
    cudaDeviceSynchronize();

    cudaMemcpy(resp, d_resp, sizeof(double), cudaMemcpyDeviceToHost);

    free(distances);
    cudaFree(d_locations);
    cudaFree(d_distances);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(resp);
    cudaFree(d_resp);

    return time;
}

/* -------------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    double *locations;
    int numRecords = atoi(argv[1]);
    int n_runs     = (argc >= 3) ? atoi(argv[2]) : N_RUNS;

    locations = (double *)malloc(sizeof(double) * 2 * numRecords);
    loadData(locations, numRecords);

    float *tempos = (float *)malloc(n_runs * sizeof(float));

    for (int r = 0; r < n_runs; r++)
        tempos[r] = run_nn(locations, numRecords);

    double soma  = 0.0;
    float minimo = tempos[0], maximo = tempos[0];
    for (int r = 0; r < n_runs; r++) {
        soma += tempos[r];
        if (tempos[r] < minimo) minimo = tempos[r];
        if (tempos[r] > maximo) maximo = tempos[r];
    }
    double media = soma / n_runs;
    double var   = 0.0;
    for (int r = 0; r < n_runs; r++) { double d = tempos[r] - media; var += d * d; }
    double desvio = sqrt(var / n_runs);

    free(tempos);
    free(locations);

    printf("%-12s %-10s %-12s %-12s %-12s %-12s\n",
           "Ferramenta", "Tamanho", "media_ms", "desvio_ms", "min_ms", "max_ms");
    printf("%-12s %-10d %-12.2f %-12.2f %-12.2f %-12.2f\n",
           "CUDA", numRecords, media, desvio, (double)minimo, (double)maximo);

    return 0;
}

void loadData(double* locations, int size){

	for (int i=0;i<size;i++){

            locations[0] = ((double)(7 + rand() % 63)) + ((double) rand() / (double) 0x7fffffff);

            locations[1] = ((double)(rand() % 358)) + ((double) rand() / (double) 0x7fffffff);

            locations = locations +2;


        }

}
