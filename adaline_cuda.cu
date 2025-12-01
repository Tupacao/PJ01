/* adaline_cuda_opt.cu
 *
 * ADALINE - CUDA otimizado (redução por bloco em shared memory)
 *
 * Compilar:
 *   nvcc -O3 adaline_cuda_opt.cu -o adaline_cuda_opt
 *
 * Uso:
 *   ./adaline_cuda_opt [n_samples] [n_features] [epochs] [blocksize]
 *
 * Observações:
 * - Este kernel faz redução por bloco em memória compartilhada para diminuir
 *   atomicAdd globais (melhora desempenho).
 * - Se n_features for muito grande, aumente blocksize para amortizar trabalho por bloco.
 * - Se usar dataset real, substitua generate_data_host por load_dataset e linearize.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECKCALL(x)                                                                              \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = (x);                                                                      \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            fprintf(stderr, "CUDA err %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

#define MAX_LINE 1024

/* --------------------
   Helpers / Data gen
   -------------------- */

void generate_data_host(float **X, float *y, int n_samples, int n_features)
{
    srand(42);
    float *true_w = (float *)malloc(n_features * sizeof(float));
    for (int j = 0; j < n_features; j++)
        true_w[j] = ((float)rand() / RAND_MAX) * 2 - 1;

    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_features; j++)
            X[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float s = 0.0f;
        for (int j = 0; j < n_features; j++)
            s += true_w[j] * X[i][j];
        y[i] = s + (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;
    }
    free(true_w);
}

/* host MSE (for monitoring) */
float host_mse(const float *weights, float bias, float **X, float *y, int n_samples, int n_features)
{
    double sum = 0.0;
    for (int i = 0; i < n_samples; i++)
    {
        double p = bias;
        for (int j = 0; j < n_features; j++)
            p += (double)weights[j] * (double)X[i][j];
        double e = (double)y[i] - p;
        sum += e * e;
    }
    return (float)(sum / n_samples);
}

/* --------------------
   CUDA kernel
   - cada thread processa 1 amostra
   - cada bloco possui um buffer shared s_grad_w[ n_features ]
   - threads do bloco acumulam em s_grad_w (usando atomicAdd sobre shared)
   - depois um único thread por bloco (threadIdx.x==0) faz atomicAdd global por feature
   Nota: atomicAdd sobre shared é suportado e é rápido; reduzimos global atomics a (#blocks * n_features)
   em vez de (n_samples * n_features).
   -------------------- */

/* Kernel: cada thread processa 1 amostra; redução por bloco em shared memory.
   shared memory size = n_features * sizeof(float)  -> passado dinamicamente no launch. */
__global__ void compute_contribs_block_kernel(const float *X_lin, const float *y,
                                              const float *d_weights, float bias,
                                              float *d_grad_w, float *d_grad_b,
                                              int n_samples, int n_features)
{
    extern __shared__ float s_grad[]; // size = n_features floats per block, mapped dynamically
    float *s_grad_w = s_grad;         // shared gradient per feature for this block
    // we also use one shared float for bias accumulation
    float *s_grad_b = (float *)(s_grad + n_features); // place bias after features (ensure shared size >= (n_features+1)*4)

    // initialize shared buffer to zero in parallel: each thread handles a strided set of indices
    for (int j = threadIdx.x; j < n_features; j += blockDim.x)
        s_grad_w[j] = 0.0f;

    if (threadIdx.x == 0)
        *s_grad_b = 0.0f;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples)
    {
        const float *x = X_lin + (size_t)idx * n_features;
        // compute prediction
        float pred = bias;
        // unrolling could help, but keep simple and generic
        for (int j = 0; j < n_features; j++)
            pred += d_weights[j] * x[j];
        float err = y[idx] - pred;

        // accumulate into shared arrays - use atomicAdd on shared memory
        // each thread will do n_features atomic ops in shared mem (fast)
        atomicAdd(s_grad_b, err);
        for (int j = 0; j < n_features; j++)
        {
            float v = err * x[j];
            atomicAdd(&s_grad_w[j], v);
        }
    }

    __syncthreads();

    // now thread 0 of block will add s_grad_w to global arrays (one atomic per feature per block)
    if (threadIdx.x == 0)
    {
        // accumulate bias
        atomicAdd(d_grad_b, *s_grad_b);
        // accumulate per-feature gradient to global memory
        for (int j = 0; j < n_features; j++)
        {
            float val = s_grad_w[j];
            if (val != 0.0f)
                atomicAdd(&d_grad_w[j], val);
        }
    }
}

/* --------------------
   Main (host)
   -------------------- */

int main(int argc, char **argv)
{
    int n_samples = 200000; // default
    int n_features = 128;
    int epochs = 20;
    int blocksize = 256;

    if (argc >= 2)
        n_samples = atoi(argv[1]);
    if (argc >= 3)
        n_features = atoi(argv[2]);
    if (argc >= 4)
        epochs = atoi(argv[3]);
    if (argc >= 5)
        blocksize = atoi(argv[4]);

    printf("ADALINE CUDA optimized: n_samples=%d n_features=%d epochs=%d blocksize=%d\n",
           n_samples, n_features, epochs, blocksize);

    /* allocate host matrix pointers (row pointers) */
    float **X = (float **)malloc(n_samples * sizeof(float *));
    for (int i = 0; i < n_samples; i++)
        X[i] = (float *)malloc(n_features * sizeof(float));
    float *y = (float *)malloc(n_samples * sizeof(float));

    /* generate synthetic data (or load dataset) */
    generate_data_host(X, y, n_samples, n_features);
    // If you prefer to load CSV, call your load_dataset(...) here and then normalize.

    /* linearize X into contiguous buffer for device transfer */
    size_t X_bytes = (size_t)n_samples * n_features * sizeof(float);
    float *X_lin = (float *)malloc(X_bytes);
    if (!X_lin)
    {
        fprintf(stderr, "malloc X_lin failed\n");
        return 1;
    }
    for (int i = 0; i < n_samples; i++)
        memcpy(X_lin + (size_t)i * n_features, X[i], n_features * sizeof(float));

    /* init weights on host */
    float *weights = (float *)malloc(n_features * sizeof(float));
    srand(1234);
    for (int j = 0; j < n_features; j++)
        weights[j] = ((float)rand() / RAND_MAX) * 0.01f;
    float bias = 0.0f;
    float lr = 0.05f;

    /* device buffers */
    float *d_X = NULL, *d_y = NULL, *d_weights = NULL, *d_grad_w = NULL, *d_grad_b = NULL;
    CHECKCALL(cudaMalloc((void **)&d_X, X_bytes));
    CHECKCALL(cudaMalloc((void **)&d_y, (size_t)n_samples * sizeof(float)));
    CHECKCALL(cudaMalloc((void **)&d_weights, (size_t)n_features * sizeof(float)));
    CHECKCALL(cudaMalloc((void **)&d_grad_w, (size_t)n_features * sizeof(float)));
    CHECKCALL(cudaMalloc((void **)&d_grad_b, sizeof(float)));

    /* copy X and y once (X doesn't change) */
    CHECKCALL(cudaMemcpy(d_X, X_lin, X_bytes, cudaMemcpyHostToDevice));
    CHECKCALL(cudaMemcpy(d_y, y, (size_t)n_samples * sizeof(float), cudaMemcpyHostToDevice));

    int nblocks = (n_samples + blocksize - 1) / blocksize;

    /* shared memory size per block: (n_features + 1) * sizeof(float) */
    size_t shmem_bytes = (size_t)(n_features + 1) * sizeof(float);
    if (shmem_bytes > 48 * 1024)
    {
        fprintf(stderr, "Warning: shared memory per block = %zu bytes; may exceed typical limit (48KB). Reduce n_features or switch strategy.\n", shmem_bytes);
    }

    clock_t t0 = clock();
    for (int e = 0; e < epochs; e++)
    {
        /* copy current weights to device */
        CHECKCALL(cudaMemcpy(d_weights, weights, (size_t)n_features * sizeof(float), cudaMemcpyHostToDevice));

        /* zero global gradients on device */
        CHECKCALL(cudaMemset(d_grad_w, 0, (size_t)n_features * sizeof(float)));
        CHECKCALL(cudaMemset(d_grad_b, 0, sizeof(float)));

        /* launch kernel: each block reduces to shared, block-0 writes to global */
        compute_contribs_block_kernel<<<nblocks, blocksize, shmem_bytes>>>(d_X, d_y, d_weights, bias, d_grad_w, d_grad_b, n_samples, n_features);
        CHECKCALL(cudaGetLastError());
        CHECKCALL(cudaDeviceSynchronize());

        /* copy gradients back */
        float *grad_w = (float *)malloc((size_t)n_features * sizeof(float));
        float grad_b;
        CHECKCALL(cudaMemcpy(grad_w, d_grad_w, (size_t)n_features * sizeof(float), cudaMemcpyDeviceToHost));
        CHECKCALL(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost));

        /* update host weights */
        float scale = lr / (float)n_samples;
        for (int j = 0; j < n_features; j++)
            weights[j] += scale * grad_w[j];
        bias += scale * grad_b;

        free(grad_w);

        /* monitoring */
        float mse = host_mse(weights, bias, X, y, n_samples, n_features);
        printf("Epoch %d MSE=%.6f\n", e + 1, mse);
    }
    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Tempo total (s): %.4f\n", elapsed);

    /* cleanup */
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_weights);
    cudaFree(d_grad_w);
    cudaFree(d_grad_b);
    free(weights);
    free(X_lin);
    for (int i = 0; i < n_samples; i++)
        free(X[i]);
    free(X);
    free(y);

    return 0;
}
