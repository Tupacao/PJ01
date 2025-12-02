/* adaline_cuda.cu
 *
 * ADALINE - CUDA otimizado (redução por bloco em shared memory)
 *
 * Compilar:
 *   nvcc -O3 adaline_cuda.cu -o adaline_cuda
 *
 * Uso:
 *   ./adaline_cuda [dataset.csv] [epochs] [blocksize]
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>


#define MAX_LINE 1024

/* Carrega dataset burnout (mesmo formato das versões seq/OMP) */
int load_dataset(const char *filename, float ***X_out, float **y_out, int *n_features)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir CSV: %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];
    int count = 0;

    if (!fgets(line, MAX_LINE, f)) // header
    {
        fprintf(stderr, "CSV vazio ou corrompido: %s\n", filename);
        fclose(f);
        exit(1);
    }
    while (fgets(line, MAX_LINE, f))
        count++;

    rewind(f);
    if (!fgets(line, MAX_LINE, f)) // header novamente
    {
        fprintf(stderr, "CSV vazio ou corrompido: %s\n", filename);
        fclose(f);
        exit(1);
    }

    *n_features = 7;
    float **X = (float **)malloc(count * sizeof(float *));
    float *y = (float *)malloc(count * sizeof(float));
    if (!X || !y)
    {
        fprintf(stderr, "malloc fail\n");
        exit(1);
    }

    int idx = 0;
    while (fgets(line, MAX_LINE, f))
    {
        char *token = strtok(line, ",");

        char name[64];
        char gender_str[16];
        char jobrole[32];
        float Age, Experience, WorkHours, RemoteRatio, Sat, Stress;
        int Gender, Burnout;

        if (!token)
            continue;
        strncpy(name, token, sizeof(name));
        name[sizeof(name) - 1] = 0;

        token = strtok(NULL, ",");
        Age = token ? atof(token) : 0.0f;

        token = strtok(NULL, ",");
        strncpy(gender_str, token ? token : "", sizeof(gender_str));
        gender_str[sizeof(gender_str) - 1] = 0;

        token = strtok(NULL, ",");
        strncpy(jobrole, token ? token : "", sizeof(jobrole));
        jobrole[sizeof(jobrole) - 1] = 0;

        token = strtok(NULL, ",");
        Experience = token ? atof(token) : 0.0f;

        token = strtok(NULL, ",");
        WorkHours = token ? atof(token) : 0.0f;

        token = strtok(NULL, ",");
        RemoteRatio = token ? atof(token) : 0.0f;

        token = strtok(NULL, ",");
        Sat = token ? atof(token) : 0.0f;

        token = strtok(NULL, ",");
        Stress = token ? atof(token) : 0.0f;

        token = strtok(NULL, ",");
        Burnout = token ? atoi(token) : 0;

        Gender = (strcmp(gender_str, "Male") == 0) ? 0 : 1;

        X[idx] = (float *)malloc((*n_features) * sizeof(float));
        if (!X[idx])
        {
            fprintf(stderr, "malloc fail\n");
            exit(1);
        }
        X[idx][0] = Age;
        X[idx][1] = (float)Gender;
        X[idx][2] = Experience;
        X[idx][3] = WorkHours;
        X[idx][4] = RemoteRatio;
        X[idx][5] = Sat;
        X[idx][6] = Stress;

        y[idx] = (float)Burnout;
        idx++;
    }

    fclose(f);
    *X_out = X;
    *y_out = y;
    return idx;
}

void normalize_dataset(float **X, int n_samples, int n_features)
{
    for (int j = 0; j < n_features; j++)
    {
        float minv = X[0][j];
        float maxv = X[0][j];

        for (int i = 1; i < n_samples; i++)
        {
            if (X[i][j] < minv)
                minv = X[i][j];
            if (X[i][j] > maxv)
                maxv = X[i][j];
        }

        float range = maxv - minv;
        if (range < 1e-6f)
            range = 1.0f;

        for (int i = 0; i < n_samples; i++)
            X[i][j] = (X[i][j] - minv) / range;
    }
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

//executado em GPU
__global__ void compute_contribs_block_kernel(const float *X_lin, const float *y,
                                              const float *d_weights, float bias,
                                              float *d_grad_w, float *d_grad_b,
                                              int n_samples, int n_features)
{
    extern __shared__ float s_grad[];
    
    float *s_grad_w = s_grad;        

    float *s_grad_b = (float *)(s_grad + n_features);

    for (int j = threadIdx.x; j < n_features; j += blockDim.x)
        s_grad_w[j] = 0.0f; //zerando gradiente dos pesos

    if (threadIdx.x == 0)
        *s_grad_b = 0.0f; //zera bias
    __syncthreads(); //aguarda todas

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples)
    {
        // acessa região do bloco
        const float *x = X_lin + (size_t)idx * n_features;
        
        float pred = bias;

        for (int j = 0; j < n_features; j++)
            pred += d_weights[j] * x[j];

        float err = y[idx] - pred;

        atomicAdd(s_grad_b, err);
        for (int j = 0; j < n_features; j++)
        {
            float v = err * x[j];
            atomicAdd(&s_grad_w[j], v);
        }
    }

    __syncthreads(); //sincroniza

    if (threadIdx.x == 0)
    {
        atomicAdd(d_grad_b, *s_grad_b);
        for (int j = 0; j < n_features; j++)
        {
            float val = s_grad_w[j];
            if (val != 0.0f)
                atomicAdd(&d_grad_w[j], val);
        }
    }
}


int main()
{
    const char *dataset_path = "./data/synthetic_employee_burnout.csv";
    int epochs = 20;
    int blocksize = 256; 

    float **X = NULL;
    float *y = NULL;
    int n_features = 0;
    int n_samples = load_dataset(dataset_path, &X, &y, &n_features);

    normalize_dataset(X, n_samples, n_features);

    /* lineariza X  */
    size_t X_bytes = (size_t)n_samples * n_features * sizeof(float);
    float *X_lin = (float *)malloc(X_bytes);
    for (int i = 0; i < n_samples; i++)
        memcpy(X_lin + (size_t)i * n_features, X[i], n_features * sizeof(float));

    /* init weights no host */
    float *weights = (float *)malloc(n_features * sizeof(float));
    srand(1234);
    for (int j = 0; j < n_features; j++)
        weights[j] = ((float)rand() / RAND_MAX) * 0.01f;
    float bias = 0.0f;
    float lr = 0.000001f; 

    /* device buffers */
    float *d_X = NULL, *d_y = NULL, *d_weights = NULL, *d_grad_w = NULL, *d_grad_b = NULL;
    cudaMalloc((void **)&d_X, X_bytes)
    cudaMalloc((void **)&d_y, (size_t)n_samples * sizeof(float))
    cudaMalloc((void **)&d_weights, (size_t)n_features * sizeof(float))
    cudaMalloc((void **)&d_grad_w, (size_t)n_features * sizeof(float))
    cudaMalloc((void **)&d_grad_b, sizeof(float))

    /* copy X e y sao constantes */
    cudaMemcpy(d_X, X_lin, X_bytes, cudaMemcpyHostToDevice)
    cudaMemcpy(d_y, y, (size_t)n_samples * sizeof(float), cudaMemcpyHostToDevice)

    int nblocks = (n_samples + blocksize - 1) / blocksize;
    size_t shmem_bytes = (size_t)(n_features + 1) * sizeof(float);

    float *grad_w = (float *)malloc((size_t)n_features * sizeof(float));
   
    float grad_b = 0.0f;

    for (int e = 0; e < epochs; e++)
    {
        /* copia pesos atuais para device */
        cudaMemcpy(d_weights, weights, (size_t)n_features * sizeof(float), cudaMemcpyHostToDevice)

        /* zera gradientes globais no device */
        cudaMemset(d_grad_w, 0, (size_t)n_features * sizeof(float))
        cudaMemset(d_grad_b, 0, sizeof(float))

        /*  acumula em shared, depois escreve em global */
        compute_contribs_block_kernel<<<nblocks, blocksize, shmem_bytes>>>(d_X, d_y, d_weights, bias, d_grad_w, d_grad_b, n_samples, n_features);
        cudaGetLastError()
        cudaDeviceSynchronize()

        /* copia gradientes de volta */
        cudaMemcpy(grad_w, d_grad_w, (size_t)n_features * sizeof(float), cudaMemcpyDeviceToHost)
        cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost)

        /* update host weights */
        float scale = lr / (float)n_samples;
        for (int j = 0; j < n_features; j++)
            weights[j] += scale * grad_w[j];
        bias += scale * grad_b;

        float mse = host_mse(weights, bias, X, y, n_samples, n_features);
        // printf("Epoch %d MSE=%.6f\n", e + 1, mse);
    }

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_weights);
    cudaFree(d_grad_w);
    cudaFree(d_grad_b);
    free(grad_w);
    free(weights);
    free(X_lin);
    for (int i = 0; i < n_samples; i++)
        free(X[i]);
    free(X);
    free(y);

    return 0;
}
