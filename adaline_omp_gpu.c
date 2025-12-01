/* adaline_omp_gpu.c
 *
 * ADALINE - OpenMP target (GPU offload) - Batch gradient (versão ajustada)
 *
 * Compilar (exemplo com clang + NVPTX offload):
 *   clang -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda adaline_omp_gpu.c -o adaline_omp_gpu -lm
 *
 * Se seu toolchain não suportar offload, compile sem targets:
 *   gcc adaline_omp_gpu.c -o adaline_omp_gpu -fopenmp -lm
 *
 * Observação: só o comando clang acima garante realmente offload para GPU NVidia.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX_LINE 1024
#define EPOCHS 50

typedef struct
{
    int n_inputs;
    float *weights;
    float bias;
    float learning_rate;
} Adaline;

/* ------------ Inicialização ------------ */
void adaline_init(Adaline *m, int n_inputs, float lr)
{
    m->n_inputs = n_inputs;
    m->learning_rate = lr;
    m->bias = 0.0f;
    m->weights = (float *)malloc(n_inputs * sizeof(float));
    if (!m->weights)
    {
        fprintf(stderr, "malloc fail\n");
        exit(1);
    }
    srand(1234);
    for (int i = 0; i < n_inputs; i++)
        m->weights[i] = ((float)rand() / (float)RAND_MAX) * 0.01f;
}

/* ------------ Predict (host) ------------ */
static inline float adaline_predict_host(const float *weights, int n_inputs, float bias, const float *x)
{
    float s = bias;
    for (int i = 0; i < n_inputs; i++)
        s += weights[i] * x[i];
    return s;
}

/* ------------ MSE (host, parallel) ------------ */
float adaline_mse_host(Adaline *m, float **X, float *y, int n_samples)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n_samples; i++)
    {
        float p = adaline_predict_host(m->weights, m->n_inputs, m->bias, X[i]);
        float e = y[i] - p;
        sum += (double)e * (double)e;
    }
    return (float)(sum / n_samples);
}

/* ------------ load_dataset (igual ao sequencial) ------------ */
int load_dataset(const char *filename, float ***X_out, float **y_out, int *n_features)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir CSV: %s\n", filename);
        return 0;
    }

    char line[MAX_LINE];
    int count = 0;
    fgets(line, MAX_LINE, f); // header
    while (fgets(line, MAX_LINE, f))
        count++;
    rewind(f);
    fgets(line, MAX_LINE, f); // skip header again

    *n_features = 7;
    float **X = malloc(count * sizeof(float *));
    float *y = malloc(count * sizeof(float));
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
            break;
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

        X[idx] = malloc((*n_features) * sizeof(float));
        if (!X[idx])
        {
            fprintf(stderr, "malloc fail\n");
            exit(1);
        }
        X[idx][0] = Age;
        X[idx][1] = Gender;
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

/* ------------ normalize (host) ------------ */
void normalize_dataset(float **X, int n_samples, int n_features)
{
    for (int j = 0; j < n_features; j++)
    {
        float minv = X[0][j], maxv = X[0][j];
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

/* ------------ Train epoch with OpenMP target offload ------------ */
void adaline_train_epoch_omp_target(Adaline *m, float **X, float *y, int n_samples, int n_teams)
{
    int n_features = m->n_inputs;

    /* Linearize X for contiguous mapping */
    float *X_lin = (float *)malloc((size_t)n_samples * n_features * sizeof(float));
    if (!X_lin)
    {
        fprintf(stderr, "malloc fail\n");
        exit(1);
    }
    for (int i = 0; i < n_samples; i++)
        for (int j = 0; j < n_features; j++)
            X_lin[(size_t)i * n_features + j] = X[i][j];

    /* allocate grad arrays on host (will be mapped to device) */
    float *grad_w = (float *)calloc(n_features, sizeof(float));
    float grad_b = 0.0f;
    if (!grad_w)
    {
        fprintf(stderr, "malloc fail\n");
        exit(1);
    }

    /* Copy weights and bias to simple arrays for mapping */
    float *weights_copy = (float *)malloc(n_features * sizeof(float));
    if (!weights_copy)
    {
        fprintf(stderr, "malloc fail\n");
        exit(1);
    }
    for (int j = 0; j < n_features; j++)
        weights_copy[j] = m->weights[j];
    float bias_copy = m->bias;

/* Offload: each work-item processes one sample, accumulates into grad_w/grad_b on device using atomic ops */
/* Note: atomic updates on grad_w can be slower but are portable. */
#pragma omp target data map(to : X_lin[0 : (size_t)n_samples * n_features], y[0 : n_samples], weights_copy[0 : n_features], bias_copy) \
    map(tofrom : grad_w[0 : n_features], grad_b)
    {
/* distribute work across teams; thread_limit chosen by runtime */
#pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < n_samples; i++)
        {
            float pred = bias_copy;
            for (int j = 0; j < n_features; j++)
            {
                pred += weights_copy[j] * X_lin[(size_t)i * n_features + j];
            }
            float err = y[i] - pred;

/* atomic accumulate bias */
#pragma omp atomic update
            grad_b += err;

            /* atomic accumulate each weight gradient */
            for (int j = 0; j < n_features; j++)
            {
                float v = err * X_lin[(size_t)i * n_features + j];
#pragma omp atomic update
                grad_w[j] += v;
            }
        }
    } /* end target data */

    /* Update weights on host */
    float scale = m->learning_rate / (float)n_samples;
    for (int j = 0; j < n_features; j++)
        m->weights[j] += scale * grad_w[j];
    m->bias += scale * grad_b;

    free(weights_copy);
    free(grad_w);
    free(X_lin);
}

/* ------------ main ------------ */
int main(int argc, char **argv)
{
    const char *path = "./data/synthetic_employee_burnout.csv";
    int n_teams = 128;
    int epochs = EPOCHS;

    float **X;
    float *y;
    int n_features, n_samples;

    // printf("Carregando dataset: %s\n", path);
    
    n_samples = load_dataset(path, &X, &y, &n_features);
    // printf("Dataset carregado: %d amostras, %d features\n", n_samples, n_features);

    normalize_dataset(X, n_samples, n_features);
    // printf("Dataset normalizado.\n");

    Adaline model;
    adaline_init(&model, n_features, 0.000001f); /* learning rate conservador */

    double t0 = omp_get_wtime();

    for (int e = 0; e < epochs; e++)
    {
        adaline_train_epoch_omp_target(&model, X, y, n_samples, n_teams);
        float mse = adaline_mse_host(&model, X, y, n_samples);
        // printf("Epoch %d MSE=%.6f\n", e + 1, mse);
    }

    // double t1 = omp_get_wtime();
    // printf("Tempo total (s): %.6f\n", t1 - t0);

    free(model.weights);
    for (int i = 0; i < n_samples; i++)
        free(X[i]);
    free(X);
    free(y);

    return 0;
}
