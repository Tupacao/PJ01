/* adaline_omp_cpu.c
 *
 * ADALINE - Versão OpenMP (Batch Gradient Descent)
 *
 * Compilar:
 *   gcc adaline_omp_cpu.c -o adaline_omp_cpu -fopenmp -lm
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MAX_LINE 1024
#define EPOCHS 100

/*
      Estrutura ADALINE
   */
typedef struct
{
    int n_inputs;
    float *weights;
    float bias;
    float learning_rate;
} Adaline;

/*
      Inicialização
   */
void adaline_init(Adaline *m, int n_inputs, float lr)
{
    m->n_inputs = n_inputs;
    m->learning_rate = lr;
    m->bias = 0.0f;
    m->weights = (float *)malloc(n_inputs * sizeof(float));

    srand(1234);
    for (int i = 0; i < n_inputs; i++)
        m->weights[i] = ((float)rand() / RAND_MAX) * 0.01f;
}

/*
      Forward
   */
float adaline_predict(Adaline *m, const float *x)
{
    float s = m->bias;
    for (int i = 0; i < m->n_inputs; i++)
        s += m->weights[i] * x[i];
    return s;
}

/*
      Treinamento BATCH (estável)
   */
void adaline_train_epoch_batch(
    Adaline *m, float **X, float *y, int n_samples, int n_threads)
{
    float *grad_w = calloc(m->n_inputs, sizeof(float));
    float grad_b = 0.0f;

    #pragma omp parallel num_threads(n_threads)
    {
        float *local_grad_w = calloc(m->n_inputs, sizeof(float));
        float local_grad_b = 0.0f;

        #pragma omp for
        for (int s = 0; s < n_samples; s++)
        {
            float pred = adaline_predict(m, X[s]);
            float err = y[s] - pred;

            local_grad_b += err;
            for (int j = 0; j < m->n_inputs; j++)
                local_grad_w[j] += err * X[s][j];
        }

        #pragma omp critical
        {
            grad_b += local_grad_b;
            for (int j = 0; j < m->n_inputs; j++)
                grad_w[j] += local_grad_w[j];
        }

        free(local_grad_w);
    }

    float lr = m->learning_rate / n_samples;

    for (int j = 0; j < m->n_inputs; j++)
        m->weights[j] += lr * grad_w[j];

    m->bias += lr * grad_b;

    free(grad_w);
}

/*
      MSE
   */
float adaline_mse(Adaline *m, float **X, float *y, int n_samples)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int s = 0; s < n_samples; s++)
    {
        float p = adaline_predict(m, X[s]);
        float e = y[s] - p;
        sum += e * e;
    }

    return (float)(sum / n_samples);
}

/*
      Carregar dataset
  */
int load_dataset(const char *filename, float ***X_out, float **y_out, int *n_features)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        printf("Erro ao abrir CSV: %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];
    int count = 0;

    fgets(line, MAX_LINE, f);

    while (fgets(line, MAX_LINE, f)) count++;

    rewind(f);
    fgets(line, MAX_LINE, f);

    *n_features = 7;

    float **X = malloc(count * sizeof(float *));
    float *y = malloc(count * sizeof(float));

    int idx = 0;

    while (fgets(line, MAX_LINE, f))
    {
        char *token = strtok(line, ",");

        char name[64];
        char gender_str[16];
        char jobrole[32];
        float Age, Exp, WH, RR, Sat, Stress;
        int Gender, Burnout;

        strcpy(name, token);

        token = strtok(NULL, ","); Age = atof(token);
        token = strtok(NULL, ","); strcpy(gender_str, token);
        token = strtok(NULL, ","); strcpy(jobrole, token);
        token = strtok(NULL, ","); Exp = atof(token);
        token = strtok(NULL, ","); WH = atof(token);
        token = strtok(NULL, ","); RR = atof(token);
        token = strtok(NULL, ","); Sat = atof(token);
        token = strtok(NULL, ","); Stress = atof(token);
        token = strtok(NULL, ","); Burnout = atoi(token);

        Gender = (strcmp(gender_str, "Male") == 0) ? 0 : 1;

        X[idx] = malloc((*n_features) * sizeof(float));
        X[idx][0] = Age;
        X[idx][1] = Gender;
        X[idx][2] = Exp;
        X[idx][3] = WH;
        X[idx][4] = RR;
        X[idx][5] = Sat;
        X[idx][6] = Stress;

        y[idx] = Burnout;
        idx++;
    }

    fclose(f);

    *X_out = X;
    *y_out = y;

    return count;
}

/* Normalização  */
void normalize_dataset(float **X, int n_samples, int n_features)
{
    for (int j = 0; j < n_features; j++)
    {
        float minv = X[0][j];
        float maxv = X[0][j];

        for (int i = 1; i < n_samples; i++)
        {
            if (X[i][j] < minv) minv = X[i][j];
            if (X[i][j] > maxv) maxv = X[i][j];
        }

        float range = maxv - minv;
        if (range < 1e-6f) range = 1.0f;

        for (int i = 0; i < n_samples; i++)
            X[i][j] = (X[i][j] - minv) / range;
    }
}

int main()
{
    float **X;
    float *y;
    int n_features, n_samples;

    // printf("Carregando dataset fixo: ./data/synthetic_employee_burnout.csv\n");

    n_samples = load_dataset("./data/synthetic_employee_burnout.csv",
                             &X, &y, &n_features);
    // printf("Dataset carregado: %d amostras, %d features\n",n_samples, n_features);

    normalize_dataset(X, n_samples, n_features);
    // printf("Dataset normalizado.\n");

    Adaline model;
    adaline_init(&model, n_features, 0.000001f); // learning rate seguro

    int threads = 32;

    // double t0 = omp_get_wtime();

    for (int e = 0; e < EPOCHS; e++)
    {
        adaline_train_epoch_batch(&model, X, y, n_samples, threads);
        float mse = adaline_mse(&model, X, y, n_samples);

        // printf("Epoch %d MSE=%.6f\n", e+1, mse);
    }

    // double t1 = omp_get_wtime();
    // printf("\nTempo total (s): %.4f\n", t1 - t0);

    free(model.weights);

    for (int i = 0; i < n_samples; i++)
        free(X[i]);
    
        free(X);
    
        free(y);

    return 0;
}