/* adaline_seq.c
 *
 * ADALINE - Versão Sequencial (online / SGD)
 *
 * Instruções de compilação:
 *   gcc adaline_seq.c -o adaline_seq -lm
 *
 * Uso:
 *   ./adaline_seq
 *
 * ---------------------------
 * TEMPOS (preencher após testes):
 * Sequencial: TODO
 * OpenMP (1,2,4,8,16,32): TODO
 * CUDA: TODO
 * ---------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE 1024
#define EPOCHS 100



/* Modelo ADALINE */
typedef struct
{
    int n_inputs;
    float *weights;
    float bias;
    float learning_rate;
} Adaline;

/* Inicialização do modelo */
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

/* y = w·x + b */
float adaline_predict(Adaline *m, const float *x)
{
    float s = m->bias;
    for (int i = 0; i < m->n_inputs; i++)
        s += m->weights[i] * x[i];
        
    return s;
}

/* Treina 1 época (online / SGD) */
void adaline_train_epoch_sgd(Adaline *m, float **X, float *y, int n_samples)
{
    for (int s = 0; s < n_samples; s++)
    {
        float pred = adaline_predict(m, X[s]);
        float err = y[s] - pred;

        //coreção dos pesos
        for (int j = 0; j < m->n_inputs; j++)
            m->weights[j] += m->learning_rate * err * X[s][j];

        m->bias += m->learning_rate * err; //correção da bias
    }
}

/* MSE */
float adaline_mse(Adaline *m, float **X, float *y, int n_samples)
{
    double sum = 0.0;
    for (int s = 0; s < n_samples; s++)
    {
        float p = adaline_predict(m, X[s]);
        float e = y[s] - p;
        sum += e * e;
    }
    return (float)(sum / n_samples);
}

/* Carrega dataset burnout */
int load_dataset(
    const char *filename,
    float ***X_out,
    float **y_out,
    int *n_features)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        printf("Erro ao abrir CSV: %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];
    int count = 0;

    fgets(line, MAX_LINE, f); // ignore header

    while (fgets(line, MAX_LINE, f))
        count++;

    rewind(f);
    fgets(line, MAX_LINE, f); // header de novo

    *n_features = 7; // Age, Gender, Exp, WorkHours, RemoteRatio, Sat, Stress

    float **X = malloc(count * sizeof(float *));
    float *y = malloc(count * sizeof(float));

    int idx = 0;
    while (fgets(line, MAX_LINE, f))
    {
        char *token = strtok(line, ",");

        char name[64];
        char gender_str[16];
        char jobrole[32];
        float Age, Experience, WorkHours, RemoteRatio, Sat, Stress;
        int Gender, Burnout;

        strcpy(name, token);

        token = strtok(NULL, ",");
        Age = atof(token);

        token = strtok(NULL, ",");
        strcpy(gender_str, token);

        token = strtok(NULL, ",");
        strcpy(jobrole, token);

        token = strtok(NULL, ",");
        Experience = atof(token);

        token = strtok(NULL, ",");
        WorkHours = atof(token);

        token = strtok(NULL, ",");
        RemoteRatio = atof(token);

        token = strtok(NULL, ",");
        Sat = atof(token);

        token = strtok(NULL, ",");
        Stress = atof(token);

        token = strtok(NULL, ",");
        Burnout = atoi(token);

        if (strcmp(gender_str, "Male") == 0)
            Gender = 0;
        else
            Gender = 1;

        X[idx] = malloc((*n_features) * sizeof(float));
        X[idx][0] = Age;
        X[idx][1] = Gender;
        X[idx][2] = Experience;
        X[idx][3] = WorkHours;
        X[idx][4] = RemoteRatio;
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
        if (range < 1e-6f) range = 1.0f;  // evita divisão por zero

        for (int i = 0; i < n_samples; i++)
            X[i][j] = (X[i][j] - minv) / range;
    }
}

/* MAIN */
int main(int argc, char **argv)
{

    float **X;
    float *y;
    int n_features;
    int n_samples;

    /* Carregar dataset REAL */
    n_samples = load_dataset("./data/synthetic_employee_burnout.csv", &X, &y, &n_features);
    // printf("Dataset carregado: %d amostras, %d features\n", n_samples, n_features);

    normalize_dataset(X, n_samples, n_features);
    // printf("Dataset normalizado.\n");

    Adaline model;
    adaline_init(&model, n_features, 0.01f);

    // clock_t t0 = clock();

    for (int e = 0; e < EPOCHS; e++)
    {
        adaline_train_epoch_sgd(&model, X, y, n_samples);
        adaline_mse(&model, X, y, n_samples);
        // float mse = adaline_mse(&model, X, y, n_samples);
        // printf("Epoch %d MSE=%.6f\n", e, mse);
    }

    // clock_t t1 = clock();
    // double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    // printf("\nTempo total (s): %.4f\n", elapsed);

    free(model.weights);
    for (int i = 0; i < n_samples; i++)
        free(X[i]);
    free(X);
    free(y);

    return 0;
}