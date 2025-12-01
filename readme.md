# Trabalho de Paralelização – ADALINE (rede neural linear)
## Integrantes : Arthur Oliveira, Luis Phillip, Fernanda Rodrigues
Este projeto implementa um ADALINE usando:
- Versão sequencial em C;
- Versão paralela com **OpenMP para CPU**;
- Versão paralela com **OpenMP target (GPU)**;
- Versão paralela em **CUDA para GPU**.

Os dados são sintéticos: cada execução gera amostras aleatórias com pesos-alvo ocultos.

---

## Arquivos principais

- `adaline_seq.c`  
  Versão sequencial em C.

- `adaline_omp_cpu.c`  
  Versão OpenMP para CPU, testada com 1, 2, 4, 8, 16 e 32 threads.

- `adaline_omp_gpu.c`  
  Versão OpenMP com offload para GPU (target).

- `adaline_cuda.cu`  
  Versão CUDA otimizada com redução por bloco em shared memory.

- Pasta `resp/`  
  Logs de execução já medidos (`seq.txt`, `omp_cpu*.txt`, `omp_gpu.txt`, `cuda.txt`).

---

## Como compilar

Assumindo GCC com suporte a OpenMP e NVCC instalado:

```bash
# Sequencial
gcc adaline_seq.c -O2 -o adaline_seq -lm

# OpenMP CPU
gcc adaline_omp_cpu.c -O2 -fopenmp -o adaline_omp_cpu -lm

# OpenMP target (GPU) – ajuste a flag de offload conforme seu GCC/Clang
gcc adaline_omp_gpu.c -O2 -fopenmp -foffload=nvptx-none -o adaline_omp_gpu -lm

# CUDA
nvcc -O3 adaline_cuda.cu -o adaline_cuda
```

---

## Execução

Cada binário aceita parâmetros opcionais: `./prog [n_samples] [n_features] [epochs] [blocksize]` (blocksize só para CUDA). Valores padrão usados nas medições:
- `n_samples = 200000`
- `n_features = 128`
- `epochs = 20`
- `blocksize = 256` (CUDA)

Exemplos:
```bash
./adaline_seq
./adaline_omp_cpu
./adaline_omp_gpu
./adaline_cuda
```

---

## Configuração dos testes de desempenho

- Ambiente: WSL2, GPU RTX 2060, driver 565 / CUDA 12.7.
- Métrica: `/usr/bin/time -p` em 10 execuções consecutivas; logs em `resp/`.
- Dataset: gerado aleatoriamente a cada execução (parâmetros padrão acima).

---

## Resultados experimentais (médias de 10 execuções)

| Versão                  | Tempo médio (s) | Speedup vs seq. |
|-------------------------|-----------------|-----------------|
| Sequencial              | **16,40**       | 1,0×            |
| OpenMP CPU – 1 thread   | 12,59           | 1,30×           |
| OpenMP CPU – 2 threads  | 9,53            | 1,72×           |
| OpenMP CPU – 4 threads  | 7,86            | 2,09×           |
| OpenMP CPU – 8 threads  | 8,03            | 2,04×           |
| OpenMP CPU – 16 threads | 8,30            | 1,98×           |
| OpenMP CPU – 32 threads | 8,10            | 2,02×           |
| OpenMP GPU (target)     | 28,04           | 0,59×           |
| CUDA                    | 1,33            | 12,31×          |

Valores completos: `resp/seq.txt`, `resp/omp_cpu*.txt`, `resp/omp_gpu.txt`, `resp/cuda.txt`.
