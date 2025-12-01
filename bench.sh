#!/bin/bash

N=10   # número de execuções

# echo "Benchmarking SEQ:"
# for i in $(seq 1 $N); do
#     time ./adaline_seq > /dev/null
# done

# echo "-----------------------------"

echo "Benchmarking OMP CPU:"
for i in $(seq 1 $N); do
    time ./adaline_omp_cpu > /dev/null
done

echo "-----------------------------"

# echo "Benchmarking OMP GPU:"
# for i in $(seq 1 $N); do
#     time ./adaline_omp_gpu > /dev/null
done
