#!/bin/bash
#BSUB -q hpcintrogpu
#BSUB -o lib_output.dat
#BSUB -n 12
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes" 
module load cuda/10.0
numactl --cpunodebind=0

for mat_size in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
	for num_threads in 1 2 4 8 10 12 16 24
	do
		MKL_NUM_THREADS=$num_threads ./matmult_f.nvcc lib $mat_size $mat_size $mat_size
	done
done
