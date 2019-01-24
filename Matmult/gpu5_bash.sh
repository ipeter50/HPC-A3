#!/bin/bash
#BSUB -q hpcintrogpu
#BSUB -o gpu5_4_output.dat
#BSUB -J gpu5_4
#BSUB -n 12
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes" 
module load cuda/10.0
numactl --cpunodebind=0

for mat_size in 512 1024 1504 2048 2496 3008 4000 5024 6016 8000 10016
do
	./matmult_f.nvcc gpu5 $mat_size $mat_size $mat_size
done
