#!/bin/bash
#BSUB -q hpcintrogpu
#BSUB -o 2gpu_output.dat
#BSUB -J 2_gpu
#BSUB -n 12
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes" 
module load cuda/10.0
numactl --cpunodebind=0

for mat_size in 100 200 300 700 800 1000 2000 4000 8000 10000
do
	
	nvprof poisson.nvcc $mat_size 
done
