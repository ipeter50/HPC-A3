#!/bin/bash
#BSUB -q hpcintrogpu
#BSUB -o gpu1_output.dat
#BSUB -J gpu1
#BSUB -n 12
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes" 
module load cuda/10.0
numactl --cpunodebind=0

for mat_size in 100 200 300 500 700 800 1000 1500
do
	
	./matmult_f.nvcc gpu1 $mat_size $mat_size $mat_size
	
done
