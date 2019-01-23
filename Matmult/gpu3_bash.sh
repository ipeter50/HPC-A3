#!/bin/bash
#BSUB -q hpcintrogpu
#BSUB -o gpu3_bellow_output.dat
#BSUB -J gpu3_bellow
#BSUB -n 12
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes" 
module load cuda/10.0
numactl --cpunodebind=0

for mat_size in 500 1000 1500 2000 2500 3000 4000 5000
do
	./matmult_f.nvcc gpu3 $mat_size $mat_size $mat_size	
done
