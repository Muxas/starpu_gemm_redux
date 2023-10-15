#!/bin/bash
#SBATCH --job-name=gemm_redux
#SBATCH --output=%j.out
#SBATCH --error=%j.err
### Time and memory limit for the job
#SBATCH --time=1:00:00
#SBATCH --mem=900G
### Number of nodes
#SBATCH -n 1
### Number of CPU cores
#SBATCH -c 64
### Number of GPUs
#SBATCH -G 8
### Partition name
#SBATCH -p ais-gpu

export STARPU_NCPU=8
export TOKENIZERS_PARALLELISM=false
export STARPU_NCUDA=8

set -x

#for sched in dm dmda dmdar
for sched in eager
do
    export STARPU_SCHED=${sched}
    for ncuda in $(seq 8)
    do
        export STARPU_NCUDA=${ncuda}
        echo "Check STARPU_RW|STARPU_COMMUTE data access mode"
        echo "STARPU_SCHED=${sched} STARPU_NCUDA=${ncuda} gemm_redux 256" \
            "1536 1536 32 100 50 0"
        ./build/gemm_redux 256 1536 1536 32 100 50 0
        echo
        echo "Check STARPU_REDUX data access mode"
        echo "STARPU_SCHED=${sched} STARPU_NCUDA=${ncuda} gemm_redux 256" \
            "1536 1536 32 100 50 1"
        ./build/gemm_redux 256 1536 1536 32 100 50 1
        echo
    done
done

