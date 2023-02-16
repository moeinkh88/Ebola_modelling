#!/bin/bash
#SBATCH --job-name=julia_serial
#SBATCH --output=array_job_out_%A_%a.txt
#SBATCH --error=array_job_err_%A_%a.txt
#SBATCH --account=project_2007115
#SBATCH --partition=small
#SBATCH --time=02:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=5G
#SBATCH --array=216-300

module load julia
srun julia Best_diving_time_CSC.jl ${SLURM_ARRAY_TASK_ID}
