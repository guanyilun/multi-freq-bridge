#!/bin/bash
#PBS -N mf_mcmc
#PBS -o logs/mf.out
#PBS -e logs/mf.err
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=16
#PBS -q sandyq

# cd $PBS_O_WORKDIR
module load openmpi

name=test
config=configs/config_mcmc.yaml
odir=/fs/lustre/scratch/yguan/data/multi-freq-bridge/runs/${name}
cov=${odir}/cov

echo "Making covariance matrix"
python cov_maker.py --config ${config} --cov ${cov} 

echo "Running MCMC"
mpirun -n 16 python run_mcmc.py --config ${config} --cov ${cov} --odir ${odir}
