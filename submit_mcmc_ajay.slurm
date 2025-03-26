#!/bin/bash
#PBS -N mcmc_ajay_case66
#PBS -o logs_ajay/mf_baseline.out
#PBS -e logs_ajay/mf_baseline.err
#PBS -l walltime=5:00:00
#PBS -l nodes=3:ppn=16
#PBS -q sandyq

cd $PBS_O_WORKDIR
module load miniconda
module load intel
module load openmpi

name=case66
prefix=baseline
# config=configs/${name}.yaml
# odir=/home/agill/multi-freq-bridge/run_ajay/${name}
odir=/fs/lustre/scratch/agill/run_ajay/${name}
cov=${odir}/cov
config=configs/${name}_ajay.yaml

# base_config=configs/${name}.yaml
# config=configs/_tmp_${name}.yaml
# sed "s/region_width.*/region_width: 2.5/g" ${base_config} > ${config}
# sed -i "s/smooth_total_pix.*/smooth_total_pix: 1.25/g" ${config}

echo "Making covariance matrix"
python cov_maker.py --config ${config} --cov ${cov} 

echo "Running MCMC"
mpirun -n 48 python run_mcmc.py --config ${config} --cov ${cov} --odir ${odir}
