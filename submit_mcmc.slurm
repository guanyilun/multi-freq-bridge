#!/bin/bash
#PBS -N mcmc_ajay_baseline
#PBS -o logs/mf_baseline.out
#PBS -e logs/mf_baseline.err
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=32
#PBS -q greenq

cd $PBS_O_WORKDIR
module load openmpi

# name=case1_1
name=config_ajay
prefix=baseline
# config=configs/${name}.yaml
odir=/fs/lustre/scratch/yguan/data/multi-freq-bridge/runs/${prefix}_${name}
cov=${odir}/cov
config=configs/${name}.yaml

# base_config=configs/${name}.yaml
# config=configs/_tmp_${name}.yaml
# sed "s/region_width.*/region_width: 2.5/g" ${base_config} > ${config}
# sed -i "s/smooth_total_pix.*/smooth_total_pix: 1.25/g" ${config}

echo "Making covariance matrix"
python cov_maker.py --config ${config} --cov ${cov} 

echo "Running MCMC"
mpirun -n 32 python run_mcmc.py --config ${config} --cov ${cov} --odir ${odir}