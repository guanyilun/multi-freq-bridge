#!/bin/bash
#PBS -N mcmc_ajay_case42
#PBS -o logs_ajay/mf_baseline.out
#PBS -e logs_ajay/mf_baseline.err
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=32
#PBS -q greenq

cd $PBS_O_WORKDIR
module load openmpi

# name=case1_1
name=case42 
prefix=baseline
# config=configs/${name}.yaml
odir=/fs/lustre/scratch/agill/run_ajay/${name}
cov=${odir}/cov
config=configs/${name}_ajay.yaml

# base_config=configs/${name}.yaml
# config=configs/_tmp_${name}.yaml
# sed "s/region_width.*/region_width: 1.8/g" ${base_config} > ${config}
# sed -i "s/smooth_total_pix.*/smooth_total_pix: 1.25/g" ${config}

echo "Making covariance matrix"
python cov_maker.py --config ${config} --cov ${cov} 

cp ${config} ${odir}/config.yaml

echo "Running MCMC"
mpirun -n 64 python run_mcmc.py --config ${config} --cov ${cov} --odir ${odir}
