#!/bin/bash
#PBS -N mcmc_cluster_only
#PBS -o logs/mcmc1.log
#PBS -e logs/mcmc1.err
#PBS -l walltime=24:00:00
#PBS -l nodes=2:ppn=32
#PBS -q greenq

cd $PBS_O_WORKDIR
module load openmpi

name=250312/test1
config=configs/config_ajay.yaml
odir=/fs/lustre/scratch/yguan/data/multi-freq-bridge/runs/${name}
cov=${odir}/cov

# config=configs/${name}.yaml
# base_config=configs/${name}.yaml
# config=configs/_tmp_${name}.yaml
# sed "s/region_width.*/region_width: 1.8/g" ${base_config} > ${config}
# sed -i "s/smooth_total_pix.*/smooth_total_pix: 1.25/g" ${config}

echo "Making covariance matrix"
python cov_maker.py --config ${config} --cov ${cov} 

cp ${config} ${odir}/config.yaml

echo "Running MCMC"
mpirun -n 64 python run_mcmc.py --config ${config} --cov ${cov} --odir ${odir}
