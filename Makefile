.DEFAULT_GOAL = mcmc_test

mcmc_test:
	# need to run `module load openmpi` first
	@if [ -f /fs/lustre/scratch/yguan/data/multi-freq-bridge/run_test.h5 ]; \
	then \
		rm /fs/lustre/scratch/yguan/data/multi-freq-bridge/run_test.h5; \
		mpirun -n 32 python run_mcmc.py; \
	else \
		mpirun -n 32 python run_mcmc.py; \
	fi

cov:
	python cov_maker.py --config configs/config_mcmc.yaml

check_cov_valid:
	python test_cov_valid.py