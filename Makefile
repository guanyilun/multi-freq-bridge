.DEFAULT_GOAL = mcmc_test

mcmc_test:
	@if [ -f /fs/lustre/scratch/yguan/data/multi-freq-bridge/run_test.h5 ]; \
	then \
		rm /fs/lustre/scratch/yguan/data/multi-freq-bridge/run_test.h5; \
		mpirun -n 30 python run_mcmc.py; \
	else \
		mpirun -n 30 python run_mcmc.py; \
	fi

cov:
	python cov_maker.py