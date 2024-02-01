check:
	squeue -u zw2374

attach:
	srun --jobid $(id) --pty /bin/bash

clean:
	rm -f *.err *.out

index:
	sbatch encode.slurm

train:
	sbatch train.slurm

cancel:
	scancel  $(id)
