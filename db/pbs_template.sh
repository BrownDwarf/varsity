#PBS -N JOBNAME_HERE
#PBS -l select=1:ncpus=16:model=san
#PBS -l walltime=8:00:00

# module load ???

# By default, PBS executes your job from your home directory.
# However, you can use the environment variable
# PBS_O_WORKDIR to change to the directory where
# you submitted your job.

cd $PBS_O_WORKDIR

conda activate varsity
$varsity/code/star_marley_beta.py --samples=N_SAMPLES --incremental_save=INC_SAVE
