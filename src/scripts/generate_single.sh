#!/bin/bash

##############################
####        Params        ####
##############################

JOB_NAME=$1

TASKS=$2
NODES=$3
PPN=$4
THREADS=$5

DIMENSION_X=$6
DIMENSION_Y=$7

EXEC_FILENAME=$8

FLAGS="-np ${TASKS}"
if [[ ${THREADS} != 1 ]]; then
  FLAGS="${FLAGS} -npernode ${PPN}"
fi

################################
####        Template        ####
################################

cat <<EOF
#!/bin/bash

#Max VM size
#PBS -l pvmem=2G

# Max Wall time
#PBS -l walltime=0:01:00  		# Example, 1 minute

# How many nodes and tasks per node
#PBS -l nodes=${NODES}:ppn=8  			# ${NODES} nodes with ${PPN} tasks/node => ${TASKS} tasks

#Which Queue
#PBS -q parsys			 	# This is the only accessible queue for rbs

#PBS -N ${JOB_NAME}   			# Jobname - it gives the stdout/err filenames

### Merge std[out|err]
#PBS -k oe

#Change Working directory to SUBMIT directory
cd \$PBS_O_WORKDIR  			# THIS IS MANDATORY,  PBS Starts everything from \$HOME, one should change to submit directory

#OpenMP Threads
export OMP_NUM_THREADS=${THREADS}
# OMP_NUM_THREADS * ppn should be max 8 (the total number of node cores= 8).
# To use OpenMPI remember to include -fopenmp in compiler flags in order to activate OpenMP directives.

# Having modules mpiP and openmpi loaded i.e.
module load mpiP openmpi

mpirun ${FLAGS} ${EXEC_FILENAME} ${DIMENSION_X} ${DIMENSION_Y}				# That was compiled on front-end


# No need for -np -machinefile etc. MPI gets this information from PBS

EOF
