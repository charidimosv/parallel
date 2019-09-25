#!/bin/bash

####################################
####        Parse Params        ####
####################################

createScripts=0
runScripts=0
useMpi=0
useOpenMP=0

while [[ $# -gt 0 ]]; do
  case $1 in
  -mpi)
    useMpi=1
    shift
    ;;

  -op)
    useOpenMP=1
    shift
    ;;

  -cs)
    createScripts=1
    shift
    ;;

  -rs)
    runScripts=1
    shift
    ;;

  *)
    shift
    ;;
  esac
done

##############################
####        Config        ####
##############################

nodesList='1 2 8 16 20'
ppnList='2 4 8'
threadsList='1 2 4 8 16 32'

#initialDimension_x=80
#initialDimension_y=64

initialDimension_x=80
initialDimension_y=80

maxDimensionDoubling=3

mpiExecFileName="mpi_heat2D.x"
openmpExecFileName="mpi_heat2D.x"

scriptFolderName="scripts"

########################################
####        Every occurrence        ####
########################################

if [[ (${createScripts} == 1 || ${runScripts} == 1) && (${useMpi} == 1 || ${useOpenMP} == 1) ]]; then
  mkdir -p ${scriptFolderName}
fi

for nodes in ${nodesList}; do
  for ppn in ${ppnList}; do
    tasks=$((nodes * ppn))
    if [[ ${tasks} != 1 && ${tasks} != 4 && ${tasks} != 16 && ${tasks} != 64 && ${tasks} != 128 && ${tasks} != 160 ]]; then
      continue
    fi

    for threads in ${threadsList}; do
      printf "Ts${tasks}: N${nodes} x P${ppn} x T${threads}\n"

      currentDimension_x=${initialDimension_x}
      currentDimension_y=${initialDimension_y}

      for ((i = 0; i < maxDimensionDoubling; i++)); do
        printf "\t${currentDimension_x}x${currentDimension_y}\n"

        jobName="Ts${tasks}__N${nodes}_P${ppn}_T${threads}__X${currentDimension_x}_Y${currentDimension_y}"

        mpiJobName="mpi__${jobName}"
        mpiScriptName="${scriptFolderName}/${mpiJobName}.sh"

        openmpJobName="openmp__${jobName}"
        openmpScriptName="${scriptFolderName}/${openmpJobName}.sh"

        if [[ ${createScripts} == 1 ]]; then
          if [[ ${useMpi} == 1 && ${threads} == 1 ]]; then
            ./generate_single.sh ${mpiJobName} ${tasks} ${nodes} ${ppn} ${threads} ${currentDimension_x} ${currentDimension_y} ${mpiExecFileName} >${mpiScriptName}
            chmod 770 ${mpiScriptName}
            printf "\t\tMPI Script created\n"
          fi

          if [[ ${useOpenMP} == 1 ]]; then
            ./generate_single.sh ${openmpJobName} ${tasks} ${nodes} ${ppn} ${threads} ${currentDimension_x} ${currentDimension_y} ${openmpExecFileName} >${openmpScriptName}
            chmod 770 ${openmpScriptName}
            printf "\t\tOpenMP Script created\n"
          fi
        fi

        if [[ ${runScripts} == 1 ]]; then
          if [[ ${useMpi} == 1 && ${threads} == 1 ]]; then
            qsub ${mpiScriptName}
            printf "\t\tMPI Script ran\n"
          fi

          if [[ ${useOpenMP} == 1 ]]; then
            qsub ${openmpScriptName}
            printf "\t\tOpenMP Script ran\n"
          fi
        fi

        currentDimension_x=$((2 * currentDimension_x))
        currentDimension_y=$((2 * currentDimension_y))
      done

    done
  done
done
