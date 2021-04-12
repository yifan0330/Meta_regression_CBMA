#!/bin/bash
#$ -wd /well/nichols/users/pra123/Meta_regression_CBMA_GPU
#$ -P nichols.prjc
#$ -N Meta-regression
#$ -q gpu8.q
#$ -t 1-n:1
#$ -cwd

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"



#!/bin/bash

#$ -j y
#$ -N extract
#$ -P doherty.prjc -q short.qe
#$ -t 1-9146
#$ -o /well/doherty/projects/hang/processed_group0/extractionLogs
#$ -pe shmem 1
#$ -cwd

source ~/.bashrc 
conda activate raine_parsing
export OMP_NUM_THREADS=${NSLOTS:-1}

SECONDS=0
echo $(date +%d/%m/%Y\ %H:%M:%S)

cmdList="extract.sh"
cmd=$(sed -n ${SGE_TASK_ID}p $cmdList)
echo $cmd
bash -c "$cmd"

duration=$SECONDS
echo "CPU time $pheno: $(($duration / 60)) min $((duration % 60)) sec"
echo $(date +%d/%m/%Y\ %H:%M:%S)