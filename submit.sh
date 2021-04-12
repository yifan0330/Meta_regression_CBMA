#!/bin/bash
#$ -wd /well/nichols/users/pra123/Meta_regression_CBMA_GPU
#$ -cwd
#$ -P nichols.prjc
#$ -N Meta-regression
#$ -q gpu8.q
#$ -o logs/test.out 
#$ -e logs/test.err 
#$ -t 1-27:1
#$ -l gpu=1

source ~/.bashrc 
cd Meta_regression_CBMA_GPU
source /well/nichols/users/pra123/anaconda3/bin/activate torch


SECONDS=0
echo $(date +%d/%m/%Y\ %H:%M:%S)

cmdList="$1"
cmd=$(sed -n ${SGE_TASK_ID}p $cmdList)
echo "$cmd"
bash -c "$cmd"

duration=$SECONDS
echo "CPU time $pheno: $(($duration / 60)) min $((duration % 60)) sec"
echo $(date +%d/%m/%Y\ %H:%M:%S)

