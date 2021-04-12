#!/bin/bash
python run.py --gpus 0 --spacing 15 --n_group 2 --n_study 25 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False

python run.py --gpus 0 --spacing 15 --n_group 2 --n_study 100 --mu_list 1,2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 15 --n_group 2 --n_study 100 --mu_list 0.1,0.2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 15 --n_group 2 --n_study 100 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False


python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 25 --mu_list 1,2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 25 --mu_list 0.1,0.2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 25 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False

python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 50 --mu_list 1,2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 50 --mu_list 0.1,0.2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 50 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False

python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 100 --mu_list 1,2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 100 --mu_list 0.1,0.2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 10 --n_group 2 --n_study 100 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False


python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 25 --mu_list 1,2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 25 --mu_list 0.1,0.2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 25 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False

python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 50 --mu_list 1,2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 50 --mu_list 0.1,0.2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 50 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False

python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 100 --mu_list 1,2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 100 --mu_list 0.1,0.2 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False
python run.py --gpus 0 --spacing 5 --n_group 2 --n_study 100 --mu_list 0.01,0.02 --n_experiment 100 --clustered=False --covariates=True  --multiple_group=True --penalty=False