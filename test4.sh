#!/bin/bash
python run.py --gpus 2--spacing 5 --n_group 1 --n_study 50 --mu 0.01 --n_experiment 50 --clustered=False --covariates=True  --multiple_group=False --penalty=True
python run.py --gpus 2 --spacing 5 --n_group 1 --n_study 100 --mu 0.01 --n_experiment 50 --clustered=False --covariates=True  --multiple_group=False --penalty=True

