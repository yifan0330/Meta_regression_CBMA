import numpy as np
import torch
from train import learner_one_group, learner_multiple_group, cluster_one_group
import time
import os
from absl import app, flags, logging

FLAGS = flags.FLAGS
# parameters
flags.DEFINE_integer("spacing", 15, "spline spacing in x/y/z direction.")
flags.DEFINE_integer("n_group", 1, "number of studies per group.")
flags.DEFINE_integer("n_study", 25, "number of studies per group.")
flags.DEFINE_float("mu", 1, "Poisson rate.")
flags.DEFINE_list("mu_list", [1,2], "multiple poisson rate.")
flags.DEFINE_integer("n_experiment", 100, "the number of realization.")
flags.DEFINE_boolean("clustered", None, "whether the response is caused by bumped signal or uniformly distributed.")
flags.DEFINE_boolean("covariates", None, "whether the group covariates effect exists.")
flags.DEFINE_boolean("multiple_group", None, "whether data comes from multiple group.")
flags.DEFINE_boolean("penalty", None, "whether adding Firth-type penalty")

flags.DEFINE_string("gpus", "0", "Ids of GPUs where the program run.")
def main(argv):
    del argv  # Unused.  
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        device = 'cuda'
    else:
        device = 'cpu'
    logging.info("Device: {}".format(device))
    if FLAGS.multiple_group == False:
        train_model = learner_one_group(device=device)
        X = train_model.load_X(spacing=FLAGS.spacing, x_max=100, y_max=100)
        y = train_model.load_y(clustered=FLAGS.clustered, n_group=FLAGS.n_group, n_study=FLAGS.n_study, covariates=FLAGS.covariates, mu=FLAGS.mu)
        model = train_model.model(penalty=FLAGS.penalty)
        train = train_model.train(iter=500)
        begin = time.time()
        evaluate = train_model.evaluation(n_experiment=FLAGS.n_experiment)
        print('time elapsed: ', time.time() - begin)
    else:
        train_model = learner_multiple_group(device=device)
        X = train_model.load_X(spacing=FLAGS.spacing, x_max=100, y_max=100)
        y = train_model.load_y(clustered=FLAGS.clustered, n_group=FLAGS.n_group, n_study=FLAGS.n_study, covariates=FLAGS.covariates, mu_list=FLAGS.mu_list)
        model = train_model.model(penalty=FLAGS.penalty)
        train = train_model.train(iter=500)
        begin = time.time()
        evaluate = train_model.evaluation(n_experiment=FLAGS.n_experiment)
        print('time elapsed: ', time.time() - begin)
        
if __name__ == "__main__":
    app.run(main)