from data_generation.Data_simulation import Simulation2D_X, Simulation2D_count, Simulation2D_cluster
from models.poisson_regression import GLMPoisson_one_group, GLMPoisson_multiple_group
import os
import torch
from absl import logging
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class learner_one_group(object):
    def __init__(self, device='cpu'):
        self.device = device

    def load_X(self, spacing=15,
                x_max=100,
                y_max=100):
        self.spacing = spacing
        simulated_X = Simulation2D_X(spacing)
        X = simulated_X.Design_matrix(x_max, y_max)
        # convert to pytorch tensor
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.x_max, self.y_max = x_max, y_max
        self.X = X

    def load_y(self, 
                clustered=False,
                n_group=1,
                n_study=25,
                covariates=False,
                mu=1,
                max_val=1,
                n_covariates=2):
        self.clustered = clustered
        self.n_group = n_group
        self.n_study = n_study
        self.covariates = covariates
        self.mu = mu
        self.max_val = max_val
        self.n_covariates = n_covariates
        self.n_voxel = self.X.shape[0]
        if clustered == False:
            simulated_y = Simulation2D_count(n_voxel=self.n_voxel, n_group=n_group, n_study=n_study, covariates=covariates)
            if self.covariates == False:
                y = simulated_y.Poisson_one_group(mu)
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
                self.y_t, self.Z = None, None
                return y
            else:
                y, y_t, Z = simulated_y.Poisson_one_group(mu, max_val, n_covariates)
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
                y_t = torch.tensor(y_t, dtype=torch.float32, device=self.device)
                Z = torch.tensor(Z, dtype=torch.float32, device=self.device)
                self.y_t, self.Z = y_t, Z
                return y, y_t, Z
        else:
            # fixed locations of cluster centers
            centers = [[20,25], [40, 25], [60,25], [80,25], 
                        [20,75], [40,75], [60,75], [80,75]]
            simulated_y = Simulation2D_cluster(centers, shape=(100,100), n_group=n_group, n_study=n_study, covariates=covariates)
            if self.covariates == False:
                y = simulated_y.Poisson_one_group(mu=mu, var=10, n_cluster_foci=[0,1,2,3], max_val=max_val, n_covariates=n_covariates)
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
                return y
            else:
                y, y_t, Z = simulated_y.Poisson_one_group(mu=mu, var=10, n_cluster_foci=[0,1,2,3], max_val=max_val, n_covariates=n_covariates)
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
                y_t = torch.tensor(y_t, dtype=torch.float32, device=self.device)
                Z = torch.tensor(Z, dtype=torch.float32, device=self.device)
                self.Z = Z
                return y, y_t, Z

    def model(self, penalty=True):
        self.penalty = penalty
        beta_dim = self.X.shape[1]
        if self.covariates == False:
            gamma_dim = None
        else: 
            gamma_dim = self.Z.shape[1]
        model = GLMPoisson_one_group(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=penalty)
        if 'cuda' in self.device:
            model = model.cuda()
        return model
    
    def train(self, iter=500, lr=0.1):
        self.iter = iter
        self.lr = lr
        model = self.model(penalty = self.penalty)
        # re-generate count y per realization
        if self.covariates == False:
            y = self.load_y(clustered=self.clustered, n_group=self.n_group, n_study=self.n_study, covariates=self.covariates, mu=self.mu, max_val=self.max_val, n_covariates=self.n_covariates)
            y_t, Z = None, None
        else: 
            y, y_t, Z = self.load_y(clustered=self.clustered, n_group=self.n_group, n_study=self.n_study, covariates=self.covariates, mu=self.mu, max_val=self.max_val, n_covariates=self.n_covariates)
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        prev_loss = torch.tensor(float('inf'))
        loss_diff = torch.tensor(float('inf'))
        step = 0
        while torch.abs(loss_diff) > 1e-6: 
            if step <= iter:
                def closure():
                    optimizer.zero_grad()
                    loss = model(self.X, y, Z, y_t)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                print("step {0}: loss {1}".format(step, loss))
                loss_diff = loss - prev_loss
                prev_loss = loss
                step += 1
            else:
                print('it did not converge \n')
                print('The difference of loss in the current and previous iteration is', loss_diff)
        if self.clustered == False:
            if self.covariates == False:
                return model.beta_linear.weight
            else:
                return model.beta_linear.weight, model.gamma_linear.weight
        else:
            if self.covariates == False:
                return model.beta_linear.weight, y
            else:
                return model.beta_linear.weight, model.gamma_linear.weight, y, y_t, Z
    
    def uniform_response_evaluation(self, n_experiment=100):
        actual_mu_element = torch.tensor([self.n_study*self.mu], dtype=torch.float32, device=self.device)
        actual_mu = torch.repeat_interleave(input=actual_mu_element, repeats=self.n_voxel, dim=0)
        actual_mu = torch.reshape(input=actual_mu, shape=(self.n_voxel, 1))
        actual_mu_per_study = actual_mu / self.n_study
        
        if self.covariates == False:
            actual_beta = torch.repeat_interleave(input=torch.log(actual_mu_element), repeats=self.X.shape[1], dim=0) # X * beta = eta; 
            actual_beta = torch.reshape(input=actual_beta, shape=(self.X.shape[1], 1))
            actual_eta = torch.log(actual_mu)
        else:
            # mu_t
            actual_mu_t_element = torch.tensor([self.n_voxel*self.mu], dtype=torch.float32, device=self.device)
            actual_mu_t = torch.repeat_interleave(input=actual_mu_t_element, repeats=self.n_study, dim=0)
            actual_mu_t = torch.reshape(input=actual_mu_t, shape=(self.n_study, 1))
            actual_mu_t_per_voxel = actual_mu_t / self.n_voxel
            # beta
            actual_mu_rate_element = torch.tensor([self.mu], dtype=torch.float32, device=self.device)
            actual_beta = torch.repeat_interleave(input=torch.log(actual_mu_rate_element), repeats=self.X.shape[1], dim=0) 
            actual_beta = torch.reshape(input=actual_beta, shape=(self.X.shape[1], 1))
            # gamma
            actual_gamma_element = torch.tensor([0], dtype=torch.float32, device=self.device)
            actual_gamma = torch.repeat_interleave(input=actual_gamma_element, repeats=self.Z.shape[1], dim=0)
            actual_gamma = torch.reshape(input=actual_gamma, shape=(self.Z.shape[1], 1))

        beta_output, gamma_output, eta_output, mu_per_study_output, mu_t_per_voxel_output = [], [], [], [], []
        for i in range(n_experiment):
            if self.covariates == False:
                estimated_beta = self.train(iter=self.iter, lr=self.lr).t().detach().clone()
                # eta = X * beta
                estimated_eta = torch.matmul(self.X, estimated_beta)
                # log(mu) = eta = X * beta => mu = exp(eta)
                estimated_mu = torch.exp(estimated_eta)
            else: 
                estimated_beta, estimated_gamma = self.train(iter=self.iter, lr=self.lr)
                estimated_beta, estimated_gamma = estimated_beta.t().detach().clone(), estimated_gamma.t().detach().clone()
                # mu^X = exp(X * beta); mu^Z = exp(Z * gamma)
                estimated_mu_X = torch.exp(torch.matmul(self.X, estimated_beta)) 
                estimated_mu_Z = torch.exp(torch.matmul(self.Z, estimated_gamma))
                # mu_g = mu^X * (sum_i mu_i^Z)
                estimated_mu = torch.sum(estimated_mu_Z) * estimated_mu_X 
                # mu^t = (sum_j mu_j^X) * mu^Z
                estimated_mu_t = torch.sum(estimated_mu_X) * estimated_mu_Z 
            # append tensor to list
            beta_output.append(estimated_beta) 
            mu_per_study_output.append(estimated_mu / self.n_study)
            if self.covariates == False: 
                eta_output.append(estimated_eta)
            else:
                gamma_output.append(estimated_gamma)
                mu_t_per_voxel_output.append(estimated_mu_t / self.n_voxel)
        # convert list to 3-dimensional tensor
        beta_output = torch.stack(beta_output)
        mu_per_study_output = torch.stack(mu_per_study_output)
        # save tensors to pt file
        filename = 'n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu) + '.pt' 
        if self.covariates == False:
            eta_output = torch.stack(eta_output)
            if self.penalty == False:
                torch.save(obj=beta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_no_covariates/one_group_no_covariate_no_penalty/beta/'+filename)
                torch.save(obj=mu_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_no_covariates/one_group_no_covariate_no_penalty/mu/'+filename)
                torch.save(obj=eta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_no_covariates/one_group_no_covariate_no_penalty/eta/'+filename)
            else:
                torch.save(obj=beta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_no_covariates/one_group_no_covariate_with_penalty/beta/'+filename)
                torch.save(obj=mu_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_no_covariates/one_group_no_covariate_with_penalty/mu/'+filename)
                torch.save(obj=eta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_no_covariates/one_group_no_covariate_with_penalty/eta/'+filename)
        else:
            gamma_output = torch.stack(gamma_output)
            mu_t_per_voxel_output = torch.stack(mu_t_per_voxel_output)
            if self.penalty == False:
                torch.save(obj=beta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_no_penalty/beta/'+filename)
                torch.save(obj=mu_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_no_penalty/mu/'+filename)
                torch.save(obj=gamma_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_no_penalty/gamma/'+filename)
                torch.save(obj=mu_t_per_voxel_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_no_penalty/mu_t/'+filename)
            else:
                torch.save(obj=beta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_with_penalty/beta/'+filename)
                torch.save(obj=mu_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_no_penalty/mu/'+filename)
                torch.save(obj=gamma_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_no_penalty/gamma/'+filename)
                torch.save(obj=mu_t_per_voxel_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/one_group_with_covariate_no_penalty/mu_t/'+filename)
        # analysis of simulation results
        if self.covariates == False: 
            eta_mean = torch.mean(input=eta_output, dim=0)
            eta_bias = eta_mean - actual_eta
            eta_var = torch.var(input=eta_output, dim=0)
            eta_MSE = eta_var + eta_bias**2
        else:
            # gamma
            gamma_mean = torch.mean(input=gamma_output, dim=0) # shape: [2, 1]
            gamma_bias = gamma_mean - actual_gamma
            gamma_var = torch.var(input=gamma_output, dim=0)
            gamma_MSE = gamma_var + gamma_bias**2
            # mu_t
            mu_t_per_voxel_mean = torch.mean(input=mu_t_per_voxel_output, dim=0)
            mu_t_per_voxel_bias = mu_t_per_voxel_mean - actual_mu_t_per_voxel
            mu_t_per_voxel_var = torch.var(input=mu_t_per_voxel_output, dim=0)
            mu_t_per_voxel_MSE = mu_t_per_voxel_var + mu_t_per_voxel_bias**2
        # mean: E(theta)
        beta_mean = torch.mean(input=beta_output, dim=0)
        mu_mean_per_study = torch.mean(input=mu_per_study_output, dim=0)
        # bias(theta) = E(theta) - theta
        beta_bias = beta_mean - actual_beta
        mu_bias_per_study = mu_mean_per_study - actual_mu_per_study
        # variance
        beta_var = torch.var(input=beta_output, dim=0)
        mu_var_per_study = torch.var(input=mu_per_study_output, dim=0)
        # MSE (theta) = Var(theta) + Bias(theta)^2
        beta_MSE = beta_var + beta_bias**2
        mu_MSE_per_study = mu_var_per_study + mu_bias_per_study**2
        # save bias/var/MSE info to summary file
        if self.covariates == False:
            if self.penalty == False:
                f = open('results/one_group_no_covariates/one_group_no_covariate_no_penalty/summary.txt', 'a')
            else: 
                f = open('results/one_group_no_covariates/one_group_no_covariate_with_penalty/summary.txt', 'a')
            f.write('settings: n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu) + '\n')
            f.write('eta: '+ str(torch.mean(eta_bias).item()) + ',' + str(torch.mean(eta_var).item()) + ',' + str(torch.mean(eta_MSE).item()) + '\n')
        else:
            if self.penalty == False:
                f = open('results/one_group_with_covariate_no_penalty/summary.txt', 'a')
            else: 
                f = open('results/one_group_with_covariate_with_penalty/summary.txt', 'a')
            f.write('settings: n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu) + '\n')
            f.write('gamma: '+ str(torch.mean(gamma_bias).item()) + ',' + str(torch.mean(gamma_var).item()) + ',' + str(torch.mean(gamma_MSE).item()) + '\n')
            f.write('mu_t: '+ str(torch.mean(mu_t_per_voxel_bias).item()) + ',' + str(torch.mean(mu_t_per_voxel_var).item()) + ',' + str(torch.mean(mu_t_per_voxel_MSE).item()) + '\n')
        f.write('beta: '+ str(torch.mean(beta_bias).item()) + ',' + str(torch.mean(beta_var).item())+ ',' + str(torch.mean(beta_MSE).item()) + '\n')
        f.write('mu: '+ str(torch.mean(mu_bias_per_study).item()) + ',' + str(torch.mean(mu_var_per_study).item()) + ',' + str(torch.mean(mu_MSE_per_study).item()) + '\n')
        f.write('------------------------------------------------------------------\n')
        f.close()
    
    def clustered_evaluation(self, n_experiment=100):
        actual_y_array, actual_y_t_array = [], []
        beta_output, mu_per_study_output, eta_output, gamma_output, mu_t_per_voxel_output = [], [], [], [], []
        for i in range(n_experiment):
            if self.covariates == False:
                estimated_beta, y = self.train(iter=self.iter, lr=self.lr)
                estimated_beta, y = estimated_beta.t().detach().clone(), y.t().detach().clone()
                # eta = X * beta
                estimated_eta = torch.matmul(self.X, estimated_beta)
                # log(mu) = eta = X * beta => mu = exp(eta)
                estimated_mu = torch.exp(estimated_eta)
            else: 
                estimated_beta, estimated_gamma, y, y_t, Z = self.train(iter=self.iter, lr=self.lr)
                estimated_beta, estimated_gamma = estimated_beta.t().detach().clone(), estimated_gamma.t().detach().clone()
                y, y_t = y.t().detach().clone(), y_t.t().detach().clone()
                # mu^X = exp(X * beta); mu^Z = exp(Z * gamma)
                estimated_mu_X = torch.exp(torch.matmul(self.X, estimated_beta)) 
                estimated_mu_Z = torch.exp(torch.matmul(self.Z, estimated_gamma))
                # mu_g = mu^X * (sum_i mu_i^Z)
                estimated_mu = torch.sum(estimated_mu_Z) * estimated_mu_X 
                # mu^t = (sum_j mu_j^X) * mu^Z
                estimated_mu_t = torch.sum(estimated_mu_X) * estimated_mu_Z 
            # append tensor to list
            actual_y_array.append(y)
            beta_output.append(estimated_beta) 
            mu_per_study_output.append(estimated_mu)
            if self.covariates == False: 
                eta_output.append(estimated_eta)
            else:
                actual_y_t_array.append(y_t)
                gamma_output.append(estimated_gamma)
                mu_t_per_voxel_output.append(estimated_mu_t / self.n_voxel)
        # convert list to 3-dimensional tensor
        actual_y_array = torch.stack(actual_y_array)
        beta_output = torch.stack(beta_output)
        mu_per_study_output = torch.stack(mu_per_study_output)
        # take the average parameter value over all the experiments
        estimated_beta = torch.mean(input=beta_output, dim=0)
        estimated_mu = torch.reshape(torch.mean(input=mu_per_study_output, dim=0), (self.x_max, self.y_max))
        estimated_mu = estimated_mu.detach().cpu().numpy().T
        actual_y = torch.reshape(torch.mean(input=actual_y_array, dim=0), (self.x_max, self.y_max))
        actual_y = actual_y.detach().cpu().numpy().T
        # summary
        if self.covariates == False:
            if self.penalty == False:
                f = open('results/clustered_one_group_no_covariate_no_penalty/summary.txt', 'a')
            else: 
                f = open('results/clustered_one_group_no_covariate_with_penalty/summary.txt', 'a')
            f.write('settings: n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu) + '\n')
            f.write('actual y: '+ str(np.min(actual_y).item()) + ',' + str(np.mean(actual_y).item()) + ',' + str(np.max(actual_y).item()) + '\n')
            f.write('estimated intensity: '+ str(np.min(estimated_mu).item()) + ',' + str(np.mean(estimated_mu).item()) + ',' + str(np.max(estimated_mu).item()) + '\n')
        # else:
        #     if self.penalty == False:
        #         f = open('results/one_group_with_covariate_no_penalty/summary.txt', 'a')
        #     else: 
        #         f = open('results/one_group_with_covariate_with_penalty/summary.txt', 'a')
        #     f.write('settings: n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu) + '\n')
        #     f.write('gamma: '+ str(torch.mean(gamma_bias).item()) + ',' + str(torch.mean(gamma_var).item()) + ',' + str(torch.mean(gamma_MSE).item()) + '\n')
        #     f.write('mu_t: '+ str(torch.mean(mu_t_per_voxel_bias).item()) + ',' + str(torch.mean(mu_t_per_voxel_var).item()) + ',' + str(torch.mean(mu_t_per_voxel_MSE).item()) + '\n')
            f.write('------------------------------------------------------------------\n')
            f.close()
        
        # save the estimated voxelwise intensity and parameter values to pt file
        filename_pt = 'n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu) + '.pt' 
        if self.covariates == False:
            eta_output = torch.stack(eta_output)
            if self.penalty == False:
                torch.save(obj=beta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_no_penalty/beta/'+filename_pt)
                torch.save(obj=mu_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_no_penalty/mu/'+filename_pt)
                torch.save(obj=eta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_no_penalty/eta/'+filename_pt)
            else:
                torch.save(obj=beta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_with_penalty/beta/'+filename_pt)
                torch.save(obj=mu_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_with_penalty/mu/'+filename_pt)
                torch.save(obj=eta_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_with_penalty/eta/'+filename_pt)
        # save the estimated voxelwise intensity to image
        filename_jpg = 'n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu) + '.jpg'
        fig=plt.figure(figsize=(16, 16))
        fig.add_subplot(1, 2, 1)
        plt.imshow(actual_y, cmap='gray_r')
        fig.add_subplot(1, 2, 2)
        plt.imshow(estimated_mu, cmap='gray_r')
        if self.penalty == False:
            plt.savefig('/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_no_penalty/images/' + filename_jpg)
        else:
            plt.savefig('/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/clustered_one_group_no_covariate_with_penalty/images/' + filename_jpg)

        return
    
    def evaluation(self, n_experiment=100):
        if self.clustered == False:
            self.uniform_response_evaluation(n_experiment)
        else:
            self.clustered_evaluation(n_experiment)
        return




class learner_multiple_group(object):
    def __init__(self, device='cpu'):
        self.device = device

    def load_X(self, spacing=15,
                x_max=100,
                y_max=100):
        self.spacing = spacing
        simulated_X = Simulation2D_X(spacing)
        X = simulated_X.Design_matrix(x_max, y_max)
        # convert to pytorch tensor
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.X = X
         
    def load_y(self,
                clustered=False,
                n_group=2,
                n_study=25, 
                covariates=False, 
                mu_list=[1,1],
                max_val=1,
                n_covariates=2):
        self.clustered = clustered
        self.covariates = covariates
        self.n_group = n_group
        self.n_study = n_study
        self.mu_list = mu_list
        self.max_val = max_val
        self.n_covariates = n_covariates
        self.n_voxel = self.X.shape[0]
        if clustered == False:
            simulated_count = Simulation2D_count(n_voxel=self.n_voxel, n_group=n_group, n_study=n_study, covariates=covariates)
        # study_covariates is always True
        y_g_array, y_t, Z = simulated_count.Poisson_multiple_group_covariates(self.mu_list, max_val, n_covariates)
        # convert to pytorch tensor
        y_t = torch.tensor(y_t, dtype=torch.float32, device=self.device)
        Z = torch.tensor(Z, dtype=torch.float32, device=self.device)
        y_g_array = torch.tensor(y_g_array, dtype=torch.float32, device=self.device)
        self.y_g_array = y_g_array
        self.y_t = y_t
        self.Z = Z

    def model(self, penalty=True):
        self.penalty = penalty
        beta_dim = self.X.shape[1]
        gamma_dim = self.Z.shape[1]
        model = GLMPoisson_multiple_group(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=penalty)
        if 'cuda' in self.device:
            model = model.cuda()
        return model

    def train(self, iter=100, lr=0.1):
        self.iter = iter
        self.lr = lr
        y = self.load_y(n_group=self.n_group, n_study=self.n_study, covariates=self.covariates, mu_list=self.mu_list, max_val=self.max_val, n_covariates=self.n_covariates)
        model = self.model(penalty = self.penalty)
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        prev_loss = torch.tensor(float('inf'))
        loss_diff = torch.tensor(float('inf'))
        step = 0
        while torch.abs(loss_diff) > 1e-6: 
            if step <= iter:
                def closure():
                    optimizer.zero_grad()
                    loss = model(self.X, self.Z, self.y_g_array, self.y_t)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                # print("step {0}: loss {1}".format(step, loss))
                loss_diff = loss - prev_loss
                prev_loss = loss
                step += 1
            else:
                print('it did not converge \n')
                print('The difference of loss in the current and previous iteration is', loss_diff)
        return model.beta1_linear.weight, model.beta2_linear.weight, model.gamma_linear.weight
    
    def evaluation(self, n_experiment):
        # beta
        mu_1, mu_2 = [float(i) for i in self.mu_list]
        mu_1_element = torch.tensor([mu_1], dtype=torch.float32, device=self.device) # log(y) = X * beta
        actual_beta1 = torch.repeat_interleave(input=torch.log(mu_1_element), repeats=self.X.shape[1], dim=0).reshape((self.X.shape[1], 1))
        mu_2_element = torch.tensor([mu_2], dtype=torch.float32, device=self.device) # log(y) = X * beta
        actual_beta2 = torch.repeat_interleave(input=torch.log(mu_2_element), repeats=self.X.shape[1], dim=0).reshape((self.X.shape[1], 1))
        # gamma
        actual_gamma_element = torch.tensor([0], dtype=torch.float32, device=self.device)
        actual_gamma = torch.repeat_interleave(input=actual_gamma_element, repeats=self.Z.shape[1], dim=0).reshape((self.Z.shape[1], 1))
        # mu per study
        actual_mu_group1_per_study = torch.repeat_interleave(input=mu_1_element, repeats=self.n_voxel, dim=0).reshape((self.n_voxel, 1))
        actual_mu_group2_per_study = torch.repeat_interleave(input=mu_2_element, repeats=self.n_voxel, dim=0).reshape((self.n_voxel, 1))
        # mu_t per voxel
        actual_mu_t_per_voxel = torch.repeat_interleave(input=mu_1_element+mu_2_element, repeats=self.n_study*2, dim=0).reshape((self.n_study*2, 1))
        # 100 realizations
        beta1_output, beta2_output, gamma_output, mu_group1_per_study_output, mu_group2_per_study_output, mu_t_per_voxel_output = [], [], [], [], [], []
        for i in range(n_experiment):
            estimated_beta1 = self.train(iter=self.iter, lr=self.lr)[0].t().detach().clone()
            estimated_beta2 = self.train(iter=self.iter, lr=self.lr)[1].t().detach().clone()
            estimated_gamma = self.train(iter=self.iter, lr=self.lr)[2].t().detach().clone()
            # mu_g^X = exp(X*beta_g): the vector of spatial spline effect of studies in group g
            estimated_log_mu_X_group1 = torch.matmul(self.X, estimated_beta1)
            estimated_mu_X_group1 = torch.exp(estimated_log_mu_X_group1)
            estimated_log_mu_X_group2 = torch.matmul(self.X, estimated_beta2)
            estimated_mu_X_group2 = torch.exp(estimated_log_mu_X_group2)
            # mu_g^Z = exp(Z_g * gamma): the vector of study-level covariates of studies in group g
            estimated_log_mu_Z = torch.matmul(self.Z, estimated_gamma)
            estimated_mu_Z = torch.exp(estimated_log_mu_Z)
            # mu_g = [sum_i mu_i^Z] * mu_g^X: the voxel-wise total intensity over all studies in group g
            mu_group1 = torch.sum(estimated_mu_Z[:self.n_study, :]) * estimated_mu_X_group1
            mu_group2 = torch.sum(estimated_mu_Z[self.n_study:, :]) * estimated_mu_X_group2
            mu_group1_per_study = mu_group1 / self.n_study # shape: (n_voxel, 1)
            mu_group2_per_study = mu_group2 / self.n_study
            # mu^t = [sum_j mu_j^X] * mu^Z: the total sum of intensity in study i regardless of foci location in the whole dataset
            estimated_mu_X = estimated_mu_X_group1 + estimated_mu_X_group2
            mu_t = torch.sum(estimated_mu_X) * estimated_mu_Z
            mu_t_per_voxel = mu_t / self.n_voxel
            # append tensor to list
            beta1_output.append(estimated_beta1) 
            beta2_output.append(estimated_beta2) 
            gamma_output.append(estimated_gamma)
            mu_group1_per_study_output.append(mu_group1_per_study)
            mu_group2_per_study_output.append(mu_group2_per_study)
            mu_t_per_voxel_output.append(mu_t_per_voxel)
        # convert list to 3-dimensional tensor
        beta1_output = torch.stack(beta1_output)
        beta2_output = torch.stack(beta2_output)
        gamma_output = torch.stack(gamma_output)
        mu_group1_per_study_output = torch.stack(mu_group1_per_study_output)
        mu_group2_per_study_output = torch.stack(mu_group2_per_study_output)
        mu_t_per_voxel_output = torch.stack(mu_t_per_voxel_output)
        
        # save tensors to pt file
        filename = 'n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_' + str(self.mu_list) + '.pt' 
        if mu_1 != mu_2:
            if self.penalty == False:
                torch.save(obj=beta1_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_no_penalty/beta1/'+filename)
                torch.save(obj=beta2_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_no_penalty/beta2/'+filename)
                torch.save(obj=gamma_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_no_penalty/gamma/'+filename)
                torch.save(obj=mu_group1_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_no_penalty/mu_group1/'+filename)
                torch.save(obj=mu_group2_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_no_penalty/mu_group2/'+filename)
                torch.save(obj=mu_t_per_voxel_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_no_penalty/mu_t/'+filename)
            else:
                torch.save(obj=beta1_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_with_penalty/beta1/'+filename)
                torch.save(obj=beta2_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_with_penalty/beta2/'+filename)
                torch.save(obj=gamma_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_with_penalty/gamma/'+filename)
                torch.save(obj=mu_group1_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_with_penalty/mu_group1/'+filename)
                torch.save(obj=mu_group2_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_with_penalty/mu_group2/'+filename)
                torch.save(obj=mu_t_per_voxel_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_with_covariate_with_penalty/mu_t/'+filename)
        else:
            if self.penalty == False:
                torch.save(obj=beta1_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_no_penalty/beta1/'+filename)
                torch.save(obj=beta2_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_no_penalty/beta2/'+filename)
                torch.save(obj=gamma_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_no_penalty/gamma/'+filename)
                torch.save(obj=mu_group1_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_no_penalty/mu_group1/'+filename)
                torch.save(obj=mu_group2_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_no_penalty/mu_group2/'+filename)
                torch.save(obj=mu_t_per_voxel_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_no_penalty/mu_t/'+filename)
            else:
                torch.save(obj=beta1_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_with_penalty/beta1/'+filename)
                torch.save(obj=beta2_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_with_penalty/beta2/'+filename)
                torch.save(obj=gamma_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_with_penalty/gamma/'+filename)
                torch.save(obj=mu_group1_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_with_penalty/mu_group1/'+filename)
                torch.save(obj=mu_group2_per_study_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_with_penalty/mu_group2/'+filename)
                torch.save(obj=mu_t_per_voxel_output, f='/well/nichols/users/pra123/Meta_regression_CBMA_GPU/results/multiple_group_equal_response_with_penalty/mu_t/'+filename)
        # mean
        beta1_mean = torch.mean(input=beta1_output, dim=0)
        beta2_mean = torch.mean(input=beta2_output, dim=0)
        gamma_mean = torch.mean(input=gamma_output, dim=0)
        mu_group1_per_study_mean = torch.mean(input=mu_group1_per_study_output, dim=0)
        mu_group2_per_study_mean = torch.mean(input=mu_group2_per_study_output, dim=0)
        mu_t_per_voxel_mean = torch.mean(input=mu_t_per_voxel_output, dim=0)
        # variance
        beta1_var = torch.var(input=beta1_output, dim=0)
        beta2_var = torch.var(input=beta2_output, dim=0)
        gamma_var = torch.var(input=gamma_output, dim=0)
        mu_group1_per_study_var = torch.var(input=mu_group1_per_study_output, dim=0)
        mu_group2_per_study_var = torch.var(input=mu_group2_per_study_output, dim=0)
        mu_t_per_voxel_var = torch.var(input=mu_t_per_voxel_output, dim=0)
        
        if mu_1 == mu_2:
            # two-sample wald statistics for beta and mu
            beta1_var = torch.var(input=beta1_output, dim=0)
            beta2_var = torch.var(input=beta2_output, dim=0)
            W_beta = (beta1_mean - beta2_mean)**2 / (beta1_var/n_experiment + beta2_var/n_experiment)
            W_beta = W_beta.detach().cpu().numpy()
            p_value_beta = np.minimum(stats.chi2.sf(np.abs(W_beta), 1)*2, 1) #two-tailed
            W_mu = (mu_group1_per_study_mean - mu_group2_per_study_mean)**2 / (mu_group1_per_study_var/n_experiment + mu_group2_per_study_var/n_experiment)
            W_mu = W_mu.detach().cpu().numpy()
            p_value_mu = np.minimum(stats.chi2.sf(np.abs(W_mu), 1)*2, 1)
            # save bias/var/MSE info to summary file
            if self.penalty == False:
                f = open('results/multiple_group_equal_response_no_penalty/summary.txt', 'a')
            else:
                f = open('results/multiple_group_equal_response_with_penalty/summary.txt', 'a')
            f.write('settings: n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_list_' + str(self.mu_list) + '\n')
            f.write('The p-value range of wald statistics of beta difference in two groups is: [' + str(np.min(p_value_beta)) + ' , ' + str(np.max(p_value_beta)) + ' ]\n')
            f.write('reject null hypothesis at ' + str(np.count_nonzero(p_value_beta < 0.05)) + ' out of ' + str(self.X.shape[1]) + ' element in beta. \n')
            f.write('The p-value range of wald statistics of mu_per_study difference in two groups: [' + str(np.min(p_value_mu)) + ' , ' + str(np.max(p_value_mu)) + ' ]\n')
            f.write('reject null hypothesis at ' + str(np.count_nonzero(p_value_mu < 0.05)) + ' voxels. \n')
            f.write('------------------------------------------------------------------\n')
            f.close()
        else:
            # bias(theta) = E(theta) - theta
            beta1_bias = beta1_mean - actual_beta1
            beta2_bias = beta2_mean - actual_beta2
            gamma_bias = gamma_mean - actual_gamma
            mu_group1_per_study_bias = mu_group1_per_study_mean - actual_mu_group1_per_study
            mu_group2_per_study_bias = mu_group2_per_study_mean - actual_mu_group2_per_study
            mu_t_per_voxel_bias = mu_t_per_voxel_mean - actual_mu_t_per_voxel
            # MSE (theta) = Var(theta) + Bias(theta)^2
            beta1_MSE = beta1_var + beta1_bias**2
            beta2_MSE = beta2_var + beta2_bias**2
            gamma_MSE = gamma_var + gamma_bias**2
            mu_group1_per_study_MSE = mu_group1_per_study_var + mu_group1_per_study_bias**2
            mu_group2_per_study_MSE = mu_group2_per_study_var + mu_group2_per_study_bias**2
            mu_t_per_voxel_MSE = mu_t_per_voxel_var + mu_t_per_voxel_bias**2
            # save bias/var/MSE info to summary file
            if self.penalty == False:
                f = open('results/multiple_group_with_covariate_no_penalty/summary.txt', 'a')
            else:
                f = open('results/multiple_group_with_covariate_with_penalty/summary.txt', 'a')
            f.write('settings: n_study_'+ str(self.n_study) + '_spacing_' + str(self.spacing) + '_mu_list_' + str(self.mu_list) + '\n')
            f.write('beta1: '+ str(torch.mean(beta1_bias).item()) + ',' + str(torch.mean(beta1_var).item()) + ',' + str(torch.mean(beta1_MSE).item()) + '\n')
            f.write('beta2: '+ str(torch.mean(beta2_bias).item()) + ',' + str(torch.mean(beta2_var).item()) + ',' + str(torch.mean(beta2_MSE).item()) + '\n')
            f.write('gamma: '+ str(torch.mean(gamma_bias).item()) + ',' + str(torch.mean(gamma_var).item()) + ',' + str(torch.mean(gamma_MSE).item()) + '\n')
            f.write('mu_group1_per_study: '+ str(torch.mean(mu_group1_per_study_bias).item()) + ',' + str(torch.mean(mu_group1_per_study_var).item()) + ',' + str(torch.mean(mu_group1_per_study_MSE).item()) + '\n')
            f.write('mu_group2_per_study: '+ str(torch.mean(mu_group2_per_study_bias).item()) + ',' + str(torch.mean(mu_group2_per_study_var).item()) + ',' + str(torch.mean(mu_group2_per_study_MSE).item()) + '\n')
            f.write('mu_t_per_voxel: '+ str(torch.mean(mu_t_per_voxel_bias).item()) + ',' + str(torch.mean(mu_t_per_voxel_var).item()) + ',' + str(torch.mean(mu_t_per_voxel_MSE).item()) + '\n')
            f.write('------------------------------------------------------------------\n')
            f.close()



class cluster_one_group(object):
    def __init__(self, device='cpu'):
        self.device = device

    def load_data(self, centers=[[10,10],[20,10],[20,10],[30,30],[40,40],[50,50],[60,60],[70,70],[80,80]], 
                var = 20,
                spacing=15, 
                x_max=100,
                y_max=100,
                n_group=1,
                n_study=25, 
                covariates=False,
                r=42,
                cluster_foci=[0,1,2,3],
                prob=[0.35, 0.5, 0.1, 0.05],
                max_val=1,
                n_covariates=2):
        self.covariates = covariates
        self.n_study = n_study
        self.prob = prob
        self.spacing = spacing
        simulated_X = Simulation2D_X(spacing)
        X = simulated_X.Design_matrix(x_max, y_max)
        self.shape = (x_max, y_max)
        simulated_count = Simulation2D_cluster(centers=centers, var=20, shape=self.shape, n_group=n_group, n_study=n_study, covariates=covariates, r=r)
        if self.covariates == False:
            y = simulated_count.one_group(cluster_foci, prob, max_val, n_covariates)
            self.y_t = None
            self.Z = None
        else: 
            y, y_t, Z = simulated_count.one_group(cluster_foci, prob, max_val, n_covariates)
            y_t = torch.tensor(y_t, dtype=torch.float32, device=self.device)
            Z = torch.tensor(Z, dtype=torch.float32, device=self.device)
            self.y_t = y_t
            self.Z = Z
        # convert to pytorch tensor
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.X = X
        self.y = y

    def model(self, penalty=True):
        self.penalty = penalty
        beta_dim = self.X.shape[1]
        if self.covariates == False:
            gamma_dim = None
        else: 
            gamma_dim = self.Z.shape[1]
        model = GLMPoisson_one_group(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=penalty)
        if 'cuda' in self.device:
            model = model.cuda()
        self.model = model
    
    def train(self, iter=100, lr=0.1):
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr)
        for step in range(iter):
            def closure():
                optimizer.zero_grad()
                loss = self.model(self.X, self.y, Z=self.Z, y_t=self.y_t)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            #print("step {0}: loss {1}".format(step, loss))
    
    def evaluation(self):
        # estimated voxelwise mean
        estimated_beta = self.model.beta_linear.weight
        estimated_log_mu_X = torch.matmul(self.X, estimated_beta.t())
        estimated_mu_X = torch.exp(estimated_log_mu_X)
        estimated_mu_X_array = estimated_mu_X.cpu().detach().numpy()
        estimated_mu_X_array = estimated_mu_X_array.reshape(self.shape)
        # one group, no covariate effect
        if self.covariates == False:
            file_name = 'plots/spacing_' + str(self.spacing) + '_n_study_' + str(self.n_study) + '_probability_' + str(self.prob) + '.png'
            plt.imsave(file_name, estimated_mu_X_array, cmap='Greys')
            print('(one group, no covariate effect, penalty:{0}) basis spline spacing = {1}; n_study={2}; probability={3}: estimated mu in [{4}, {5}]'.format(self.penalty, self.spacing, self.n_study, self.prob, np.min(estimated_mu_X_array), np.max(estimated_mu_X_array)))
            print(torch.min(self.y), torch.max(self.y))
        else:
            estimated_gamma = self.model.gamma_linear.weight
            estimated_log_mu_Z = torch.matmul(self.Z, estimated_gamma.t())
            estimated_mu_Z = torch.exp(estimated_log_mu_Z)
            estimated_mu_Z_array = estimated_mu_Z.cpu().detach().numpy()
            file_name = 'plots/group_effect_spacing_' + str(self.spacing) + '_n_study_' + str(self.n_study) + '_probability_' + str(self.prob) + '.png'
            plt.imsave(file_name, estimated_mu_Z_array, cmap='Greys')
            print('(one group, no covariate effect, penalty:{0}) basis spline spacing = {1}; n_study={2}; probability={3}: estimated mu in [{4}, {5}]'.format(self.penalty, self.spacing, self.n_study, self.prob, np.min(estimated_mu_X_array), np.max(estimated_mu_X_array)))
            
        return 