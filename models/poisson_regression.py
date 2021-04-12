import torch

class GLMPoisson_one_group(torch.nn.Module):
    def __init__(self, beta_dim, gamma_dim=None, covariates=False, penalty=True):
        super().__init__()
        self.covariates = covariates
        self.penalty = penalty
        # beta
        self.beta_dim = beta_dim
        self.beta_linear = torch.nn.Linear(self.beta_dim, 1, bias=False)
        # initialization for beta
        torch.nn.init.uniform_(self.beta_linear.weight, a=-1, b=1)
        # gamma 
        if self.covariates == True:
            self.gamma_dim = gamma_dim
            self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False)
            torch.nn.init.constant_(self.gamma_linear.weight, 1)
        else:
             self.gamma_dim = None

    def Fisher_information(self, X, mu_X, Z=None, mu_Z=None):
        if self.covariates == False:
            mu_X_sqrt = torch.sqrt(mu_X)
            X_star = mu_X_sqrt * X # X* = W^(1/2) X
            I = torch.mm(X_star.t(), X_star)
        else:
            # the voxelwise total intensity over all studies in group g
            # mu_g = [sum_i mu_i^Z] * mu_g^X
            mu = torch.sum(mu_Z) * mu_X
            mu_sqrt = torch.sqrt(mu)
            # the total sum of intensity in study i regardless of foci location in the whole dataset
            # mu_t = [sum_j mu_j^X] mu^Z
            mu_t = torch.sum(mu_X) * mu_Z
            mu_t_sqrt = torch.sqrt(mu_t)
            # Fisher Information matrix
            # block matrix at top left: I(beta)
            X_star = mu_sqrt * X # X* = W^(1/2) X
            I_beta = torch.mm(X_star.t(), X_star)
            # block matrix of the cross term: I(beta, gamma)
            I_cross_term = torch.mm(torch.mm(X.t(), mu_X), torch.mm(mu_Z.t(), Z))
            # block matrix at bittom right: I(gamma)
            Z_star = mu_t_sqrt * Z # Z* = V^(1/2) Z
            I_gamma = torch.mm(Z_star.t(), Z_star) # ZVZ = (V^(1/2) Z)^T (V^(1/2) Z)
            # concatenate to the Fisher Information matrix
            I_top = torch.cat((I_beta, I_cross_term), axis=1) # shape: (P, P+R)
            I_bottom = torch.cat((I_cross_term.t(), I_gamma), axis=1) # shape(R, P+R)
            I = torch.cat((I_top, I_bottom), axis=0) # shape: (P+R, P+R)
        return I

    def forward(self, X, y, Z=None, y_t=None):
        # mu^X = exp(X * beta)
        log_mu_X = self.beta_linear(X)
        mu_X = torch.exp(log_mu_X)
        if self.covariates == False:
            #l = sum_j [Y_j * log(mu_j) - mu_j - log(y_j!)]
            # print(log_mu_X)
            l = torch.sum(torch.mul(y, log_mu_X)) - torch.sum(mu_X) - torch.sum(torch.lgamma(y+1))
            # print(-l)
            Z, mu_Z = None, None
            #print(l)
        else:
            # mu^Z = exp(Z * gamma)
            log_mu_Z = self.gamma_linear(Z)
            mu_Z = torch.exp(log_mu_Z)
            # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
            l = torch.sum(torch.mul(y, log_mu_X)) + torch.sum(torch.mul(y_t, log_mu_Z)) - torch.sum(mu_X) * torch.sum(mu_Z) 
        # Firth-type penalty
        if self.penalty == True:
            I = self.Fisher_information(X, mu_X, Z, mu_Z)
            I_eigens = torch.symeig(I, eigenvectors=True)[0]
            log_I_eigens = torch.log(I_eigens)
            log_det_I = torch.sum(log_I_eigens)
            # if log_det_I.isnan() == True:
            #     print(I_eigens)
            #     exit()
            
            l = l + 1/2 * log_det_I
        return -l

class GLMPoisson_multiple_group(torch.nn.Module):
    def __init__(self, beta_dim, gamma_dim, covariates=False, penalty=True):
        super().__init__()
        self.covariates = covariates
        self.penalty = penalty
        # initialization: beta and gamma
        self.beta_dim = beta_dim
        self.gamma_dim = gamma_dim
        self.beta1_linear = torch.nn.Linear(self.beta_dim, 1, bias=False)
        self.beta2_linear = torch.nn.Linear(self.beta_dim, 1, bias=False)
        self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False)
        torch.nn.init.uniform_(self.beta1_linear.weight, a=-1, b=1)
        torch.nn.init.uniform_(self.beta2_linear.weight, a=-1, b=1)
        torch.nn.init.constant_(self.gamma_linear.weight, 0)
    
    def Fisher_Information(self, X, Z, mu_X_group1, mu_X_group2, mu_Z_group1, mu_Z_group2):
        P = X.shape[1]
        mu_Z = torch.cat((mu_Z_group1, mu_Z_group2), dim=0)
        # the voxelwise total intensity over all studies in group g
        # mu_g = [sum_i mu_i^Z] * mu_g^X
        mu_group1 = torch.sum(mu_Z_group1) * mu_X_group1
        mu_group2 = torch.sum(mu_Z_group2) * mu_X_group2
        # the total sum of intensity in study i regardless of foci location in the whole dataset
        # mu_t = [sum_j mu_j^X] mu^Z
        mu_X = mu_X_group1 + mu_X_group2
        mu_t = torch.sum(mu_X) * mu_Z
        mu_t_sqrt = torch.sqrt(mu_t)
        # Fisher Information matrix
        # block matrix at top left: I(beta)
        mu_group1_sqrt = torch.sqrt(mu_group1)
        mu_group2_sqrt = torch.sqrt(mu_group2)
        X_star_group1 = mu_group1_sqrt * X # X* = W^(1/2) X
        # XWX = (W^(1/2) X)^T (W^(1/2) X) 
        XWX_group1 = torch.mm(X_star_group1.t(), X_star_group1) 
        X_star_group2 = mu_group2_sqrt * X # X* = W^(1/2) X
        XWX_group2 = torch.mm(X_star_group2.t(), X_star_group2)
        zero_matrix_p = torch.zeros(size=(P, P), device='cuda')
        I_beta_top = torch.cat((XWX_group1, zero_matrix_p), axis=1)
        I_beta_bottom = torch.cat((zero_matrix_p, XWX_group2), axis=1)
        I_beta = torch.cat((I_beta_top, I_beta_bottom), axis=0)
        # block matrix of the cross term: I(beta, gamma)
        cross_derivative_group1 = torch.mm(torch.mm(X.t(), mu_X_group1), torch.mm(mu_Z.t(), Z))
        cross_derivative_group2 = torch.mm(torch.mm(X.t(), mu_X_group2), torch.mm(mu_Z.t(), Z))
        I_cross_term = torch.cat((cross_derivative_group1, cross_derivative_group2), axis=0)
        # block matrix at bittom right: I(gamma)
        Z_star = mu_t_sqrt * Z # Z* = V^(1/2) Z
        I_gamma = torch.mm(Z_star.t(), Z_star) # ZVZ = (V^(1/2) Z)^T (V^(1/2) Z)
        # concatenate to the Fisher Information matrix
        I_top = torch.cat((I_beta, I_cross_term), axis=1) # shape: (2*P, 2*P+R)
        I_bottom = torch.cat((I_cross_term.t(), I_gamma), axis=1) # shape(R, 2*P+R)
        I = torch.cat((I_top, I_bottom), axis=0) # shape: (2*P+R, 2*P+R)

        return I


    def forward(self, X, Z, y_g_array, y_t):
        P = X.shape[1]
        y_group1, y_group2 = y_g_array[0], y_g_array[1]
        n_study = int(Z.shape[0] / 2)
        Z_group1 = Z[:n_study, :]
        Z_group2 = Z[n_study:, :]
        # mu_g^X = exp(X beta_g)
        log_mu_X_group1 = self.beta1_linear(X)
        mu_X_group1 = torch.exp(log_mu_X_group1)
        log_mu_X_group2 = self.beta2_linear(X)
        mu_X_group2 = torch.exp(log_mu_X_group2)
        # mu_g^Z = exp(Z_g gamma)
        log_mu_Z_group1 = self.gamma_linear(Z_group1)
        mu_Z_group1 = torch.exp(log_mu_Z_group1)
        log_mu_Z_group2 = self.gamma_linear(Z_group2)
        mu_Z_group2 = torch.exp(log_mu_Z_group2)
        log_mu_Z = self.gamma_linear(Z)
        mu_Z = torch.exp(log_mu_Z)
        # l = sum_g [Y_g]^T * log(mu_g^X) + [Y^t]^T * log(mu^Z) - sum_g [1^T mu_g^X]*[1^T mu_g^Z]
        l = torch.sum(torch.mul(y_group1, log_mu_X_group1)) + torch.sum(torch.mul(y_group2, log_mu_X_group2)) \
            + torch.sum(torch.mul(y_t, log_mu_Z)) \
            - torch.sum(mu_X_group1) * torch.sum(mu_Z_group1) - torch.sum(mu_X_group2) * torch.sum(mu_Z_group2)
        # Add penalty
        # l*(beta) = l(beta) + 1/2 log(|I(beta, gamma)|) 
        I = self.Fisher_Information(X, Z, mu_X_group1, mu_X_group2, mu_Z_group1, mu_Z_group2)
        I_eigens = torch.symeig(I, eigenvectors=True)[0]#[:,0]
        log_I_eigens = torch.log(I_eigens)
        log_det_I = torch.sum(log_I_eigens)
        l_fr = l + 1/2 * log_det_I
        
        return -l_fr