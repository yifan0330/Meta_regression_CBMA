#import torch
#from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import patsy
from scipy.sparse import csr_matrix, save_npz
from scipy.stats import poisson

class Simulation2D_X():
    def __init__(self, spacing=15):
        self.spacing = spacing

    def Design_matrix(self, x_max=100, y_max=100):
        xx = np.arange(x_max)
        yy = np.arange(y_max)
        ## create B-spline basis for x/y/z coordinate
        x_knots = np.arange(min(xx), max(xx), step=self.spacing)
        x_design_matrix = patsy.dmatrix("bs(x, knots=x_knots, degree=3,include_intercept=False)", data={"x":xx},return_type='matrix')
        x_design_array = np.array(x_design_matrix)
        x_spline = x_design_array[:,1:] # remove the first column (every element is 1)
        #x_rowsum = np.sum(x_spline, axis=1)
        y_knots = np.arange(min(yy), max(yy), step=self.spacing)
        y_design_matrix = patsy.dmatrix("bs(y, knots=y_knots, degree=3,include_intercept=False)", data={"y":yy},return_type='matrix')
        y_design_array = np.array(y_design_matrix) 
        y_spline = y_design_array[:,1:] 
        #y_rowsum = np.sum(y_spline, axis=1)
        ## convert spline bases in 3 dimesion to data matrix by tensor product
        X = np.kron(x_spline, y_spline)
        # Row sums of X are all 1=> There is no need to re-normalise X
        return X

class Simulation2D_count():
    def __init__(self, n_voxel=10000, n_group=1, n_study=25, covariates=False):
        self.n_voxel = n_voxel
        self.n_group = n_group
        self.n_study = n_study
        self.covariates = covariates

    def Poisson_one_group(self, mu, max_val=1, n_covariates=2):
        if self.covariates == False:
            voxelwise_foci = np.zeros(shape=(self.n_voxel, 1))
            for study in range(self.n_study):
                Poisson_generator = np.random.poisson(mu, size=(self.n_voxel, 1))
                voxelwise_foci = voxelwise_foci + Poisson_generator
            return voxelwise_foci # y
        else: 
            # spatial info
            voxelwise_foci = np.zeros(shape=(self.n_voxel, 1))
            studywise_foci = np.zeros(shape=(self.n_study, 1))
            for study_index in range(self.n_study):
                Poisson_generator = np.random.poisson(mu, size=(self.n_voxel, 1))
                voxelwise_foci = voxelwise_foci + Poisson_generator
                studywise_foci[study_index, :] = np.sum(Poisson_generator)
            # study-covariates info
            covariates_array = np.random.uniform(low=-max_val, high=max_val, size=(self.n_study, n_covariates))

            # normalization
            covariates_array = covariates_array - np.mean(covariates_array, axis=0)
            covariates_array = covariates_array / np.std(covariates_array, axis=0)
            return voxelwise_foci, studywise_foci, covariates_array # y_g, y_t, Z

    def Poisson_multiple_group_covariates(self, mu_list, max_val=1, n_covariates=2):
        mu_list = [float(i) for i in mu_list]
        # spatial info
        voxelwise_foci = list()
        y_t = list()
        for group_index in range(self.n_group):
            y_g = np.zeros(shape=(self.n_voxel, 1))
            group_mu = mu_list[group_index]
            for study_index in range(self.n_study):
                Poisson_generator = np.random.poisson(group_mu, size=(self.n_voxel, 1))
                y_g = y_g + Poisson_generator
                y_t.append(np.sum(Poisson_generator))
            voxelwise_foci.append(y_g)
        voxelwise_foci = np.array(voxelwise_foci) # shape: (n_group, n_voxel, 1)
        y_t = np.array(y_t).reshape(len(y_t), 1) # shape:(n_group*n_study, 1)
        # study-covariates info
        covariates_array = np.random.uniform(low=-max_val, high=max_val, size=(self.n_group*self.n_study, n_covariates))
        # normalization
        covariates_array = covariates_array - np.mean(covariates_array, axis=0)
        covariates_array = covariates_array / np.std(covariates_array, axis=0) # shape: (n_group*n_study, n_covariates)
        return voxelwise_foci, y_t, covariates_array

class Simulation2D_cluster():
    def __init__(self, centers, shape=(100,100), n_group=1, n_study=25, covariates=False):
        self.centers = centers
        self.shape = shape
        self.n_group = n_group
        self.n_study = n_study
        self.covariates = covariates
    
    def Poisson_one_group(self, mu=6.5, var=10, n_cluster_foci=[0,1,2,3], max_val=1, n_covariates=2):
        self.var = var
        # the probability of a valid study report [0,1,2,3] foci 
        prob = poisson.pmf(np.arange(4), mu)
        # scale the sum to 1
        prob = prob / np.sum(prob)
        # print(prob)
        # print(np.sum(prob*np.array(n_cluster_foci))*8)
        y = np.zeros(shape=self.shape)
        y_t = []
        for study_index in range(self.n_study):
            studywise_foci = 0
            for center_coord in self.centers:
                n_foci = np.random.choice(n_cluster_foci, size=1, replace=True, p=prob).item()
                studywise_foci += n_foci
                for foci in range(n_foci):
                    foci_coord = np.random.multivariate_normal(mean=center_coord, cov=self.var*np.eye(2))
                    foci_coord = np.rint(foci_coord).astype(int)
                    y[foci_coord[0], foci_coord[1]] += 1
            y_t.append(studywise_foci)
        y_t = np.array(y_t).reshape(self.n_study, 1)
        y = y.reshape(np.prod(self.shape), 1)
        if self.covariates == False:
            return y
        else:
            # study-covariates info
            covariates_array = np.random.uniform(low=-max_val, high=max_val, size=(self.n_group*self.n_study, n_covariates))
            # normalization
            covariates_array = covariates_array - np.mean(covariates_array, axis=0)
            covariates_array = covariates_array / np.std(covariates_array, axis=0) # shape: (n_group*n_study, n_covariates)
            return y, y_t, covariates_array