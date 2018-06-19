import argparse
import numpy as np
import h5py
import time

from scipy.linalg import block_diag
from sklearn.metrics import r2_score

from PyUoI.UoI_Lasso import UoI_Lasso

### parse arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--block_size', type=int, default=5)
parser.add_argument('--kappa', type=float, default=0.3)
parser.add_argument('--reps', type=int, default=50)
parser.add_argument('--results_file', default='results.h5')
args = parser.parse_args()

block_size = args.block_size
kappa = args.kappa
reps = args.reps
results_file = args.results_file

n_blocks = 5
n_features = block_size * n_blocks
n_samples = 5 * n_features

correlations = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
selection_thres_mins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

results = h5py.File(results_file, 'w')
fn_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_true_results = np.zeros((reps, correlations.size, selection_thres_mins.size))

for rep in range(reps):
	beta = np.random.uniform(low=0, high=10, size=(n_features, 1))
	for corr_idx, correlation in enumerate(correlations):
		# create covariance matrix for block
		block_Sigma = correlation * np.ones((block_size, block_size)) 
		np.fill_diagonal(block_Sigma, np.ones(block_size))
		# populate entire covariance matrix
		Sigma = block_diag(block_Sigma, block_Sigma, block_Sigma, block_Sigma, block_Sigma)
		# draw samples
		X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=Sigma, size=n_samples)
		X_test = np.random.multivariate_normal(mean=np.zeros(n_features), cov=Sigma, size=n_samples)
		# signal and noise variance
		signal_variance = np.sum(Sigma * np.dot(beta, beta.T))
		noise_variance = kappa * signal_variance
		# draw noise
		noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
		noise_test = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
		# response variable
		y = np.dot(X, beta) + noise
		y_test = np.dot(X_test, beta) + noise_test
		for thres_idx, selection_thres_min in enumerate(selection_thres_mins):
			start = time.time()
			uoi = UoI_Lasso(
				normalize=True,
				n_boots_sel=48,
				n_boots_est=48,
				selection_thres_min=selection_thres_min,
				n_selection_thres=48,
				estimation_score='BIC'
			)
			uoi.fit(X, y.ravel())
			beta_hat = uoi.coef_
			fn_results[rep, corr_idx, thres_idx] = np.count_nonzero(beta_hat == 0)
			r2_results[rep, corr_idx, thres_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
			r2_true_results[rep, corr_idx, thres_idx] = r2_score(y_test, np.dot(X_test, beta))
			print(time.time() - start)
results['fn'] = fn_results
results['r2'] = r2_results
results['r2_true'] = r2_true_results
results.close()


