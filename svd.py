#==============================================================================#
# Singular Value Decomposition (SVD)
# by Mohamad Arif Andira
# -----------------------------------------------------------------------------#
# Description:
# This utility helps to compute SVD from a nxm matrix as well as plot the
# singular value (sigma)
#
#==============================================================================#

import numpy as np
from matplotlib import pyplot as plt


def compute(A, full_matrices=True, compute_uv=True, hermitian=False):
	u, s, vt = np.linalg.svd(A, full_matrices, compute_uv, hermitian)
	s = sigma_to_matrix(u, s, vt, full_matrices)

	return u, s, vt


def reconstruct(u, s, vt, n_basis=0):
	if n_basis == 0:
		n_basis = s.shape[0]

	u = u[:, :n_basis]
	s = s[:n_basis, :]

	return u@s@vt


def sigma_to_matrix(u, s, vt, full_matrices=True):
	n_u = u.shape[0]
	n_s = s.shape[0]
	n_vt = vt.shape[0]

	if full_matrices:
		sigma = np.zeros((n_u, n_vt))
	else:
		sigma = np.zeros((n_s, n_s))

	for i in range(sigma.shape[0]):
		for j in range(sigma.shape[1]):
			if i == j:
				sigma[i][j] = 1
				sigma[i][j] *= s[i]

	return sigma


def plot_sigma(s, modes_limit=0, diag_matrix=True):
	sigma = []
	n_sigma = s.shape[0]

	if modes_limit == 0:
		n_modes = n_sigma
	else:
		n_modes = modes_limit

	if diag_matrix:
		for i in range(n_sigma):
			for j in range(n_sigma):
				if i==j:
					sigma.append(s[i][j])
	else:
		sigma = s

	plt.title("Singular Values based on Sigma Matrix",fontsize=12)
	plt.semilogy(sigma, '-ko')
	plt.xlim(0, n_modes-1)
	plt.xlabel('Index', fontsize=10)
	plt.ylabel('Singular Value', fontsize=10)
	plt.show()

"""
#Test case
A = np.random.rand(100, 100)
u, s, vt = compute(A)
plot_sigma(s)
"""
