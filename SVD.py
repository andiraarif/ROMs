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
	U, S, VT = np.linalg.svd(A, full_matrices, compute_uv, hermitian)
	S = sigma_matrix(S)

	return U, S, VT


def sigma_matrix(s):
	size = s.shape[0]
	sigma = np.eye(size)

	for i in range(size):
		for j in range(size):
			if i == j:
				sigma[i][j] *= s[i]

	return sigma


def reconstruct_matrix(u, s, vt):
	return u@sigma_matrix(s)@vt


def plot_sigma(s, diag_matrix=True, modes_limit=0):
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

	plt.semilogy(sigma, '-ko')
	plt.xlim(0, n_modes-1)
	plt.xlabel('Index', fontsize=12)
	plt.ylabel('Singular Value', fontsize=12)
	plt.show()

"""
#Test case
A = np.random.rand(100, 100)
U, S, VT = compute(A)
plot_sigma(S)
"""
