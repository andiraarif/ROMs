#==============================================================================#
# POD of Oscillatory Gaussian Equation
# by Mohamad Arif Andira
# -----------------------------------------------------------------------------#
# Description:
# Perform Proper Orthogonal Decomposition (POD) to 1D oscillatory gaussian
# and reconstruct the solution using selected number of POD modes
#
#==============================================================================#

import svd
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Spatial discretization
x = np.linspace(-2, 2, 401)
Nx = np.size(x)

# Temporal discretization
t = np.linspace(0, 10, 1001)
Nt = np.size(t)

# Function parameters
amp1 = 1
x01 = 0.5
sigmay1 = 0.6
omega1 = 1.3

amp2 = 1.2
x02 = -0.5
sigmay2 = 0.3
omega2 = 4.1

# Calculate function
y1 = amp1*np.exp(-(x-x01)**2/(2*sigmay1**2))
y2 = amp2*np.exp(-(x-x02)**2/(2*sigmay2**2))

Y = np.zeros([Nx, Nt])

for i in range(Nt):
	Y[:, i] = y1*np.sin(2*np.pi*omega1*t[i]) + y2*np.sin(2*np.pi*omega2*t[i])

# Perform POD
u, s, vt = svd.compute(Y, full_matrices=False)
svd.plot_sigma(s, modes_limit=10)

# Reconstruct matrix
Yred = svd.reconstruct(u, s, vt, n_basis=2)


def plot_function(x, Y, Yred):
	plt.ion()
	figure, ax = plt.subplots()
	line1, = ax.plot(x, Y[:, 0], label="Analytic")
	line2, = ax.plot(x, Yred[:, 0], label="POD")

	plt.title("1-Dimensional Oscillatory Gaussan",fontsize=12)
	plt.xlabel("X",fontsize=10)
	plt.ylabel("Y",fontsize=10)
	plt.axis([-2, 2, -1.5, 1.5])
	plt.legend()

	for t in range(Nt-1):
		updated_y = Y[:, t+1]
		updated_yred = Yred[:, t+1]

		line1.set_xdata(x)
		line1.set_ydata(updated_y)

		line2.set_xdata(x)
		line2.set_ydata(updated_yred)

		figure.canvas.draw()
		figure.canvas.flush_events()

		time.sleep(0.01)


plot_function(x, Y, Yred)
