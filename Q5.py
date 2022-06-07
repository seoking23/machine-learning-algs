import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

x_1 = np.linspace(-5,5)
x_2 = np.linspace(-5,5)
X, Y = np.meshgrid(x_1, x_2)
p = 0.5

def least_squared_array(z, y, l_var, w_norm):
    return (z-y)**2 + l_var*w_norm

def norm_w(p, w):
    sum_abs_w = 0
    
    for w_i in w:
        sum_abs_w += (np.abs(w_i)**p)
    
    return sum_abs_w**(1/p)

#norm_w derivative = 1/p* (np.sum(np.abs(w)**p))**-(1 - (1/p)) * p * (np.sum)

def w(X,y):
    return X.T @ y

z = norm_w(p, [X, Y])
print(norm_w(p, [X, Y]))

fig = plt.figure()
ax = fig.add_subplot(111)
pyplot.contourf(X, Y, z)
pyplot.title("l_"+str(p)+" norm isocontour")
ax.set_aspect('equal', adjustable='box')
pyplot.show()
