import numpy as np
import random
from matplotlib import pyplot as plt

# Least square estimation with 1 degree polynomial
def l_sq_1_deg(p):
    pass

def _y(m, x, b):
    return m*x + b

def gen_linear(m, b, err=5):
    ## Generate linearly correlated points
    X = np.linspace(0, 100, 500)

    Y = np.array([_y(m, x, b) + random.gauss(-1*err,err) for x in X])
    return X,Y

X, Y = gen_linear(1, 25)
plt.plot(X, Y, "bo")
plt.show()

