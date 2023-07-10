import numpy as np
import random
from matplotlib import pyplot as plt

# Least square estimation with 1 degree polynomial
def l_sq_1_deg(X, Y):
    daX = X.T.dot(X)
    daY = X.T.dot(np.ones((X.shape[0], 1)))[0]

    dbX = daY
    dbY = X.shape[0]
    
    rA = X.T.dot(Y)
    rB = Y.T.dot(np.ones((Y.shape[0], 1)))[0]
    A = np.array([[daX, dbX], [daY, dbY]])
    res = np.linalg.solve(A, np.array([rA, rB]).reshape(2, 1))
    return res[0], res[1] # a/m, b


def _y(m, x, b):
    return m*x + b

def gen_linear(m, b, err=5):
    ## Generate linearly correlated points
    X = np.linspace(0, 100, 500)

    Y = np.array([_y(m, x, b) + random.gauss(-1*err,err) for x in X])
    return X,Y

X, Y = gen_linear(1, 25)
m, b = l_sq_1_deg(X, Y)
plt.plot(X, Y, "bo")
Y_p = (X * m) + b
plt.plot(X, Y_p, "r")
plt.show()

