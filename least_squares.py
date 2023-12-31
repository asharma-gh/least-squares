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
    A = (np.array([[daX, dbX], [daY, dbY]]))
    res = np.linalg.solve(A, np.array([rA, rB]).reshape(2, 1))
    return res[0], res[1] # a/m, b

def l_sq_n_deg(X,Y,n):
    # Generalized Least Squares for Degree n
    pd = {} # Map of partial derivatives for each power
    yvals = []
    for ii in reversed(range(2*n + 1)):
        # For each n compute Sum(x^n)
        res = (X**ii)
        pd[ii] = res.T.dot(np.ones((X.shape[0], 1)))
        if ii <= n:
            yvals.append((res * Y).dot(np.ones((X.shape[0], 1))))
    
    pd_A = np.zeros((n+1, n+1))
    for ii in range(n+1):
        for jj in range(n+1):
            # Construct matrix A
            pd_A[ii][jj] = pd[2*n - ii - jj][0]
    pd_A = pd_A.reshape((n+1, n+1))
    yvals = np.array(yvals).reshape((n+1, 1))
    res = np.linalg.solve(pd_A, yvals).reshape((n+1, 1))
    
    return res

def l_sq_2_deg(X,Y):
    xones = np.ones((X.shape[0], 1))
    dXa = (X ** 4).T.dot(xones)[0]
    dXb = (X ** 3).T.dot(xones)[0]
    dXc = (X ** 2).T.dot(xones)[0]
    dYa = dXb
    dYb = dXc
    dYc = X.T.dot(xones)[0]
    dZa = dXc
    dZb = dYc
    dZc = X.shape[0]

    A = [[dXa, dYa, dZa],
        [dXb, dYb, dZb],
        [dXc, dYc, dZc]]
    rA = (X**2).T.dot(Y)
    rB = X.T.dot(Y)
    rC = Y.T.dot(xones)[0]
    R = [[rA],
        [rB],
        [rC]]
    A = (np.array(A))
    R = np.array(R)
    res = np.linalg.solve(A, R.reshape(3,1))
    return res[0], res[1], res[2]

def _y(m, x, b):
    return m*x + b

def gen_linear(m, b, err=5):
    ## Generate linearly correlated points
    X = np.linspace(0, 100, 500)

    Y = np.array([_y(m, x, b) + random.gauss(-1*err,err) for x in X])
    return X,Y

def gen_d2(a, b, c=1, gmin=0, gmax=10, step=1, err=500):
    X = np.linspace(gmin, 100, 50)
    Y = np.array([a*(x**2)+b*x+c + random.gauss(-1*err,err) for x in X])
    return X,Y

def gen_dn(n, c=1, gmin=0, gmax=10, step=1, err=500):
    X = np.linspace(gmin, 100, 50)
    coefs = np.linspace(0, 10, n) # [n, n-1, ... 1]
    Y = []
    for x in X:
        xtm = [] 
        # Compute kx^n for each coefficient k
        for ii, coef in enumerate(coefs):
            xtm.append(coef * x**(n - ii))
        # Add our constant and random gauss
        xtm.append(c)
        xtm.append(random.gauss(-1*err,err))
        # Reduce
        Y.append(sum(xtm, 0))

    #Y = np.array([a*(x**2)+b*x+c + random.gauss(-1*err,err) for x in X])
    return X,Y

def main():
    ## Degree 1
    #X, Y = gen_linear(1, 25)
    #m, b = l_sq_1_deg(X, Y)
    #plt.plot(X, Y, "bo")
    #Y_p = (X * m) + b
    #plt.plot(X, Y_p, "r")

    # Degree 2
    X, Y = gen_d2(1, 2, 1)
    # Degree n
    X, Y = gen_dn(2, 1)
    res = l_sq_n_deg(X,Y, 2)
    a = res[0]
    b = res[1]
    c = res[2]
    Y_p = a*(X**2) + b*X + c

    plt.plot(X, Y, "bo")
    plt.plot(X, Y_p, "r")
    plt.show()

main()
