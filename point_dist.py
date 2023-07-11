import numpy as np
import random
from matplotlib import pyplot as plt
# custom
from least_squares import l_sq_1_deg


def dist(X,Y):
    return np.sqrt((X**2) + (Y**2))

def dist_k(x,y):
    return np.sqrt(x**2 + y**2)

def compute_perp_dist(X,Y, m, b):
    # compute and show perpendicular distance for each point to
    # best fit line (m, b)
    # center axis to 0,0
    yoff = -1*b
    xoff = b / m
    X = X - xoff
    Y = Y - yoff
    theta = np.arctan(m)
    mx = np.max(X) * 1.25
    my = _y(m, mx, 0) # compute end point for L vector given M,B for proj 
    plt.plot(mx, my, "go")
    lu = np.array([[mx], [my]]).reshape((2,1)) / dist_k(mx, my) # Unit vector of L 
    pt = np.array([[40], [80]]).reshape(2,1)
    co = pt.T.dot(np.array([[mx], [my]]).reshape((2,1))) / (dist_k(40,80)*dist_k(mx, my))
    npt = (dist_k(40, 80) * co) * lu
    npt[1] = (npt[0] * m) + b
    plt.plot(npt[0], npt[1], "g+")

    proj = (dist(X,Y) * np.cos(theta)).reshape(X.shape[0], 1)
    proj = proj * np.array([lu]).reshape(1,1,2).T #+ np.array([[[xoff], [yoff]]]).reshape(1,1,2)).T
    # Todo: vectorize this
    #resid = []
    #L = np.array([[mx], [my]]).reshape(2, 1)
    #for p in proj:
    #    resid.append(p * L)
    X = X + xoff
    Y = Y + yoff
    return proj


def plot_resids(perps, X, Y):
    # plots the line formed from (X,Y)i -> Perp_i
    #plt.plot(perps[0], perps[1], "go")
    pass



def _y(m, x, b):
    return m*x + b

def gen_linear(m, b, err=5):
    ## Generate linearly correlated points
    X = np.linspace(0, 100, 500)
    Y = np.array([_y(m, x, b) + random.gauss(-1*err,err) for x in X])
    return X,Y

X, Y = gen_linear(1, 0)
m, b = l_sq_1_deg(X, Y)
#plt.plot(X, Y, "bo")
Y_p = (X * m) + b
plt.plot(X, Y_p, "r")
perp = compute_perp_dist(X, Y, m[0], b)
plot_resids(perp, X, Y)
plt.show()