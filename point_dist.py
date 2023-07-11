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
    X = X
    Y = Y
    Yoff = b
    n = X.shape[0]
    mx = np.max(X) * 1.25
    my = _y(m, mx, 0) # compute end point for L vector given M,B for proj 
    Lvec = np.array([[mx], [my]]).reshape((2,1))
    Lu = Lvec / dist_k(mx, my) # Unit vector of L 
    tvec = np.hstack((X, Y)).reshape(n, 2)
    distxy = dist(X, Y).reshape(n, 1) # Composited vector containing magnitude xi,yi
    pt = np.array([[40], [80]]).reshape(2,1)
    costheta_v = np.divide(tvec.dot(Lvec), distxy * np.linalg.norm(Lvec))
    costheta = np.divide(pt.T.dot(Lvec), (np.linalg.norm(pt) * np.linalg.norm(Lvec)))
    proj_v = distxy * costheta * np.array([Lu]).reshape(1,1,2) + np.array([0, Yoff[0]]).reshape(1,1,2)
    print("Lu ", Lu)
    projpt = np.linalg.norm(pt) * costheta * Lu
    projpt += np.array([0, Yoff[0]]).reshape((2,1))
    plt.plot([pt[0], projpt[0]], [pt[1], projpt[1]], "g+-")

    print("Proj ", projpt, projpt / np.linalg.norm(projpt))
    return proj_v


def plot_resids(perps, X, Y):
    # plots the line formed from (X,Y)i -> Perp_i
    for ii, p in enumerate(perps[0]):
       plt.plot([X[ii], p[0]], [Y[ii], p[1]], "go-")
    pass



def _y(m, x, b):
    return m*x + b

def gen_linear(m, b, err=5):
    ## Generate linearly correlated points
    X = np.linspace(0, 100, 10)
    Y = np.array([_y(m, x, b) + random.gauss(-1*err,err) for x in X])
    return X,Y

X, Y = gen_linear(1, 0, 50)
m, b = l_sq_1_deg(X, Y)
#plt.plot(X, Y, "bo")
Y_p = (X * m) + b
plt.plot(X, Y_p, "r")
perp = compute_perp_dist(X, Y, m[0], b)
plot_resids(perp, X, Y)
plt.show()