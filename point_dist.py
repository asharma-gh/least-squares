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
    #ptest2 = np.divide(tvec.dot(Lvec), Lvec.T.dot(Lvec)) * np.array([Lu]).reshape(1,1,2)
    costheta = np.divide(pt.T.dot(Lvec), (np.linalg.norm(pt) * np.linalg.norm(Lvec)))
    proj_v = distxy * costheta_v * np.array([Lu]).reshape(1,1,2) + np.array([0, Yoff[0]]).reshape(1,1,2)
    print(proj_v)
    print("Lu ", Lu)
    for p in proj_v[0]:
        p = p - np.array([0, Yoff[0]]).reshape(1,1,2)
        print(p / np.linalg.norm(p))
    projpt = np.linalg.norm(pt) * costheta * Lu
    projpt += np.array([0, Yoff[0]]).reshape((2,1))
    #plt.plot([pt[0], projpt[0]], [pt[1], projpt[1]], "g+-")

    return proj_v


def plot_resids(perps, X, Y):
    # plots the line formed from (X,Y)i -> Perp_i
    for ii, p in enumerate(perps[0]):
       plt.plot([X[ii], p[0]], [Y[ii], p[1]], "go-")
    pass

def basic_proj():
    X1 = np.array([10, 10]).reshape(2,1)
    X2 = np.array([7, -16]).reshape(2, 1)

    nvec = np.divide(X2.T.dot(X1), np.linalg.norm(X1)) * (np.divide(X1, np.linalg.norm(X1)))
    print(np.linalg.norm(X1))
    print(np.linalg.norm(nvec))
    plt.plot([0, X2[0][0]], [0, X2[1][0]], "bo-")
    plt.plot([0, 10], [0, 10], "bo-")
    plt.plot([0, nvec[0][0]], [0, nvec[1][0]], "go-")
    plt.show()

def vec_dist(pp):
    # Line
    aa = np.array([10, 5]).reshape(2,1) # Pt on line
    nn = np.array([2, 1]).reshape(2,1) # Slope of line, unit vector 
    nn = np.divide(nn, np.linalg.norm(nn))

    # Calculate distance to point p
    distpp_aa = pp - aa #vector from P -> A
    pproj_k = (distpp_aa.T.dot(nn)) # project dist vec to n scalar
    pproj = aa + (pproj_k * nn) # Calculate point on line with this scale param
    pproj_dv = pp - pproj
    pproj_dist = np.linalg.norm(pproj_dv)
    print("Dot test: ", pproj_dv.T.dot(nn))
    print(pproj, pp)
    # Show result
    # test pt
    tpt = aa + (-15 * nn)
    plt.plot([aa[0][0], tpt[0][0]], [aa[1][0], tpt[1][0]], "bo-") # point aa
    plt.plot(pp[0], pp[1], "r+") # point pp
    plt.plot(pproj[0], pproj[1], "g+") # projected point
    plt.text(0, 0, "Distance: {}".format(pproj_dist))
    plt.show()

def _y(m, x, b):
    return m*x + b

def gen_linear(m, b, err=5):
    ## Generate linearly correlated points
    X = np.linspace(0, 100, 10)
    Y = np.array([_y(m, x, b) + random.gauss(-1*err,err) for x in X])
    return X,Y

#plt.xlim(-100, 120)
#plt.ylim(-100, 120)
X, Y = gen_linear(1, 0, 50)
m, b = l_sq_1_deg(X, Y)
#plt.plot(X, Y, "bo")
Y_p = (X * m) + b
#plt.plot(X, Y_p, "r")
#perp = compute_perp_dist(X[:5], Y[:5], m[0], b)
#basic_proj()
#plot_resids(perp, X, Y)
#plt.show()
vec_dist(np.array([8,5]).reshape(2,1))