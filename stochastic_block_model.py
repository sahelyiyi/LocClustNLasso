from numpy.random import normal, poisson
from graspy.simulations import sbm
import numpy as np


def get_B_and_weight_vec(n1=50, n2=50, pin=0.7, pout=0.01, mu_in=8, mu_out=2):
    n = [n1, n2]
    p = [[pin, pout],
         [pout, pin]]
    wt = [[normal, normal],
          [normal, normal]]
    # wt = [[normal, poisson],
    #       [poisson, normal]]
    wtargs = [[dict(loc=mu_in, scale=1), dict(loc=mu_out, scale=1)],
              [dict(loc=mu_out, scale=1), dict(loc=mu_in, scale=1)]]
    # wtargs = [[dict(loc=3, scale=1), dict(lam=5)],
    #           [dict(lam=5), dict(loc=3, scale=1)]]

    G = sbm(n=n, p=p, wt=wt, wtargs=wtargs)

    N = len(G)
    # print(len(G[G > 0]) / 2)
    E = int(len(G[G > 0]) / 2)
    B = np.zeros((E, N))
    cnt = 0
    weight_vec = np.zeros(E)
    for item in np.argwhere(G > 0):
        # if cnt % 1000000 == 0:
        #     print (cnt)
        i, j = item
        if i > j:
            continue
        if i == j:
            print ('nooooo')
        B[cnt, i] = 1
        B[cnt, j] = -1
        weight_vec[cnt] = abs(G[i, j])
        cnt += 1

    return B, weight_vec
