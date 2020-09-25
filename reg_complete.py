import numpy as np
import random
from math import sqrt
from collections import defaultdict, Counter

N = 300  # nr of nodes in chain
K = 100  # nr of iterations used for nLasso
E = int(N * (N-1) / 2)
M = random.choices([i for i in range(N)], k=int(0.1*N))
# M = [i for i in range(N)]

Y = np.array([[2] for i in range(N)])
X = np.array([[[1, 1, 1]] for i in range(N)])
m, n = X[0].shape


B = np.zeros((E, N))   # incidence matrix
cnt = 0
for i in range(N):
    for j in range(i+1, N):
        B[cnt, i] = 1
        B[cnt, j] = -1
        cnt += 1

weight_vec = 2 * np.ones(E)
weight = np.diag(weight_vec)
Sigma = np.diag(1./(2*weight_vec))

D = np.dot(weight, B)

Lambda = 1*np.diag(1./(np.sum(abs(B), 1)))
Gamma_vec = (.9/(np.sum(abs(B), 0))).T  # \in [0, 1]
Gamma = np.diag(Gamma_vec)

lambda_nLasso = 1/3  # nLasso parameter

hat_w = np.array([np.zeros(n) for i in range(N)])
new_w = np.array([np.zeros(n) for i in range(N)])
prev_w = np.array([np.zeros(n) for i in range(N)])
new_u = np.array([np.zeros(n) for i in range(E)])

K = 300
# 1.0913559876627825e-07
for iterk in range(K):
    print ('iter:', iterk)

    tilde_w = 2 * hat_w - prev_w
    new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))  # chould be negative

    hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))  # could  be negative

    for i in range(N):
        if i in M:
            mtx1 = 1.8 * np.dot(X[i].T, X[i]).astype('float64')
            if mtx1.shape:
                mtx1 += Gamma_vec[i] * np.eye(mtx1.shape[0])
                mtx_inv = np.linalg.inv(mtx1)
            else:
                mtx1 += Gamma_vec[i]
                mtx_inv = 1.0 / mtx1

            mtx2 = Gamma_vec[i] * hat_w[i] + 1.8 * np.dot(X[i].T, Y[i])

            new_w[i] = np.dot(mtx_inv, mtx2)
        else:
            new_w[i] = hat_w[i]
    prev_w = np.copy(new_w)

mse = 0
for i in range(N):
    # print (Y[i], np.dot(X[i], new_w[i]))
    mse += np.linalg.norm(Y[i] - np.dot(X[i], new_w[i]))
mse /= N

