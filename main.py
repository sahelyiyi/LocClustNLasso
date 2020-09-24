import numpy as np

N = 10  # nr of nodes in chain
K = 100  # nr of iterations used for nLasso
E = N-1  # nr of edges in chain

# Creating B matrix
B = np.diag(np.ones(N), 0) - np.diag(np.ones(E), 1)  # incidence matrix
B = B[0:(E), :]
# ---------------------
# B1 = np.zeros((N, E))
# for i in range(N):
#     B1[i, i] = 1
#     if i < N-1:
#         B1[i, i+1] = -1
#     else:
#         B1[i, 0] = -1

# Creating weight_vec
weight = np.eye(E, E)
weight_vec = np.array([1./x for x in range(1, N)])
weight_vec = (5.0/4)*np.ones(E)
eta = 1/4
eta = 1
weight_vec[1] = eta
# ----------------------
# weight_vec = weight_vec = (5.0/4)*np.ones(E)
# weight_vec[1] = eta


Lambda = np.diag(1./(np.sum(abs(B), 1)))
Gamma_vec = (1./(np.sum(abs(B), 0))).T  # \in [0, 1]
Gamma = np.diag(Gamma_vec)

weight = np.diag(weight_vec)

lambda_nLasso = 1/3  # nLasso parameter

primSLP = np.ones(N)
primSLP[N-1] = 0
dualSLP = np.array([x/(E-1) for x in range(0, E)])

hatx = np.zeros(N)
prevx = np.zeros(N)
haty = np.array([x/(E-1) for x in range(0, E)])

# Define sampling set
samplingset = [0]
# ------------------

seednodesindicator= np.zeros(N)
seednodesindicator[samplingset] = 1
noseednodeindicator = np.ones(N)
noseednodeindicator[samplingset] = 0


running_average = 0*hatx
running_averagey = 0*hatx


log_conv= np.zeros(K)
log_bound= np.zeros(K)
newx = 0*hatx

hist_y = np.zeros((K, E))
hist_x = np.zeros((K, N))

alpha = 1/10
fac_alpha = 1./(Gamma_vec*alpha+1)  # \in [0, 1]


for iterk in range(K):
    # tildex = 2 * newx - hatx
    tildex = 2 * hatx - prevx
    newy = haty + (1 / 2) * np.dot(B, tildex)  # chould be negative
    haty = newy / np.maximum(abs(newy) / (lambda_nLasso * weight_vec), np.ones(E))  # could be negative

    newx = hatx - Gamma_vec * np.dot(B.T, haty)  # could  be negative

    for dmy in range(len(samplingset)):
        idx_dmy = samplingset[dmy]
        newx[idx_dmy] = (newx[idx_dmy] + Gamma_vec[idx_dmy]) / (1 + Gamma_vec[idx_dmy])

    newx = seednodesindicator * newx + noseednodeindicator * (newx * fac_alpha)
    prevx = np.copy(hatx)
    hatx = newx  # could be negative

    hist_y[iterk, :] = haty
    hist_x[iterk, :] = hatx

    running_average = (running_average * iterk + hatx) / (iterk + 1)
    dual = np.sign(np.dot(B, running_average))

    dual[iterk:(N - 1)] = 0

    log_conv[iterk] = sum(abs(np.dot(B, running_average)))
    log_bound[iterk] = (1 / (2 * (iterk+1))) * (np.dot(primSLP.T, np.linalg.inv(Gamma)).dot(primSLP))+(np.dot((dualSLP-dual).T, np.linalg.inv(Lambda)).dot(dualSLP - dual))

