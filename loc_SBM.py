import numpy as np
import random
from stochastic_block_model import get_B_and_weight_vec
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def run(K=100, N1=100, N2=100, alpha=6, M=0.005):
    # Creating B matrix
    B, weight_vec = get_B_and_weight_vec(N1, N2, mu_in=2, mu_out=0.5)

    E, N = B.shape

    Gamma_vec = (1./(np.sum(abs(B), 0))).T  # \in [0, 1]
    Gamma = np.diag(Gamma_vec)

    Sigma = np.diag(np.full(E, 0.5))

    if np.linalg.norm(np.dot(Sigma**0.5, B).dot(Gamma**0.5), 2) > 1:
        print (np.linalg.norm(np.dot(Sigma**0.5, B).dot(Gamma**0.5), 2))
        raise Exception('norm is greater than 1')

    lambda_nLasso = 1/25  # nLasso parameter

    samplingset = random.choices([i for i in range(N)], k=int(M*N))

    seednodesindicator= np.zeros(N)
    seednodesindicator[samplingset] = 1
    noseednodeindicator = np.ones(N)
    noseednodeindicator[samplingset] = 0

    s = 0.0
    for i in range(len(B)):
        i = np.where(B[i] == -1)[0][0]
        j = np.where(B[i] == 1)[0][0]
        if i < N1 <= j:
            s += weight_vec[i]
        if i >= N1 > j:
            s += weight_vec[i]

    if lambda_nLasso * s >= alpha * N2 / 2:
        raise Exception('error')

    fac_alpha = 1./(Gamma_vec*alpha+1)  # \in [0, 1]

    hatx = np.zeros(N)
    newx = np.zeros(N)
    prevx = np.zeros(N)
    haty = np.array([x/(E-1) for x in range(0, E)])
    for iterk in range(K):
        tildex = 2 * hatx - prevx
        newy = haty + 1/2 * np.dot(B, tildex)  # chould be negative
        haty = newy / np.maximum(abs(newy) / (lambda_nLasso * weight_vec), np.ones(E))  # could be negative

        newx = hatx - Gamma_vec * np.dot(B.T, haty)  # could  be negative

        for dmy in range(len(samplingset)):
            idx_dmy = samplingset[dmy]
            newx[idx_dmy] = (newx[idx_dmy] + Gamma_vec[idx_dmy]) / (1 + Gamma_vec[idx_dmy])

        newx = seednodesindicator * newx + noseednodeindicator * (newx * fac_alpha)
        prevx = np.copy(hatx)
        hatx = newx  # could be negative

    if np.max(abs(newx-prevx)) > 1e-4:
        raise Exception('increase iterations')

    kmeans = KMeans(n_clusters=2, random_state=0).fit(newx.reshape(len(newx), 1))
    predicted_labels = kmeans.labels_
    true_labels = [1 for i in range(N1)] + [0 for i in range(N2)]
    acc1 = accuracy_score(true_labels, predicted_labels)
    true_labels = [0 for i in range(N1)] + [1 for i in range(N2)]
    acc2 = accuracy_score(true_labels, predicted_labels)
    return max(acc1, acc2)
