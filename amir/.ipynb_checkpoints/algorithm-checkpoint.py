import numpy as np
import random
import math
import datetime
import itertools

from sbm import SBM
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, spectral_clustering, SpectralClustering
from scipy.sparse import csr_matrix
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

ALPHA = 0.1

def get_B_and_weight_vec_ring(points, threshhold=0.2):
    N = len(points)
    clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(points)
    A = clustering.affinity_matrix_

    row = []
    col = []
    data = []
    weight_vec = []
    cnt = 0
    for i in range(N):
        for j in range(N):
            if j <= i:
                continue
            if A[i, j] < threshhold:
                A[i, j] = 0
                A[j, i] = 0
                continue
            row.append(cnt)
            col.append(i)
            data.append(1)

            row.append(cnt)
            col.append(j)
            data.append(-1)
            cnt += 1
            weight_vec.append(A[i, j])

    B = csr_matrix((data, (row, col)), shape=(cnt, N))
    weight_vec = np.array(weight_vec)
    return A, B, weight_vec


def algorithm(B, weight_vec, N1, K=15000, M=0.2, alpha=ALPHA, lambda_nLasso=None, check_s=False):
    E, N = B.shape
    weight_vec = np.ones(E)

    Gamma_vec = np.array(1./(np.sum(abs(B), 0)))[0]  # \in [0, 1]
    Gamma = np.diag(Gamma_vec)

    Sigma = 0.5
    
    samplingset = random.choices([i for i in range(N1)], k=int(M*N1))

    seednodesindicator= np.zeros(N)
    seednodesindicator[samplingset] = 1
    noseednodeindicator = np.ones(N)
    noseednodeindicator[samplingset] = 0
    
    if lambda_nLasso == None:
        lambda_nLasso = 2 / math.sqrt(np.sum(weight_vec))
    
    if check_s:
        s = 0.0
        for item in range(len(weight_vec)):
            x = B[item].toarray()[0]
            i = np.where(x == -1)[0][0]
            j = np.where(x == 1)[0][0]
            if i < N1 <= j:
                s += weight_vec[item]
            elif i >= N1 > j:
                s += weight_vec[item]

        if lambda_nLasso * s >= alpha * N2 / 2:
            print ('eq(24)', lambda_nLasso * s, alpha * N2 / 2)
    
    fac_alpha = 1./(Gamma_vec*alpha+1)  # \in [0, 1]

    hatx = np.zeros(N)
    newx = np.zeros(N)
    prevx = np.zeros(N)
    haty = np.array([x/(E-1) for x in range(0, E)])
    history = []
    for iterk in range(K):
        tildex = 2 * hatx - prevx
        newy = haty + Sigma * B.dot(tildex)  # chould be negative
        haty = newy / np.maximum(abs(newy) / (lambda_nLasso * weight_vec), np.ones(E))  # could be negative

        newx = hatx - Gamma_vec * B.T.dot(haty)  # could  be negative
        newx[samplingset] = (newx[samplingset] + Gamma_vec[samplingset]) / (1 + Gamma_vec[samplingset])

        newx = seednodesindicator * newx + noseednodeindicator * (newx * fac_alpha)
        prevx = np.copy(hatx)
        hatx = newx  # could be negative
        history.append(newx)
    
    history = np.array(history)

    return history
    

def run(N1, N2, K=15000, M=0.2, alpha = ALPHA):
    B, weight_vec = get_B_and_weight_vec([N1, N2], mu_in=2, mu_out=0.5, pin=0.2, pout=0.01)
    
    history = algorithm(B, weight_vec, N1, K)
    return history

n_samples = 6000

def accuracy(labels, true_labels):
    total_common = 0
    cluster_names = set(true_labels)
    permutations = list(itertools.permutations(cluster_names))
    for permutation in permutations:
        max_common = 0
        for i, cluster_name in enumerate(cluster_names):
            cluster_nodes = np.where(labels == cluster_name)[0]
            cluster_name1 = permutation[i]
            true_nodes = np.where(true_labels == cluster_name1)[0]

            common = len(set(true_nodes) - (set(true_nodes) - set(cluster_nodes)))
            max_common += common

        total_common = max(total_common, max_common)

    return total_common / len(true_labels)

# def get_B_and_weight_vec_ring(points, threshhold=0.2):
#     N = len(points)
#     clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(points)
#     A = clustering.affinity_matrix_

#     row = []
#     col = []
#     data = []
#     weight_vec = []
#     cnt = 0
#     for i in range(N):
#         for j in range(N):
#             if j <= i:
#                 continue
#             if A[i, j] < threshhold:
#                 A[i, j] = 0
#                 A[j, i] = 0
#                 continue
#             row.append(cnt)
#             col.append(i)
#             data.append(1)

#             row.append(cnt)
#             col.append(j)
#             data.append(-1)
#             cnt += 1
#             weight_vec.append(A[i, j])

#     B = csr_matrix((data, (row, col)), shape=(cnt, N))
#     weight_vec = np.array(weight_vec)
#     return A, B, weight_vec


# def run_more_plots(points, true_labels, K, alpha, lambda_nLasso, threshhold, n_clusters, M=0.2, plot=False, is_print=True):
#     A, B, weight_vec = get_B_and_weight_vec_ring(points, threshhold=threshhold)

#     E, N = B.shape

#     Gamma_vec = np.array(1. / (np.sum(abs(B), 0)))[0]  # \in [0, 1]
#     Gamma = np.diag(Gamma_vec)

#     Sigma = 0.5

#     fac_alpha = 1. / (Gamma_vec * alpha + 1)  # \in [0, 1]
#     lambda_weight = lambda_nLasso * weight_vec

#     our_labels = np.full(N, n_clusters-1)
#     our_time = datetime.datetime.now() - datetime.datetime.now()
#     for clust_num in range(n_clusters-1):

#         samplingset = random.choices(np.where(true_labels==clust_num)[0], k=int(M * len(np.where(true_labels==clust_num)[0])))
#         seednodesindicator = np.zeros(N)
#         seednodesindicator[samplingset] = 1
#         noseednodeindicator = np.ones(N)
#         noseednodeindicator[samplingset] = 0


#         hatx = np.zeros(N)
#         newx = np.zeros(N)
#         prevx = np.zeros(N)
#         haty = np.array([x / (E - 1) for x in range(0, E)])
#         gamma_plus = 1 + Gamma_vec[samplingset]
#         start = datetime.datetime.now()
#         for iterk in range(K):
#             tildex = 2 * hatx - prevx
#             newy = haty + Sigma * B.dot(tildex)  # chould be negative
#             res = abs(newy) / lambda_weight
#             res[res < 1] = 1
#             haty = newy / res

#             newx = hatx - Gamma_vec * B.T.dot(haty)  # could  be negative

#             newx[samplingset] = (newx[samplingset] + Gamma_vec[samplingset]) / gamma_plus

#             newx = seednodesindicator * newx + noseednodeindicator * (newx * fac_alpha)
#             prevx = np.copy(hatx)
#             hatx = newx  # could be negative
#         our_time += datetime.datetime.now() - start
#         X = newx
#         X = np.nan_to_num(X, 0)
#         kmeans = KMeans(n_clusters=2, random_state=0).fit(X.reshape(len(X), 1))
#         matched_label = kmeans.labels_[samplingset][0]
#         our_labels[np.where(kmeans.labels_ == matched_label)[0]] = clust_num

#     our_accuracy = accuracy(our_labels, true_labels)
#     if is_print:
#         print ('our time is:', our_time)
#         print ('our accuracy is:', our_accuracy)
#         print ('our nmi is: ', normalized_mutual_info_score(our_labels, true_labels))
    
#     if plot:
#         print('our method clusters')
#         for label_name in list(set(our_labels)):
#             plt.scatter(points[np.where(our_labels == label_name)[0]][:, 0], points[np.where(our_labels == label_name)[0]][:, 1], label='0')

#         plt.show()
#         plt.close()
    
    
#     start = datetime.datetime.now()
#     labels = spectral_clustering(A, n_clusters=n_clusters)
#     spectral_accuracy = accuracy(labels, true_labels)
#     spectral_time = datetime.datetime.now() - start
#     if is_print:
#         print ('spectral clustering time is:', spectral_time)
#         print ('spectral clustering accuracy is:', spectral_accuracy)
#         print ('spectral clustering nmi is: ', normalized_mutual_info_score(labels, true_labels))
    
#     if plot:
#         print('spectral clustering clusters')
#         for label_name in list(set(labels)):
#             plt.scatter(points[np.where(labels == label_name)[0]][:, 0], points[np.where(labels == label_name)[0]][:, 1], label='0')

#         plt.show()
#         plt.close()
    
#         print('true clusters')
#         for label_name in list(set(labels)):
#             plt.scatter(points[np.where(true_labels == label_name)[0]][:, 0], points[np.where(true_labels == label_name)[0]][:, 1], label='0')

#         plt.show()
#         plt.close()
#     return our_accuracy, our_time, spectral_accuracy, spectral_time

     
# def run_more_plots_all_together(points, true_labels, K, alpha, lambda_nLasso, threshhold, n_clusters, tmp=0, M=0.2, plot=False, is_print=True):
#     A, B, weight_vec = get_B_and_weight_vec_ring(points, threshhold=threshhold)

#     E, N = B.shape

#     samplingset = random.choices(np.where(true_labels==tmp)[0], k=int(M * len(np.where(true_labels==tmp)[0])))
# #     samplingset = random.choices(true_labels, k=int(M * len(true_labels)))
#     seednodesindicator = np.zeros(N)
#     seednodesindicator[samplingset] = 1
#     noseednodeindicator = np.ones(N)
#     noseednodeindicator[samplingset] = 0

#     Gamma_vec = np.array(1. / (np.sum(abs(B), 0)))[0]  # \in [0, 1]
#     Gamma = np.diag(Gamma_vec)

#     Sigma = 0.5

#     fac_alpha = 1. / (Gamma_vec * alpha + 1)  # \in [0, 1]

#     hatx = np.zeros(N)
#     newx = np.zeros(N)
#     prevx = np.zeros(N)
#     haty = np.array([x / (E - 1) for x in range(0, E)])
#     lambda_weight = lambda_nLasso * weight_vec
#     gamma_plus = 1 + Gamma_vec[samplingset]
#     start = datetime.datetime.now()
#     for iterk in range(K):
#         tildex = 2 * hatx - prevx
#         newy = haty + Sigma * B.dot(tildex)  # chould be negative
#         res = abs(newy) / lambda_weight
#         res[res < 1] = 1
#         haty = newy / res

#         newx = hatx - Gamma_vec * B.T.dot(haty)  # could  be negative

#         newx[samplingset] = (newx[samplingset] + Gamma_vec[samplingset]) / gamma_plus

#         newx = seednodesindicator * newx + noseednodeindicator * (newx * fac_alpha)
#         prevx = np.copy(hatx)
#         hatx = newx  # could be negative
#     our_time = datetime.datetime.now() - start
#     X = newx
#     X = np.nan_to_num(X, 0)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X.reshape(len(X), 1))
#     our_accuracy = accuracy(kmeans.labels_, true_labels)
#     if is_print:
#         print('our time is:', our_time)
#         print ('our accuracy is:', our_accuracy)
    
#     if plot:
#         print('our method clusters')
#         for label_name in list(set(kmeans.labels_)):
#             plt.scatter(points[np.where(kmeans.labels_ == label_name)[0]][:, 0], points[np.where(kmeans.labels_ == label_name)[0]][:, 1], label='0')
    
#         plt.show()
#         plt.close()
    

#     start = datetime.datetime.now()
#     labels = spectral_clustering(A, n_clusters=n_clusters)
#     spectral_time = datetime.datetime.now() - start
#     spectral_accuracy = accuracy(labels, true_labels)
#     if is_print:
#         print ('spectral clustering time is:', spectral_time)
#         print ('spectral clustering accuracy is:', spectral_accuracy)

#     if plot:
#         print('spectral clustering clusters')
#         for label_name in list(set(labels)):
#             plt.scatter(points[np.where(labels == label_name)[0]][:, 0], points[np.where(labels == label_name)[0]][:, 1], label='0')
    
#         plt.show()
#         plt.close()
    
#     if plot:
#         print('true clusters')
#         for label_name in list(set(labels)):
#             plt.scatter(points[np.where(true_labels == label_name)[0]][:, 0], points[np.where(true_labels == label_name)[0]][:, 1], label='0')
    
#         plt.show()
#         plt.close()

#     return our_accuracy, our_time, spectral_accuracy, spectral_time


def get_iter_score(node_labels):
    clust_labels = np.array([len(np.where(node_labels==i)[0]) for i in range(2)])
    clust_labels = clust_labels / len(node_labels)
    return clust_labels


def run_more_plots(points, true_labels, K, alpha, lambda_nLasso, threshhold, n_clusters, M=0.2, plot=False, is_print=True, auto=False):
    A, B, weight_vec = get_B_and_weight_vec_ring(points, threshhold=threshhold)

    E, N = B.shape

    Gamma_vec = np.array(1. / (np.sum(abs(B), 0)))[0]  # \in [0, 1]
    Gamma = np.diag(Gamma_vec)

    Sigma = 0.5

    fac_alpha = 1. / (Gamma_vec * alpha + 1)  # \in [0, 1]
    lambda_weight = lambda_nLasso * weight_vec

    our_labels = np.full(N, n_clusters-1)
    our_time = datetime.datetime.now() - datetime.datetime.now()
    
    samples = {}
    nonsamples = {}
    for clust_num in range(n_clusters):
        samples[clust_num] = random.choices(np.where(true_labels==clust_num)[0], k=int(M * len(np.where(true_labels==clust_num)[0])))
    
    for clust_num1 in range(n_clusters):
        clust_nonsamples = []
        for clust_num2 in range(n_clusters):
            if clust_num1 == clust_num2:
                continue
            clust_nonsamples += samples[clust_num2]
        nonsamples[clust_num1] = clust_nonsamples
    
    for clust_num in range(n_clusters-1):

        samplingset = samples[clust_num]
        seednodesindicator = np.zeros(N)
        seednodesindicator[samplingset] = 1
        noseednodeindicator = np.ones(N)
        noseednodeindicator[samplingset] = 0


        hatx = np.zeros(N)
        newx = np.zeros(N)
        prevx = np.zeros(N)
        haty = np.array([x / (E - 1) for x in range(0, E)])
        gamma_plus = 1 + Gamma_vec[samplingset]
        prev_sample_count = -1
        prev_nonsample_count = -1
        prev_newx = np.zeros(N)
        start = datetime.datetime.now()
        for iterk in range(K):
            tildex = 2 * hatx - prevx
            newy = haty + Sigma * B.dot(tildex)  # chould be negative
            res = abs(newy) / lambda_weight
            res[res < 1] = 1
            haty = newy / res

            newx = hatx - Gamma_vec * B.T.dot(haty)  # could  be negative

            newx[samplingset] = (newx[samplingset] + Gamma_vec[samplingset]) / gamma_plus

            newx = seednodesindicator * newx + noseednodeindicator * (newx * fac_alpha)
            prevx = np.copy(hatx)
            hatx = newx  # could be negative
            
            if auto:
                if iterk % 5 == 0 and iterk >= 10 :
                    X = newx
                    X = np.nan_to_num(X, 0)
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(X.reshape(len(X), 1))
                    sampling_labels = np.array(kmeans.labels_[samplingset])
                    sampling_count = get_iter_score(sampling_labels)
                    max_sampling_count = np.max(sampling_count)


                    nonsampling_labels = np.array(kmeans.labels_[nonsamples[clust_num]])
                    nonsample_count = get_iter_score(nonsampling_labels)
                    max_nonsampling_count = np.max(nonsample_count)


                    if sampling_count[0] == max_sampling_count and nonsample_count[0] == max_nonsampling_count:
                        continue

                    if (max_sampling_count == 1 or max_sampling_count == prev_sample_count) and (max_nonsampling_count == prev_nonsample_count or max_nonsampling_count == 1):
                        break

                    prev_sample_count = max_sampling_count
                    prev_nonsample_count = max_nonsampling_count

                
        print("cluster num:", clust_num, ", iterk:", iterk)
        our_time += datetime.datetime.now() - start
        X = newx
        X = np.nan_to_num(X, 0)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X.reshape(len(X), 1))
        matched_label = kmeans.labels_[samplingset][0]
        our_labels[np.where(kmeans.labels_ == matched_label)[0]] = clust_num

#     our_accuracy = accuracy(our_labels, true_labels)
#     if is_print:
#         print ('our time is:', our_time)
#         print ('our accuracy is:', our_accuracy)
    
    if plot:
        print('our method clusters')
        for label_name in list(set(our_labels)):
            plt.scatter(points[np.where(our_labels == label_name)[0]][:, 0], points[np.where(our_labels == label_name)[0]][:, 1], label='0')

        plt.show()
        plt.close()
    
    
    start = datetime.datetime.now()
    labels = spectral_clustering(A, n_clusters=n_clusters)
    spectral_accuracy = accuracy(labels, true_labels)
    spectral_time = datetime.datetime.now() - start
#     if is_print:
#         print ('spectral clustering time is:', spectral_time)
#         print ('spectral clustering accuracy is:', spectral_accuracy)
    
#     if plot:
#         print('spectral clustering clusters')
#         for label_name in list(set(labels)):
#             plt.scatter(points[np.where(labels == label_name)[0]][:, 0], points[np.where(labels == label_name)[0]][:, 1], label='0')

#         plt.show()
#         plt.close()
    
#         print('true clusters')
#         for label_name in list(set(labels)):
#             plt.scatter(points[np.where(true_labels == label_name)[0]][:, 0], points[np.where(true_labels == label_name)[0]][:, 1], label='0')

#         plt.show()
#         plt.close()
    out = {
        'our_labels': our_labels,
        'spectral_labels': labels,
        'our_time': our_time,
        'spectral_time': spectral_time
    }
    
    return out