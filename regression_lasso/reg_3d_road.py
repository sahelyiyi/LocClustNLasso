# import numpy as np
# from math import sqrt
# from collections import defaultdict, Counter
#
#
# with open('/Users/sahel/Downloads/3D_spatial_network.txt', 'r') as f:
#     data = f.read().split('\n')
#
# data = data[:-1]
# data = data[:1000]
# fixed_data = []
# for item in data:
#     item = item.split(',')
#     item0, item1, item2, item3 = item[0], float(item[1]), float(item[2]), float(item[3])
#     fixed_data.append([item0, item1, item2, item3])
#
# lats = sorted(fixed_data, key=lambda x: x[1])
# longs = sorted(fixed_data, key=lambda x: x[2])
#
#
# E = 0
# neighbours = defaultdict(list)
# degrees = Counter()
# for i in range(len(lats)):
#     lat1, long1 = lats[i][1], lats[i][2]
#     for j in range(i+1, len(lats)):
#         lat2, long2 = lats[j][1], lats[j][2]
#         dist = sqrt((lat1-lat2)**2 + (long1-long2)**2)
#         if dist > 0.01:
#             break
#         if dist == 0:
#             continue
#         neighbours[(lat1, long1)].append(((lat2, long2), dist))
#         degrees[(lat1, long1)] += 1
#         degrees[(lat2, long2)] += 1
#         E += 1
#
# for i in range(len(longs)):
#     lat1, long1 = lats[i][1], lats[i][2]
#     for j in range(i+1, len(longs)):
#         lat2, long2 = lats[j][1], lats[j][2]
#         dist = sqrt((lat1-lat2)**2 + (long1-long2)**2)
#         if dist > 0.01:
#             break
#         if dist == 0:
#             continue
#         neighbours[(lat1, long1)].append(((lat2, long2), dist))
#         degrees[(lat1, long1)] += 1
#         degrees[(lat2, long2)] += 1
#         E += 1
#
# cnt = 0
# node_indices = {}
# for item in fixed_data:
#     lat, log = item[1], item[2]
#     if degrees[(lat, log)] == 0:
#         continue
#     if (lat, log) in node_indices:
#         continue
#     node_indices[(lat, log)] = cnt
#     cnt += 1
#
# N = len(node_indices)
# X = np.array([[[0, 0]] for i in range(N)])
# Y = np.array([[0] for i in range(N)])
# for item in fixed_data:
#     lat, log = item[1], item[2]
#     if (lat, log) not in node_indices:
#         continue
#
#     idx = node_indices[(lat, log)]
#     X[idx] = [lat, log]
#     Y[idx] = [item[3]]
#
# X = np.array(X)
# Y = np.array(Y)
# m, n = 1, 2
#
#
# B = np.zeros((E, N))
# D = np.zeros((E, N))
# weight_vec = np.zeros(E)
# cnt = 0
# for item1 in neighbours:
#     idx1 = node_indices[item1]
#     for item2, dist in neighbours[item1]:
#         idx2 = node_indices[item2]
#         if idx1 < idx2:
#             B[cnt, idx1] = 1
#             D[cnt, idx1] = dist
#
#             B[cnt, idx2] = -1
#             D[cnt, idx2] = -dist
#         else:
#             B[cnt, idx1] = -1
#             D[cnt, idx1] = -dist
#
#             B[cnt, idx2] = 1
#             D[cnt, idx2] = dist
#         weight_vec[cnt] = dist
#         cnt += 1
#
#
# weight = np.diag(weight_vec)
# Sigma = np.diag(1./(2*weight_vec))
#
#
# Lambda = 1*np.diag(1./(np.sum(abs(B), 1)))
# Gamma_vec = (.9/(np.sum(abs(B), 0))).T  # \in [0, 1]
# Gamma = np.diag(Gamma_vec)
#
# lambda_nLasso = 1/3  # nLasso parameter
#
# hat_w = np.array([np.zeros(n) for i in range(N)])
# new_w = np.array([np.zeros(n) for i in range(N)])
# prev_w = np.array([np.zeros(n) for i in range(N)])
# new_u = np.array([np.zeros(n) for i in range(E)])
#
# K = 100
# for iterk in range(K):
#     # print ('iter:', iterk)
#
#     tilde_w = 2 * hat_w - prev_w
#     new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))  # chould be negative
#
#     hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))  # could  be negative
#
#     for i in range(N):
#         mtx1 = 1.8 * np.dot(X[i].T, X[i]).astype('float64')
#         if mtx1.shape:
#             mtx1 += Gamma_vec[i] * np.eye(mtx1.shape[0])
#             mtx_inv = np.linalg.inv(mtx1)
#         else:
#             mtx1 += Gamma_vec[i]
#             mtx_inv = 1.0 / mtx1
#
#         mtx2 = Gamma_vec[i] * hat_w[i] + 1.8 * np.dot(X[i].T, Y[i])
#
#         new_w[i] = np.dot(mtx_inv, mtx2)
#     prev_w = np.copy(new_w)
#
# # mse = 0
# # for i in range(N):
# #     print (Y[i], np.dot(X[i], new_w[i]))
# #     mse += np.linalg.norm(Y[i] - np.dot(X[i], new_w[i]))
# # mse /= N
#
