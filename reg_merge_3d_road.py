import numpy as np
import random
from math import sqrt
from collections import defaultdict, Counter


with open('/Users/sahel/Downloads/3D_spatial_network.txt', 'r') as f:
    data = f.read().split('\n')

data = data[:-1]
data = data[:200000]
fixed_data = []
for item in data:
    item = item.split(',')
    item0, item1, item2, item3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
    fixed_data.append([item0, item1, item2, item3])

sorted_lats = sorted(fixed_data, key=lambda x: x[3])

merged_data = []
for i in range(int(len(sorted_lats)/5)):
    merge_cell = []
    for j in range(5):
        merge_cell.append(sorted_lats[i*5+j])
    merged_data.append(merge_cell)
merged_data = np.array(merged_data)

sorted_merged = sorted(merged_data, key=lambda x: (np.mean(x[:,1]), np.mean(x[:,2])))

mean_merged = []
for merge_cell in sorted_merged:
    mean_merged.append((np.mean(merge_cell[:, 1]), np.mean(merge_cell[:, 2])))

E = 0
MAX_DIST = 0.01
neighbours = defaultdict(list)
degrees = Counter()
for i in range(len(mean_merged)):
    # if i % 1000 == 0:
    #     print (i)
    lat1, long1 = mean_merged[i][0], mean_merged[i][1]
    for j in range(i+1, len(mean_merged)):
        lat2, long2 = mean_merged[j][0], mean_merged[j][1]
        dist = sqrt((lat1-lat2)**2 + (long1-long2)**2)
        if dist >= MAX_DIST:
            break
        if dist == 0:
            continue
        if ((lat2, long2), dist) in neighbours[(lat1, long1)]:
            continue
        neighbours[(lat1, long1)].append(((lat2, long2), MAX_DIST-dist))
        degrees[(lat1, long1)] += 1
        degrees[(lat2, long2)] += 1
        E += 1


sorted_merged = sorted(merged_data, key=lambda x: (np.mean(x[:,2]), np.mean(x[:,1])))

mean_merged = []
for merge_cell in sorted_merged:
    mean_merged.append((np.mean(merge_cell[:, 1]), np.mean(merge_cell[:, 2])))

for i in range(len(mean_merged)):
    # if i % 1000 == 0:
    #     print (i)
    lat1, long1 = mean_merged[i][0], mean_merged[i][1]
    for j in range(i + 1, len(mean_merged)):
        lat2, long2 = mean_merged[j][0], mean_merged[j][1]
        dist = sqrt((lat1 - lat2) ** 2 + (long1 - long2) ** 2)
        if dist >= MAX_DIST:
            break
        if dist == 0:
            continue
        if ((lat2, long2), dist) in neighbours[(lat1, long1)]:
            continue
        neighbours[(lat1, long1)].append(((lat2, long2), MAX_DIST-dist))
        degrees[(lat1, long1)] += 1
        degrees[(lat2, long2)] += 1
        E += 1

cnt = 0
node_indices = {}
for item in mean_merged:
    lat, log = item[0], item[1]
    if degrees[(lat, log)] == 0:
        continue
    if (lat, log) in node_indices:
        continue
    node_indices[(lat, log)] = cnt
    cnt += 1

N = len(node_indices)
X = np.zeros((N, 5, 2))
Y = np.zeros((N, 5, 1))
for i, item in enumerate(mean_merged):
    lat, log = item[0], item[1]
    if (lat, log) not in node_indices:
        continue

    idx = node_indices[(lat, log)]
    X[idx] = np.array([sorted_merged[i][:, 1], sorted_merged[i][:, 2]]).T
    Y[idx] = np.array([sorted_merged[i][:, 3]]).T

m, n = X[0].shape
M = [i for i in range(N)]
M = random.choices([i for i in range(N)], k=100000)

B = np.zeros((E, N))
D = np.zeros((E, N))
weight_vec = np.zeros(E)
cnt = 0
for item1 in neighbours:
    idx1 = node_indices[item1]
    for item2, dist in neighbours[item1]:
        idx2 = node_indices[item2]
        if idx1 < idx2:
            B[cnt, idx1] = 1
            D[cnt, idx1] = dist

            B[cnt, idx2] = -1
            D[cnt, idx2] = -dist
        else:
            B[cnt, idx1] = -1
            D[cnt, idx1] = -dist

            B[cnt, idx2] = 1
            D[cnt, idx2] = dist
        weight_vec[cnt] = dist
        cnt += 1


weight = np.diag(weight_vec)
Sigma = np.diag(1./(2*weight_vec))


Lambda = 1*np.diag(1./(np.sum(abs(B), 1)))
Gamma_vec = (.9/(np.sum(abs(B), 0))).T  # \in [0, 1]
Gamma = np.diag(Gamma_vec)

lambda_nLasso = 1/3  # nLasso parameter

hat_w = np.array([np.zeros(n) for i in range(N)])
new_w = np.array([np.zeros(n) for i in range(N)])
prev_w = np.array([np.zeros(n) for i in range(N)])
new_u = np.array([np.zeros(n) for i in range(E)])

K = 500
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

            # mtx2 = Gamma_vec[i] * hat_w[i] + 1.8 * np.dot(X[i].T, Y[i])
            mtx2 = Gamma_vec[i] * hat_w[i] + 1.8 * np.dot(X[i].T, Y[i]).T[0]

            new_w[i] = np.dot(mtx_inv, mtx2)
        else:
            new_w[i] = hat_w[i]
    prev_w = np.copy(new_w)

mse = 0
mse1 = 0
for i in range(N):
    # print (Y[i].T[0], np.dot(X[i], new_w[i]), Y[i].T[0] - np.dot(X[i], new_w[i]))
    mse1 += np.linalg.norm(Y[i].T[0] - np.dot(X[i], new_w[i]))
    mse += (np.linalg.norm(Y[i] - np.dot(X[i], new_w[i])) / np.linalg.norm(Y[i]))
mse /= (m*N)
mse1 /= (m*N)

