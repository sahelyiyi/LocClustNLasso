import numpy as np
import random
from math import sqrt
from collections import defaultdict, Counter
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from regression_lasso.main import nmse_func


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
samplingset = random.sample([i for i in range(N)], k=int(0.7 * N))

B = np.zeros((E, N))
# D = np.zeros((E, N))
weight_vec = np.zeros(E)
cnt = 0
for item1 in neighbours:
    idx1 = node_indices[item1]
    for item2, dist in neighbours[item1]:
        idx2 = node_indices[item2]
        if idx1 < idx2:
            B[cnt, idx1] = 1
            # D[cnt, idx1] = dist

            B[cnt, idx2] = -1
            # D[cnt, idx2] = -dist
        else:
            B[cnt, idx1] = -1
            # D[cnt, idx1] = -dist

            B[cnt, idx2] = 1
            # D[cnt, idx2] = dist
        weight_vec[cnt] = dist
        cnt += 1

D = B


weight = np.diag(weight_vec)
Sigma = np.diag(1./(2*weight_vec))


Gamma_vec = (1.0/(np.sum(abs(B), 0))).T  # \in [0, 1]
Gamma = np.diag(Gamma_vec)

lambda_lasso = 0.1  # nLasso parameter
lambda_lasso = 0.08  # nLasso parameter

hat_w = np.array([np.zeros(n) for i in range(N)])
new_w = np.array([np.zeros(n) for i in range(N)])
prev_w = np.array([np.zeros(n) for i in range(N)])
new_u = np.array([np.zeros(n) for i in range(E)])


MTX1_INV = {}
MTX2 = {}
for i in samplingset:
    mtx1 = 2 * Gamma_vec[i] * np.dot(X[i].T, X[i]).astype('float64')
    if mtx1.shape:
        mtx1 += 1 * np.eye(mtx1.shape[0])
        mtx_inv = np.linalg.inv(mtx1)
    else:
        mtx1 += 1
        mtx_inv = 1.0 / mtx1
    MTX1_INV[i] = mtx_inv

    MTX2[i] = 2 * Gamma_vec[i] * np.dot(X[i].T, Y[i]).T[0]


K = 1000
limit = np.array([np.zeros(n) for i in range(E)])
for i in range(n):
    limit[:, i] = lambda_lasso*weight_vec
for iterk in range(K):
    if iterk % 100 == 0:
        print ('iter:', iterk)
    prev_w = np.copy(new_w)

    hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))  # could  be negative

    for i in range(N):
        if i in samplingset:
            mtx2 = hat_w[i] + MTX2[i]
            mtx_inv = MTX1_INV[i]

            new_w[i] = np.dot(mtx_inv, mtx2)
        else:
            new_w[i] = hat_w[i]

    tilde_w = 2 * new_w - prev_w
    new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))  # chould be negative

    normalized_u = np.where(abs(new_u) >= limit)
    new_u[normalized_u] = limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])


Y_pred = []
for i in range(N):
    Y_pred.append(np.dot(X[i], new_w[i]))

NMSE = nmse_func(Y.reshape(N, 5), Y_pred)
print('NMSE', NMSE)

x = np.mean(X, 1)
y = np.mean(Y, 1)
model = LinearRegression().fit(x[samplingset], y[samplingset])

linear_regression_score = nmse_func(y, model.predict(x))
print('linear_regression_score', linear_regression_score)

y = Y.reshape(-1, 1)
x = X.reshape(-1, 2)
decision_tree_samplingset = []
for item in samplingset:
    for i in range(m):
        decision_tree_samplingset.append(m*item+i)
decision_tree_non_samplingset = [i for i in range(len(x)) if i not in decision_tree_samplingset]

max_depth = 5
regressor = DecisionTreeRegressor(max_depth=max_depth)
regressor.fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
pred_y = regressor.predict(x)
decision_tree_score = nmse_func(y, pred_y)
print ('decision_tree_score', decision_tree_score)


# lambda=1, M=0.7
# NMSE 0.22784964082832004

# lambda=0.5, M=0.7
# NMSE 0.2349845635777743
# linear_regression_score 0.372930118135755
# decision_tree_score 0.5804098158147778

# lambda=0.1, M=0.7
# NMSE 0.20356074162702414


# lambda=0.08, M=0.7
# NMSE 0.21768699584301643

# lambda=0.05, M=0.7
# NMSE 0.22918163759820392

# lambda=0.5, M=0.6
# NMSE 0.2747315880798612
# linear_regression_score 0.372930118135755
# decision_tree_score 0.5804098158147778

# lambda=0.2, M=0.6
# NMSE 0.2826136578917171
