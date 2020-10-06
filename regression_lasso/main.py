import numpy as np
from stochastic_block_model import get_B_and_weight_vec
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import random


def nmse(Y, Y_pred):
    MSE = np.square(np.subtract(Y, Y_pred)).mean()
    NMSE = MSE / np.square(Y).mean()
    return NMSE


def run(K, B, weight_vec, Y, X, lambda_lasso=0.1, method=None, M=0.2):
    if method == 'log':
        score_func = mean_squared_log_error
    elif method == 'norm':
        score_func = nmse
    else:
        score_func = mean_squared_error

    Sigma = np.diag(1./(2*weight_vec))

    D = B

    Gamma_vec = (1.0/(np.sum(abs(B), 0))).T
    Gamma = np.diag(Gamma_vec)

    if np.linalg.norm(np.dot(Sigma**0.5, D).dot(Gamma**0.5), 2) > 1:
        print ('product norm', np.linalg.norm(np.dot(Sigma**0.5, D).dot(Gamma**0.5), 2))
        # raise Exception('higher than 1')

    E, N = B.shape
    m, n = X[0].shape
    samplingset = random.sample([i for i in range(N)], k=int(M * N))
    not_samplingset = [i for i in range(N) if i not in samplingset]
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

    our_scores = []
    for iterk in range(K):
        if iterk % 100 == 0:
            print ('iter:', iterk)
        prev_w = np.copy(new_w)

        hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))  # could  be negative

        for i in range(N):
            if i in samplingset:
                mtx_inv = MTX1_INV[i]
                mtx2 = hat_w[i] + MTX2[i]

                new_w[i] = np.dot(mtx_inv, mtx2)
            else:
                new_w[i] = hat_w[i]

        tilde_w = 2 * new_w - prev_w
        new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))  # chould be negative

        normalized_u = np.where(abs(new_u) >= lambda_lasso)
        new_u[normalized_u] = lambda_lasso * new_u[normalized_u] / abs(new_u[normalized_u])

        Y_pred = []
        for i in range(N):
            Y_pred.append(np.dot(X[i], new_w[i]))

        our_scores.append(score_func(Y, Y_pred))

    # if np.max(abs(new_w - prev_w)) > 5 * 1e-3:
    print (np.max(abs(new_w - prev_w)))
        # raise Exception('not converged')
    # print ('our mean_squared_log_error', our_scores[-1])

    x = np.mean(X, 1)
    y = np.mean(Y, 1)
    model = LinearRegression().fit(x[samplingset], y[samplingset])

    linear_regression_score = score_func(y, model.predict(x))
    # print ('linear_regression mean_squared_log_error', linear_regression_score)

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
    decision_tree_score = score_func(y, pred_y)
    # print ('decision_tree mean_squared_log_error', decision_tree_score)

    # print ('\tdecision tree max_depth:', max_depth,
    #        '\n\t\ttrain error:', score_func(y[decision_tree_samplingset], regressor.predict(x[decision_tree_samplingset])),
    #        '\n\t\ttest error:', score_func(y[decision_tree_non_samplingset], regressor.predict(x[decision_tree_non_samplingset])))

    return our_scores, linear_regression_score, decision_tree_score
