import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error
import random


def nmse_func(Y, Y_pred):
    MSE = mean_squared_error(Y, Y_pred)
    NMSE = MSE / np.square(Y).mean()
    return NMSE


def run(K, B, weight_vec, Y, X, lambda_lasso=0.1, method=None, M=0.2):
    functions = {
        'mean_squared_log_error': mean_squared_log_error,
        'mean_squared_error': mean_squared_error,
        'normalized_mean_squared_error': nmse_func,
        'mean_absolute_error': mean_absolute_error
    }

    if method == 'log':
        default_score_func = mean_squared_log_error
    elif method == 'norm':
        default_score_func = nmse_func
    else:
        default_score_func = mean_squared_error

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

    limit = np.array([np.zeros(n) for i in range(E)])
    for i in range(n):
        limit[:, i] = lambda_lasso*weight_vec
    iteration_scores = []
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

        normalized_u = np.where(abs(new_u) >= limit)
        new_u[normalized_u] = limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])

        Y_pred = []
        for i in range(N):
            Y_pred.append(np.dot(X[i], new_w[i]))
        iteration_scores.append(default_score_func(np.abs(Y), np.abs(Y_pred)))

    # if np.max(abs(new_w - prev_w)) > 5 * 1e-3:
    print (np.max(abs(new_w - prev_w)))
        # raise Exception('not converged')

    our_score = {}
    for score_func_name, score_func in functions.items():
        our_score[score_func_name] = score_func(np.abs(Y), np.abs(Y_pred))

    y = Y.reshape(-1, 1)
    x = X.reshape(-1, n)
    decision_tree_samplingset = []
    for item in samplingset:
        for i in range(m):
            decision_tree_samplingset.append(m * item + i)

    model = LinearRegression().fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = model.predict(x)
    linear_regression_score = {}
    for score_func_name, score_func in functions.items():
        linear_regression_score[score_func_name] = score_func(np.abs(y), np.abs(pred_y))

    max_depth = 5
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    regressor.fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = regressor.predict(x)
    decision_tree_score = {}
    for score_func_name, score_func in functions.items():
        decision_tree_score[score_func_name] = score_func(np.abs(y), np.abs(pred_y))

    # decision_tree_non_samplingset = [i for i in range(len(x)) if i not in decision_tree_samplingset]
    # print ('\tdecision tree max_depth:', max_depth,
    #        '\n\t\ttrain error:', default_score_func(y[decision_tree_samplingset], regressor.predict(x[decision_tree_samplingset])),
    #        '\n\t\ttest error:', default_score_func(y[decision_tree_non_samplingset], regressor.predict(x[decision_tree_non_samplingset])))

    return iteration_scores, our_score, linear_regression_score, decision_tree_score
