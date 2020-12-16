import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error
import random
from collections import defaultdict

from utils import get_matrices, get_preprocessed_matrices


def nmse_func(Y, Y_pred):
    MSE = mean_squared_error(Y, Y_pred)
    NMSE = MSE / np.square(Y).mean()
    return NMSE


def run(K, B, weight_vec, Y, X, W, samplingset, lambda_lasso=0.1, method=None):
    functions = {
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

    Sigma, Gamma, Gamma_vec, D = get_matrices(weight_vec, B)

    E, N = B.shape
    m, n = X[0].shape

    MTX1_INV, MTX2 = get_preprocessed_matrices(samplingset, Gamma_vec, X, Y)

    limit = np.array([np.zeros(n) for i in range(E)])
    for i in range(n):
        limit[:, i] = lambda_lasso*weight_vec

    not_samplingset = [i for i in range(N) if i not in samplingset]
    new_w = np.array([np.zeros(n) for i in range(N)])
    prev_w = np.array([np.zeros(n) for i in range(N)])
    new_u = np.array([np.zeros(n) for i in range(E)])
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

        normalized_u = np.where((abs(new_u) >= limit) & (new_u != 0))
        new_u[normalized_u] = limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])

        iteration_scores.append(mean_squared_error(W, new_w))

    print (np.max(abs(new_w - prev_w)))

    Y_pred = []
    for i in range(N):
        Y_pred.append(np.dot(X[i], new_w[i]))
    Y_pred = np.array(Y_pred)

    our_score = defaultdict(dict)
    for score_func_name, score_func in functions.items():
        our_score['total'][score_func_name] = score_func(Y, Y_pred)
        our_score['train'][score_func_name] = score_func(Y[samplingset], Y_pred[samplingset])
        our_score['test'][score_func_name] = score_func(Y[not_samplingset], Y_pred[not_samplingset])

    y = Y.reshape(-1, 1)
    x = X.reshape(-1, n)
    decision_tree_samplingset = []
    for item in samplingset:
        for i in range(m):
            decision_tree_samplingset.append(m * item + i)
    decision_tree_not_samplingset = [i for i in range(5*N) if i not in decision_tree_samplingset]

    model = LinearRegression().fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = model.predict(x)
    linear_regression_score = defaultdict(dict)
    for score_func_name, score_func in functions.items():
        linear_regression_score['total'][score_func_name] = score_func(y, pred_y)
        linear_regression_score['train'][score_func_name] = score_func(y[decision_tree_samplingset], pred_y[decision_tree_samplingset])
        linear_regression_score['test'][score_func_name] = score_func(y[decision_tree_not_samplingset], pred_y[decision_tree_not_samplingset])

    max_depth = 2
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    regressor.fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = regressor.predict(x)
    decision_tree_score = defaultdict(dict)
    for score_func_name, score_func in functions.items():
        decision_tree_score['total'][score_func_name] = score_func(y, pred_y)
        decision_tree_score['train'][score_func_name] = score_func(y[decision_tree_samplingset], pred_y[decision_tree_samplingset])
        decision_tree_score['test'][score_func_name] = score_func(y[decision_tree_not_samplingset], pred_y[decision_tree_not_samplingset])

    return iteration_scores, our_score, linear_regression_score, decision_tree_score
