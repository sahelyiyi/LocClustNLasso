import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error
import random

from utils import get_matrices


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

    Sigma, Gamma, Gamma_vec, D = get_matrices(weight_vec, B)

    E, N = B.shape
    m, n = X[0].shape
    samplingset = random.sample([i for i in range(N)], k=int(M * N))
    not_samplingset = [i for i in range(N) if i not in samplingset]
    new_w = np.array([np.zeros(n) for i in range(N)])
    prev_w = np.array([np.zeros(n) for i in range(N)])
    new_u = np.array([np.zeros(n) for i in range(E)])

    MTX2 = {}
    BETA = {}
    for i in samplingset:
        MTX2[i] = (Gamma_vec[i]/m * X[i]).T
        BETA[i] = Gamma_vec[i] * np.power(np.linalg.norm(X[i]), 2)

    limit = np.array([np.zeros(n) for i in range(E)])
    for i in range(n):
        limit[:, i] = lambda_lasso*weight_vec
    iteration_scores = []
    for iterk in range(K):
        if iterk % 100 == 0:
            print ('iter:', iterk)
        prev_w = np.copy(new_w)

        hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))  # could  be negative
        new_w = np.copy(hat_w)

        for i in range(N):
            if i in samplingset:
                phi_i = lambda vu: hat_w[i] + np.dot(MTX2[i], (1 / (1 + np.exp(-np.dot(X[i], vu))) - Y[i]))
                try:
                    lp = int(np.ceil(2 * np.log(iterk) / np.log(1 / BETA[i])))
                except:
                    lp = 10
                lp = max(lp, 10)
                for j in range(lp):
                    new_w[i] = phi_i(new_w[i])
            else:
                new_w[i] = hat_w[i]

        tilde_w = 2 * new_w - prev_w
        new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))  # chould be negative

        normalized_u = np.where(abs(new_u) >= limit)
        new_u[normalized_u] = limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])

    Y_pred = []
    for i in range(N):
        if np.dot(X[i], new_w[i]) > 0:
            Y_pred.append(0)
        else:
            Y_pred.append(1)
    Y_pred = np.array(Y_pred)
    # iteration_scores.append(default_score_func(Y, Y_pred))
    print (len(np.where(Y!=Y_pred)[0])/N)
    # if np.max(abs(new_w - prev_w)) > 5 * 1e-3:
    # print (np.max(abs(new_w - prev_w)))
    #     # raise Exception('not converged')
    #
    # our_score = {}
    # for score_func_name, score_func in functions.items():
    #     our_score[score_func_name] = score_func(Y, Y_pred)
    #
    # x = np.mean(X, 1)
    # y = np.mean(Y, 1)
    # model = LinearRegression().fit(x[samplingset], y[samplingset])
    # pred_y = model.predict(x)
    # linear_regression_score = {}
    # for score_func_name, score_func in functions.items():
    #     linear_regression_score[score_func_name] = score_func(y, pred_y)
    #
    # y = Y.reshape(-1, 1)
    # x = X.reshape(-1, n)
    # decision_tree_samplingset = []
    # for item in samplingset:
    #     for i in range(m):
    #         decision_tree_samplingset.append(m*item+i)
    #
    # max_depth = 5
    # regressor = DecisionTreeRegressor(max_depth=max_depth)
    # regressor.fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    # pred_y = regressor.predict(x)
    # decision_tree_score = {}
    # for score_func_name, score_func in functions.items():
    #     decision_tree_score[score_func_name] = score_func(y, pred_y)
    #
    # # decision_tree_non_samplingset = [i for i in range(len(x)) if i not in decision_tree_samplingset]
    # # print ('\tdecision tree max_depth:', max_depth,
    # #        '\n\t\ttrain error:', default_score_func(y[decision_tree_samplingset], regressor.predict(x[decision_tree_samplingset])),
    # #        '\n\t\ttest error:', default_score_func(y[decision_tree_non_samplingset], regressor.predict(x[decision_tree_non_samplingset])))
    #
    # return iteration_scores, our_score, linear_regression_score, decision_tree_score
