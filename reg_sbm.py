import numpy as np
from stochastic_block_model import get_B_and_weight_vec
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import random


def run_reg_sbm(K=300, N1=150, N2=150, W1=[0.5, 0.7], W2=[2, 4], lambda_lasso=0.1, method=None):
    if method == 'log':
        score_func = mean_squared_log_error
    else:
        score_func = mean_squared_error

    B, weight_vec = get_B_and_weight_vec(N1, N2, mu_in=40, mu_out=10)

    E, N = B.shape
    samplingset = random.sample([i for i in range(N)], k=int(0.3*N))
    not_samplingset = [i for i in range(N) if i not in samplingset]


    X = []
    for i in range(N):
        x, y = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 5))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 1.0, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        X.append(g)
    X = np.array(X)

    W1 = np.array(W1)
    W2 = np.array(W2)
    # W1 = np.random.random((2, 1))
    # W1 = np.array([10, 15])
    # W2 = random.randint(2, 10) * np.random.random((2, 1)) + random.randint(5, 10)

    Y = []
    for i in range(N):
        if i < N1:
            Y.append(np.dot(X[i], W1))
        else:
            Y.append(np.dot(X[i], W2))
    Y = np.array(Y)
    m, n = X[0].shape

    Sigma = np.diag(1./(2*weight_vec))

    # D = np.dot(weight, B)
    D = B

    Gamma_vec = (1.0/(np.sum(abs(B), 0))).T  # \in [0, 1]
    Gamma = np.diag(Gamma_vec)

    if np.linalg.norm(np.dot(Sigma**0.5, D).dot(Gamma**0.5), 2) > 1:
        print (np.linalg.norm(np.dot(Sigma**0.5, D).dot(Gamma**0.5), 2))
        raise Exception('higher than 1')

    new_w = np.array([np.zeros(n) for i in range(N)])
    prev_w = np.array([np.zeros(n) for i in range(N)])
    new_u = np.array([np.zeros(n) for i in range(E)])

    our_scores = []
    for iterk in range(K):
        # print ('iter:', iterk)
        prev_w = np.copy(new_w)

        hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))  # could  be negative

        for i in range(N):
            if i in samplingset:
                mtx1 = 2 * Gamma_vec[i] * np.dot(X[i].T, X[i]).astype('float64')
                if mtx1.shape:
                    mtx1 += 1 * np.eye(mtx1.shape[0])
                    mtx_inv = np.linalg.inv(mtx1)
                else:
                    mtx1 += 1
                    mtx_inv = 1.0 / mtx1

                mtx2 = hat_w[i] + 2 * Gamma_vec[i] * np.dot(X[i].T, Y[i]).T[0]

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

    if np.max(abs(new_w - prev_w)) > 5 * 1e-3:
        raise Exception('not converged')
    # print ('our mean_squared_log_error', our_scores[-1])

    x = np.sum(X, 1)
    y = np.sum(Y, 1)
    model = LinearRegression().fit(x[samplingset], y[samplingset])

    linear_regression_score = score_func(y, model.predict(x))
    # print ('linear_regression mean_squared_log_error', linear_regression_score)

    # coef = model.coef_.reshape(2,1)
    # mse = 0
    # for i in range(N):
    #     # print (Y[i], np.dot(X[i], coef), np.linalg.norm(Y[i] - np.dot(X[i], coef)) / np.linalg.norm(Y[i]))
    #     mse += (np.linalg.norm(Y[i] - np.dot(X[i], coef)) / np.linalg.norm(Y[i]))
    # mse /= N
    #
    y = Y.reshape(-1, 1)
    x = X.reshape(-1, 2)
    decision_tree_samplingset = []
    for item in samplingset:
        for i in range(5):
            decision_tree_samplingset.append(5*item+i)
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
