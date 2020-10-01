import numpy as np
from stochastic_block_model import get_B_and_weight_vec
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import random


def run_reg_sbm(K=300, N1=150, N2=150, W1=[0.5, 0.7], W2=[2, 4]):
    B, weight_vec = get_B_and_weight_vec(N1, N2)

    E, N = B.shape
    M = random.choices([i for i in range(N)], k=int(0.3*N))

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
    # print ('W1:', W1, '  -  W2:', W2)

    Y = []
    for i in range(N):
        if i < N1:
            Y.append(np.dot(X[i], W1))
        else:
            Y.append(np.dot(X[i], W2))
    Y = np.array(Y)
    # Y = np.array([[2] for i in range(N)])
    # X = np.array([[[1, 1, 1]] for i in range(N)])
    m, n = X[0].shape

    weight = np.diag(weight_vec)
    Sigma = np.diag(1./(2*weight_vec))
    Sigma = np.diag(1./(10*weight_vec))

    D = np.dot(weight, B)

    Lambda = 1*np.diag(1./(np.sum(abs(B), 1)))
    Gamma_vec = (.9/(np.sum(abs(B), 0))).T  # \in [0, 1]
    Gamma = np.diag(Gamma_vec)

    lambda_nLasso = 1/3  # nLasso parameter

    hat_w = np.array([np.zeros(n) for i in range(N)])
    new_w = np.array([np.zeros(n) for i in range(N)])
    prev_w = np.array([np.zeros(n) for i in range(N)])
    new_u = np.array([np.zeros(n) for i in range(E)])

    if np.linalg.norm(np.dot(Sigma**0.5, D).dot(Gamma**0.5), 2) > 1:
        print (np.linalg.norm(np.dot(Sigma**0.5, D).dot(Gamma**0.5), 2))
        raise Exception('higher than 1')

    our_scores = []
    for iterk in range(K):
        # print ('iter:', iterk)

        tilde_w = 2 * hat_w - prev_w
        prev_w = np.copy(new_w)
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

                mtx2 = Gamma_vec[i] * hat_w[i] + 1.8 * np.dot(X[i].T, Y[i]).T[0]

                new_w[i] = np.dot(mtx_inv, mtx2)
            else:
                new_w[i] = hat_w[i]

        Y_pred = []
        for i in range(N):
            Y_pred.append(np.dot(X[i], new_w[i]))

        our_scores.append(mean_squared_error(Y, Y_pred))

    if np.max(abs(new_w - prev_w)) > 1e-6:
        raise Exception('not converged')
    # print ('our mean_squared_error', our_scores[-1])

    x = np.sum(X, 1)
    y = np.sum(Y, 1)
    model = LinearRegression().fit(x, y)
    linear_regression_score = mean_squared_error(y, model.predict(x))
    # print ('linear_regression mean_squared_error', linear_regression_score)

    # coef = model.coef_.reshape(2,1)
    # mse = 0
    # for i in range(N):
    #     # print (Y[i], np.dot(X[i], coef), np.linalg.norm(Y[i] - np.dot(X[i], coef)) / np.linalg.norm(Y[i]))
    #     mse += (np.linalg.norm(Y[i] - np.dot(X[i], coef)) / np.linalg.norm(Y[i]))
    # mse /= N
    #
    y = Y.reshape(-1, 1)
    x = X.reshape(-1, 2)
    decision_tree_score = 1e5
    prev_score = 0.0
    for i in range(1, 30):
        regressor = DecisionTreeRegressor(max_depth=i)
        regressor.fit(x, y)
        pred_y = regressor.predict(x)
        score = mean_squared_error(y, pred_y)
        decision_tree_score = min(decision_tree_score, score)
        if score == prev_score:
            break
        prev_score = score
    # print ('decision_tree mean_squared_error', decision_tree_score)

    return our_scores, linear_regression_score, decision_tree_score
