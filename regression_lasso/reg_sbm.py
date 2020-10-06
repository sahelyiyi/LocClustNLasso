import numpy as np
from stochastic_block_model import get_B_and_weight_vec
from regression_lasso.main import *


def run_reg_sbm(K, lambda_lasso, m, n, N1=150, N2=150, M=0.2):
    B, weight_vec = get_B_and_weight_vec(N1, N2, mu_in=40, mu_out=10)
    E, N = B.shape

    X = []
    for i in range(N):
        x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, m))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 1.0, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        X.append(g)
    X = np.array(X)

    W1 = np.random.random(n)
    W2 = np.random.normal(5, 2, n)
    print (W1, W2)

    Y = []
    for i in range(N):
        if i < N1:
            Y.append(np.dot(X[i], W1))
        else:
            Y.append(np.dot(X[i], W2))
    Y = np.array(Y)

    return run(K, B, weight_vec, Y, X, lambda_lasso, method='norm', M=M)

# run_reg_sbm(K=2000, lambda_lasso=4, m=5, n=15) -> (0.08847329547034469, 0.3923577328240916, 0.44523171610089396)
# run_reg_sbm(K=1000, lambda_lasso=4, m=5, n=15) -> (0.09641694800821025, 0.35112997213736924, 0.4608293831740298)
# run_reg_sbm(K=500, lambda_lasso=4, m=5, n=15)  -> (0.08210340592268237, 0.42716694751090095, 0.48258665883442287)


# run_reg_sbm(K=1000, lambda_lasso=0.1, m=5, n=2) -> (0.002244112423726346, 0.40053943245846946, 0.44088712613569986)
# run_reg_sbm(K=500, lambda_lasso=0.1, m=5, n=2)  -> (0.0054678377124780814, 0.3020360482292944, 0.3521123091303605)
# run_reg_sbm(K=300, lambda_lasso=0.1, m=5, n=2)  -> (0.015739280872195678, 0.28437449803685105, 0.33406791900522087)

# run_reg_sbm(K=500, lambda_lasso=0.5, m=5, n=2)  -> (0.029084207057331054, 0.49156914634684995, 0.5404649888908892)
