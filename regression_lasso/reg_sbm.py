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

# run_reg_sbm(K=2000, lambda_lasso=0.4, m=5, n=15) -> (0.07543958130185237, 0.4327561139685135, 0.470397364281547)
# run_reg_sbm(K=1000, lambda_lasso=0.4, m=5, n=15) -> (0.09223178656091131, 0.4220348484968601, 0.5021987887453047)
# run_reg_sbm(K=500, lambda_lasso=0.4, m=5, n=15)  -> (0.10498491617601496, 0.4088509340133603, 0.4751957719337881)

# run_reg_sbm(K=1000, lambda_lasso=0.1, m=5, n=15) -> (0.08507156724382309, 0.3709281732014657, 0.4482499135199026)
# run_reg_sbm(K=500, lambda_lasso=0.1, m=5, n=15)  -> (0.09318609467039467, 0.40197666378593205, 0.4755215981200701)

# run_reg_sbm(K=500, lambda_lasso=0.07, m=5, n=15)  -> (0.09318609467039467, 0.40197666378593205, 0.4755215981200701)


# run_reg_sbm(K=500, lambda_lasso=0.05, m=5, n=2)  -> (0.047971014813955964, 0.428114030234606, 0.48165557665190534)

# run_reg_sbm(K=1000, lambda_lasso=0.01, m=5, n=2) -> (0.0022115219836184824, 0.4703629298378227, 0.5129913488269128)
# run_reg_sbm(K=500, lambda_lasso=0.01, m=5, n=2)  -> (0.0014434393839232477, 0.39116590556765635, 0.4373967984554611)
# run_reg_sbm(K=300, lambda_lasso=0.01, m=5, n=2)  -> (0.003002048073650676, 0.3836411265938509, 0.43169274260015283)

# run_reg_sbm(K=500, lambda_lasso=0.005, m=5, n=2)  -> (0.00033012642876345126, 0.4301791931028151, 0.47592633684825336)

