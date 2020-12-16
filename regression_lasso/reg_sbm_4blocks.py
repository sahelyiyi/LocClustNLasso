import numpy as np
from stochastic_block_model import get_B_and_weight_vec
from regression_lasso.main import *


def run_reg_sbm(K, lambda_lasso, m=5, n=2, M=0.2):
    block_sizes = [70, 10, 50, 100, 150]
    blocks_num = len(block_sizes)
    B, weight_vec = get_B_and_weight_vec(block_sizes, pout=0.001, mu_in=40, mu_out=10)
    E, N = B.shape

    X = []
    for i in range(N):
        sigma, mu = 1.0, 0.0
        g = np.random.normal(mu, sigma, (m, n))
        X.append(g)
    X = np.array(X)

    block_Ws = []
    for i in range(blocks_num):
        block_Ws.append(np.random.random(n))

    Y = []
    W = []
    cnt = 0
    for j in range(blocks_num):
        for i in range(block_sizes[j]):
            block_W = block_Ws[j]
            Y.append(np.dot(X[cnt], block_W))
            W.append(block_W)
            cnt += 1

    Y = np.array(Y)
    W = np.array(W)

    samplingset = random.sample([i for i in range(N)], k=int(M * N))
    return run(K, B, weight_vec, Y, X, W, samplingset, lambda_lasso, method='norm')


# colors = ['steelblue', 'darkorange', 'green', 'firebrick', 'mediumpurple']
#
# for i, lambda_lasso in enumerate(ours):
#     if i >= len(colors):
#         continue
#     plt.plot(iteration_numbers, iterations[lambda_lasso], label='lambda=%s' % lambda_lasso, color=colors[i])
#
# plt.xlabel("Iteration")
# plt.ylabel('MSE')
# plt.title('weight vector MSE over iterations')
# plt.legend(loc="upper right")
# plt.savefig('4_compare_mse.png')

