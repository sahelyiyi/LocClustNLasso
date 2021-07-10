import logging
import random

import numpy as np


class SBM(object):

    def __init__(self, num_vertices, communities, vertex_labels, pin, pout):
        logging.info('Initializing SBM Model ...')
        self.num_vertices = num_vertices
        self.communities = communities
        self.vertex_labels = vertex_labels
        self.p_matrix = [[pin, pout],
                         [pout, pin]]
        self.block_matrix = self.generate(self.num_vertices, self.communities, self.vertex_labels, self.p_matrix)
        self.B = self.get_B()

    def get_B(self):
        N = self.num_vertices
        # print(len(G[G > 0]) / 2)
        E = int(len(self.block_matrix[self.block_matrix > 0]))
        B = np.zeros((E, N))
        cnt = 0
        for item in np.argwhere(self.block_matrix > 0):
            i, j = item
            if i > j:
                print('noooo1')
            if i == j:
                print ('nooooo')
            B[cnt, i] = 1
            B[cnt, j] = -1
            cnt += 1
        return B

    def detect(self):
        logging.info('SBM detection ...')
        pass

    def generate(self, num_vertices, num_communities, vertex_labels, p_matrix):
        logging.info('Generating SBM (directed graph) ...')
        v_label_shape = (1, num_vertices)
        p_matrix_shape = (num_communities, num_communities)
        block_matrix_shape = (num_vertices, num_vertices)
        block_matrix = np.zeros(block_matrix_shape, dtype=int)

        for row, _row in enumerate(block_matrix):
            for col, _col in enumerate(block_matrix[row]):
                if row == col:
                    continue
                community_a = vertex_labels[row]
                community_b = vertex_labels[col]

                p = random.random()
                val = p_matrix[community_a][community_b]

                if p <= val:
                    if row < col:
                        block_matrix[row][col] = 1
                    else:
                        block_matrix[col][row] = 1

        return block_matrix

    def recover(self):
        logging.info('SBM recovery ...')
        pass
