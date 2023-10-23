import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 6
self_link = [(i, i) for i in range(num_node)] #单位阵 list
inward_ori_index = [(1, 2), (2, 3), (2, 4), (2, 5), (2, 6)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index] #下标从零开始
outward = [(j, i) for (i, j) in inward] #对称矩阵
neighbor = inward + outward #邻接矩阵

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
