#! /usr/bin/env python2

from PIL import Image, ImageOps
from scipy import ndimage
import numpy as np
import sys, csv, time
import parameters_env_img_robot as common_parameters

# For storing the vertex set to retrieve node with the lowest distance
class Graph:
    def __init__(self, num):
        self.adjList = {}  # To store graph: u -> (v,w)
        self.num_nodes = num  # Number of nodes in graph
        # To store the distance from source vertex
        self.dist = {}
        self.par = {}  # To store the path
        self.trust_dict = {}  # To store the trust and variance

    def add_edge(self, u, v, w):
        #  Edge going from node u to v and v to u with weight w
        # u (w)-> v, v (w) -> u
        # Check if u already in graph
        if u in self.adjList.keys():
            self.adjList[u].append((v, w))
        else:
            self.adjList[u] = [(v, w)]

        # Assuming undirected graph
        # if v in self.adjList.keys():
        #    self.adjList[v].append((u, w))
        # else:
        #    self.adjList[v] = [(u, w)]

    def show_graph(self):
        # u -> v(w)
        print(sorted(self.adjList.keys()))
        for u in self.adjList:
            print(u, "->", " -> ".join(str("{}({})".format(v, w)) for v, w in self.adjList[u]))

    def dijkstra2(self, src, cell_dict, beta_m, beta_v, tau0=[0.0, 0.001]):
        self.trust_dict = {}
        # Flush old junk values in par[]
        self.par = {}
        # src is the source node
        trust1_mean, trust1_var = trustEval2(tau0, src, cell_dict, beta_m, beta_v)
        normalized_distrust1_mean = 1.0 / (1.0 + np.exp(np.longdouble(trust1_mean)))
        self.trust_dict[src] = [normalized_distrust1_mean, trust1_mean]
        self.dist[src] = self.trust_dict[src][0]

        Q = {}

        """ initialize the value"""
        Q[src] = self.trust_dict[src][0]  # (trust from src node)
        self.par[src] = -1
        for u in self.adjList.keys():
            if u != src:
                self.dist[u] = sys.maxsize  # Infinity
                Q[u] = sys.maxsize
                self.par[u] = -1

        while bool(Q):
            q_min = min(Q.values())
            us = [key for key in Q if Q[key] == q_min]
            u = us[0]
            del Q[u]
            # Update the distance of all the neighbours of u and
            # if their prev dist was INFINITY then push them in Q
            for v, w in self.adjList[u]:
                if v not in Q.keys():
                    continue
                """ update the cost value """
                # print "Trust dict:", self.trust_dict
                trust_true = [np.log(1.0 / self.trust_dict[u][0] - 1.0), self.trust_dict[u][1]]
                trust_mean, trust_var = trustEval2(trust_true, v, cell_dict, beta_m, beta_v)
                normalized_distrust_mean = 1.0 / (1.0 + np.exp(np.longdouble(trust_mean)))
                self.trust_dict[v] = [np.copy(normalized_distrust_mean), np.copy(trust_mean)]

                new_dist = normalized_distrust_mean
                if self.dist[v] > new_dist:
                    Q[v] = new_dist
                    self.dist[v] = new_dist
                    self.par[v] = u

        # Show the shortest distances from src
        # self.show_distances(src)

    def show_distances(self, src):
        print("Distance from node: {}".format(src))
        for u in sorted(self.adjList.keys()):
            print("Node {} has distance: {}".format(u, self.dist[u]))

    def show_path(self, src, dest):
        # To show the shortest path from src to dest
        # WARNING: Use it *after* calling dijkstra
        path = []
        cost = 0
        temp = dest
        # Backtracking from dest to src
        while self.par[temp] != -1:
            path.append(temp)
            if temp != src:
                for v, w in self.adjList[temp]:
                    if v == self.par[temp]:
                        cost += w
                        break
            temp = self.par[temp]
        path.append(src)
        path.reverse()

        print("----Path to reach {} from {}----".format(dest, src))
        for u in path:
            print("{}".format(u), "(----Path from {} to {} ----)".format(src, u) + str(np.log(1.0 / self.trust_dict[u][0] - 1.0)) + " " + str(self.trust_dict[u][0])+"\n")
            # f.write("----Path from {} to {} ----".format(src, u) + str(np.log(1.0 / self.trust_dict[u][0] - 1.0)) + " " + str(self.trust_dict[u][0]) + "\n")
        # print("\nTotal cost of path: ", cost)
        # f.write("/////////\n")
        return path

    def trust_map(self, p, img_size, cell_width, cell_height, beta_means):
        trust_img = Image.new("RGBA", img_size, (0, 0, 0, 0))
        new_ = trust_img.load()
        min_trust = max(self.trust_dict.values())[1]
        max_trust = min(self.trust_dict.values())[1]

        diff_trust = 255.0 / (max_trust - min_trust)
        for node in self.trust_dict.keys():
            for dot_y in range(0, cell_width):
                for dot_x in range(0, cell_height):
                    new_[node[1] * cell_width + dot_y, node[0] * cell_height + dot_x] = \
                        (diff_trust * (self.trust_dict[node][1] - min_trust), diff_trust * (self.trust_dict[node][1] - min_trust), diff_trust * (self.trust_dict[node][1] - min_trust), 255)

        trust_img.save('./map3/paths{0}/trust_map_{1}_{2}{3}.bmp'.format(beta_means[0], p, beta_means[1], beta_means[2]))


def trustEval2(tau, node, cell_dict, beta_m, beta_v):
    feature = cell_dict[node]

    z_mean = [tau[0], feature[0], feature[1], 1.0]
    z_var = [[tau[1], 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    trust_mean = np.dot(beta_m, z_mean)

    var_x0y0 = beta_v[0][0] * ((z_mean[0]) ** 2) + z_var[0][0] * ((beta_m[0]) ** 2) + beta_v[0][0] * z_var[0][
        0]
    var_x1y1 = beta_v[1][1] * ((z_mean[1]) ** 2) + z_var[1][1] * ((beta_m[1]) ** 2) + beta_v[1][1] * z_var[1][
        1]
    var_x2y2 = beta_v[2][2] * ((z_mean[2]) ** 2) + z_var[2][2] * ((beta_m[2]) ** 2) + beta_v[2][2] * z_var[2][
        2]

    cov_x0y0_x1y1 = beta_v[0][1] * z_mean[0] * z_mean[1]
    cov_x1y1_x2y2 = beta_v[1][2] * z_mean[1] * z_mean[2]
    cov_x0y0_x2y2 = beta_v[0][2] * z_mean[0] * z_mean[2]

    trust_var = var_x0y0 + var_x1y1 + var_x2y2 + (cov_x0y0_x1y1 + cov_x1y1_x2y2 + cov_x0y0_x2y2) * 2.0

    # print("trustEval:", node, dist, np.mean(tau1_array), np.var(tau1_array))

    return trust_mean, trust_var


# discrete cell info
def gen_env(grid_height, grid_width, cell_height, cell_width, norm_traversability, norm_line_of_sight, image):
    image_height_mean = ndimage.uniform_filter(image, (cell_height, cell_width))
    cell_dict = {}
    obs_list = [(14,0), (11,3), (0,7), (14,7), (2,8), (3,8), (4,8), (5,8), (7,8), (8,8), (17,8),
                (1,9), (4,10), (6,10), (7,10), (9,10), (12,11), (1,12), (3,12), (5,12),
                (6,12), (8,12), (10,12), (14,12), (16,12), (0,13), (12,13), (0,14),
                (4,14), (5,14), (10,14), (0,15), (2,15), (0,16), (6,16), (7,16), (0,17),
                (3,17), (4,17), (17,17), (18,18), (12,19), (13,19), (14,19), (18,19)]
    for cy in range(0, grid_height):
        for cx in range(0, grid_width):
            index_x, index_y = cx * cell_width + int(cell_width / 2), cy * cell_height + int(cell_height / 2)
            # print(index_x, index_y)
            # if 2.5 > image_height_mean[index_y][index_x] > 1.5:
            if (cx, cy) in obs_list:
                # obs_list.append((cx, cy))
                continue
            cell_dict[(cx, cy)] = norm_traversability[index_y][index_x], norm_line_of_sight[index_y][index_x]
    print "cells are:", cell_dict, "Obstacle list,", obs_list
    return cell_dict, obs_list


def gen_graph(cell_dict, obs_list, env_height, env_width):
    delta = -9.5
    cells = cell_dict.keys()
    graph = Graph(len(cell_dict))
    for cell in cells:
        if cell in obs_list:
            continue

        cell1 = cell[0] - 1, cell[1]
        cell2 = cell[0] + 1, cell[1]
        cell3 = cell[0], cell[1] - 1
        cell4 = cell[0], cell[1] + 1

        if 0 <= cell1[0] < env_height and 0 <= cell1[1] < env_width and cell1 in cells:
            if cell_dict[cell1][0] > delta:
                graph.add_edge(cell, cell1, 0)

        if 0 <= cell2[0] < env_height and 0 <= cell2[1] < env_width and cell2 in cells:
            if cell_dict[cell2][0] > delta:
                graph.add_edge(cell, cell2, 0)

        if 0 <= cell3[0] < env_height and 0 <= cell3[1] < env_width and cell3 in cells:
            if cell_dict[cell3][0] > delta:
                graph.add_edge(cell, cell3, 0)

        if 0 <= cell4[0] < env_height and 0 <= cell4[1] < env_width and cell4 in cells:
            if cell_dict[cell4][0] > delta:
                graph.add_edge(cell, cell4, 0)
    return graph


# generate cell path for robots
def gen_discrete_path():
    time.sleep(2)
    # read image
    normalized_traversability = (0.15 - np.asarray(common_parameters.traversability_img)) * 10.0
    normalized_traversability = ndimage.uniform_filter(normalized_traversability,
                                (common_parameters.cell_height, common_parameters.cell_width))

    normalized_visibility = (1.50 - np.asarray(common_parameters.visibility_img)) / 10.0
    normalized_visibility = ndimage.uniform_filter(normalized_visibility,
                            (common_parameters.cell_height, common_parameters.cell_width))

    obs_img = np.asarray(common_parameters.visibility_img)

    env_dict, obs_list = gen_env(common_parameters.grid_height, common_parameters.grid_width,
                                 common_parameters.cell_height, common_parameters.cell_width,
                                 normalized_traversability, normalized_visibility, obs_img)

    ''' estimate the trust value through sampling '''
    # beta_means = [0.9, 0.02, 0.01]
    # beta_vars = [[0.01, 0, 0], [0, 0.0001, 0.0001], [0, 0, 0.0001]]
    beta_means = [0.990, 0.0, 0.10, -0.75]
    beta_vars = [[0.01, 0, 0, 0], [0, 0.01, 0.001, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.01]]

    # target positions
    p1 = (8, 13)
    p2 = (2, 9)

    # path1 2
    graph1 = gen_graph(env_dict, obs_list, common_parameters.grid_height, common_parameters.grid_width)
    graph1.show_graph()
    graph1.dijkstra2(p1, env_dict, beta_means, beta_vars)
    # graph1.trust_map(p1, traversability_img.size, cell_width, cell_height, beta_means)
    discrete_path_1to2 = graph1.show_path(p1, p2)

    path_array = np.array(discrete_path_1to2)
    discrete_path_x, discrete_path_y = path_array[:, 0] * common_parameters.cell_width + common_parameters.cell_width / 2.0, \
                                       path_array[:, 1] * common_parameters.cell_height + common_parameters.cell_height / 2.0
    discrete_path = np.array([discrete_path_x, discrete_path_y]).T
    env_discrete_path = common_parameters.imgpath2traj(discrete_path)
    print env_discrete_path, path_array
    return env_discrete_path, path_array
