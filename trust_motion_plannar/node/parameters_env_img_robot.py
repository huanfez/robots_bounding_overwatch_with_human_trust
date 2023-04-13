#! /usr/bin/env python

import rospy, rospkg
from turtlesim.msg import Pose

from PIL import Image, ImageOps
import numpy as np
import copy
import tifffile as tf

rospack = rospkg.RosPack()
# list all packages, equivalent to rospack list
rospack.list()
# get the file path for rospy_tutorials
trust_pkg_dir = rospack.get_path('trust_motion_plannar')

#####################################################################################
# Image information: DEM DSM, traversability and visibility
dem_file = trust_pkg_dir + '/node/map/yazoo_500m_dem.tif'
dem_img = tf.imread(dem_file)
# dem_500m_arr = np.asarray(dem_500m)
dsm_file = trust_pkg_dir + '/node/map/yazoo_500m_dsm.tif'
dsm_img = tf.imread(dsm_file)
# dsm_500m_arr = np.asarray(dsm_500m)

# dsm_dem_diff_500m_arr = dsm_500m_arr - dem_500m_arr
# tf.imwrite('src/trust_motion_plannar/node/map/Minus_yazoo_500m.tif', dsm_dem_diff_500m_arr)

# sx, sy = np.gradient(dem_500m_arr, 10)
# slope = np.sqrt(sx**2 + sy**2)
# tf.imwrite('src/trust_motion_plannar/node/map/slope_yazoo_500m.tif', slope)

traversability_file = trust_pkg_dir + '/node/map/slope_yazoo_500m.tif'
traversability_img = tf.imread(traversability_file)
visibility_file = trust_pkg_dir + '/node/map/Minus_yazoo_500m.tif'
visibility_img = tf.imread(visibility_file)

gis_traversability = np.copy(np.asarray(traversability_img))
gis_visibility = np.copy(np.asarray(visibility_img))

r1_dynamic_traversability = np.divide((1 - np.exp(gis_traversability * 100.0 - 2.2)),
                                      (1 + np.exp(gis_traversability * 100.0 - 2.2)))
r1_dynamic_visibility = np.divide((1 - np.exp(gis_visibility - 1.5)),
                                  (1 + np.exp(gis_visibility - 1.5)))

r2_dynamic_traversability = np.divide((1 - np.exp(gis_traversability * 100.0 - 2.2)),
                                      (1 + np.exp(gis_traversability * 100.0 - 2.2)))
r2_dynamic_visibility = np.divide((1 - np.exp(gis_visibility - 1.5)),
                                  (1 + np.exp(gis_visibility - 1.5)))

r3_dynamic_traversability = np.divide((1 - np.exp(gis_traversability * 100.0 - 2.2)),
                                      (1 + np.exp(gis_traversability * 100.0 - 2.2)))
r3_dynamic_visibility = np.divide((1 - np.exp(gis_visibility - 1.5)),
                                  (1 + np.exp(gis_visibility - 1.5)))

#####################################################################################
# define grid environment parameters
env_width, env_height = 500.0, 500.0  # meters
cell_width, cell_height = 25, 25  # pixels
img_width, img_height = traversability_img.shape  # pixels
grid_width, grid_height = int(img_width / cell_width), int(img_height / cell_height)  # discrete environment size

'''Candidate paths'''
candidate_cell_path0 = [(7, 13), (6, 13), (5, 13), (4, 13), (3, 13), (2, 13), (2, 12), (2, 11), (2, 10),
                        (2, 9)]
# candidate_cell_path0 = [(8, 13), (7, 13), (6, 13), (5, 13), (4, 13), (4, 12), (4, 11), (3, 11), (3, 10),
#                         (3, 9), (4, 9), (3, 9), (2, 9)]
candidate_cell_path1 = [(7, 13), (7, 12), (7, 11), (7, 10), (7, 9), (6, 9), (5, 9), (4, 9), (3, 9), (2, 9)]
candidate_cell_path2 = [(7, 13), (7, 12), (7, 11), (6, 11), (5, 11), (5, 10), (5, 9), (4, 9), (3, 9), (2, 9)]
candidate_cell_path3 = [(7, 13), (6, 13), (6, 14), (6, 15), (5, 15), (4, 15), (3, 15), (3, 14), (3, 13),
                        (2, 13), (2, 12), (2, 11), (2, 10), (2, 9)]
# candidate_cell_path3 = [(8, 13), (9, 13), (9, 12), (9, 11), (8, 11), (7, 11), (6, 11), (5, 11), (4, 11), (3, 11),
# (3, 10), (3, 9), (2, 9)]
candidate_cell_path4 = [(7, 13), (8, 13), (9, 13), (9, 12), (9, 11), (8, 11), (7, 11), (7, 10), (7, 9), (6, 9), (5, 9),
                        (4, 9),
                        (3, 9), (2, 9)]

candidate_cell_path_list = [candidate_cell_path0,  candidate_cell_path1, candidate_cell_path2, candidate_cell_path4,
                            candidate_cell_path3]

cp0 = copy.deepcopy(candidate_cell_path0)
cp1 = copy.deepcopy(candidate_cell_path1)
cp2 = copy.deepcopy(candidate_cell_path2)
cp4 = copy.deepcopy(candidate_cell_path4)
candidate_cell_path_list_reverse = [list(reversed(cp0)), list(reversed(cp1)), list(reversed(cp2)), list(reversed(cp4))]

######################################################################################
husky_alpha_init = Pose(-60.0, -88.0, 0.0, 0.0, 0.0)

trust_temp_tif = trust_pkg_dir + '/node/data/trust_temp.tif'
traversability_r1 = trust_pkg_dir + '/node/data/traversability_r1.tif'
traversability_r2 = trust_pkg_dir + '/node/data/traversability_r2.tif'
traversability_r3 = trust_pkg_dir + '/node/data/traversability_r3.tif'
visibility_r1 = trust_pkg_dir + '/node/data/visibility_r1.tif'
visibility_r2 = trust_pkg_dir + '/node/data/visibility_r2.tif'
visibility_r3 = trust_pkg_dir + '/node/data/visibility_r3.tif'


# def create_trust_map(height=grid_height, width=grid_width):
#     trust_dict_ = {}
#     for cy in range(0, height):
#         for cx in range(0, width):
#             trust_dict_[(cx, cy)] = np.array([0.0, 0.0, 0.0])
#     return trust_dict_

######################################################################################
# Common used functions:
def imgPos2envPos(imgPos, img_width=img_width, img_height=img_height, env_width=env_width, env_height=env_height):
    transMat = np.array([[img_width / 2.0], [img_height / 2.0]])
    scalMat = np.array([[env_width / img_width, 0], [0, -env_height / img_height]])

    array1 = imgPos - transMat
    envPos = np.dot(scalMat, array1)
    return envPos


def envPos2imgPos(envPos, img_width=img_width, img_height=img_height, env_width=env_width, env_height=env_height):
    transMat = np.array([[img_width / 2.0], [img_height / 2.0]])
    scalMat = np.array([[env_width / img_width, 0], [0, -env_height / img_height]])

    array1 = np.dot(np.linalg.inv(scalMat), np.array([envPos]).T)
    imgPos = array1 + transMat
    return imgPos.flatten().astype(int)


def imgpath2traj(localPath):
    localTraj = []
    for imgPos in localPath:
        imgPosArray = np.array([[imgPos[0]], [imgPos[1]]])
        localTraj.append(imgPos2envPos(imgPosArray))
    return localTraj
