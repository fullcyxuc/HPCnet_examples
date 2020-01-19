import torch
from typing import Tuple
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import math
from HausdorffTest.ReadShapes import read_keyfile
from HausdorffTest.ReadShapes import LoadGivenShapes
import time
import os
import sys
import torch.multiprocessing as mp

import numpy as np
import HPCnet_cuda as HPCnet

class getGtFeature(Function):
    @staticmethod
    def forward(ctx, whole_points: torch.Tensor, keypoints: torch.Tensor, \
                neighbor_points: torch.Tensor, radius: float, neighbor_point_num: float)\
                -> torch.Tensor:
        """
        whole_points: B C N
        keypoints: B N C
        neighbor_points: B N nsample C
        output: feature: B M gt_num
        """
        # print(whole_points[:,:,:].size())
        root = "/home/xue/Documents/Pirors/" + str(radius)
        prior_points, dis_dicts = LoadGivenShapes(root)

        dis_dicts = torch.cuda.FloatTensor(dis_dicts)
        prior_points = torch.cuda.FloatTensor(prior_points)
        # gt_feature_len = len(dis_dicts)

        voxel_dim = 30
        voxel_len = 2*radius / voxel_dim
        voxel_dim = int(2*radius/voxel_len + 1)

        batch_size, keypoint_num, point_dim= keypoints.size()
        whole_point_num = whole_points.size()[0]

        feature = torch.cuda.FloatTensor(batch_size, keypoint_num, len(prior_points)).zero_()

        HPCnet.get_hausdorff_dis_wrapper(neighbor_points, feature, radius,\
                                            batch_size, \
                                            whole_point_num, keypoint_num, neighbor_point_num, \
                                            prior_points, dis_dicts,\
                                            voxel_len)

        return feature

    # @staticmethod
    # def backward(feature, a = None):
    #     return None, None, None, None, None

get_gt_feature = getGtFeature.apply
