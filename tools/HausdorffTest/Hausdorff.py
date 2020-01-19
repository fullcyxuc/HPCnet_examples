#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:53:52 2019

@author: pengdi huang
@email: alualu628628@163.com

a class to implement fast hausdorff distance calculation
this method is based on the situation that oneside of shape is known
"""

import HausdorffTest.Voxelization as Voxels
from scipy import spatial
import math
import sys
from pointnet2.hpc_utils import get_neighbors_r, get_max_dis

class Hausdorff:

    def __init__(self,f_fSpaceR, f_fVoxelLength):

        self.m_oVoxel = Voxels.Voxel(f_fSpaceR, f_fVoxelLength)
        self.m_oVoxel.SpaceVoxelization()

    ###end Function __init__

    #Euclidean distance between two points
    def EuclideanDis(self,fx1, fy1, fz1, fx2, fy2, fz2):

        fxValue = (fx1 - fx2)*(fx1 - fx2)
        fyValue = (fy1 - fy2)*(fy1 - fy2)
        fzValue = (fz1 - fz2)*(fz1 - fz2)
        #sqrt
        return math.sqrt(fxValue + fyValue + fzValue)
    ###end Function EuclideanDis

    def RelativeCor(self, vCloud, neighidxs, oQueryPoint):

        vReNearCloud = [[0]*3 for row in range(len(neighidxs))]

        for i in range(len(neighidxs)):
            vReNearCloud[i][0] = vCloud[neighidxs[i]][0] - oQueryPoint[0]
            vReNearCloud[i][1] = vCloud[neighidxs[i]][1] - oQueryPoint[1]
            vReNearCloud[i][2] = vCloud[neighidxs[i]][2] - oQueryPoint[2]

        return vReNearCloud

    ###end RelativeCor EuclideanDis

    def LocationtoDis(self, oQueryPoint, vDisDict):

        #get index (int) and then the corresponding distance
        iOneIdx = self.m_oVoxel.LocationTo1DIdx(oQueryPoint)

        return vDisDict[iOneIdx]

    ##########end Function LocationtoDis

    #Hausdorff distance from target points to source points
    def HausdorffDisMax_cuda(self,vCloud, points_neighbor):
        return get_max_dis(vCloud, points_neighbor)

    def HausdorffDisMax(self,vCloud, oKDTree):

        #set the max of anything
        fTotalMax = -1.0*sys.float_info.max

        #search the nearest contour point
        for j in range(len(vCloud)):

            #compute distance of two point
            oneneardis, nearestidx = oKDTree.query(vCloud[j],1)

            if fTotalMax < oneneardis:
                fTotalMax = oneneardis

        return fTotalMax
    # end Function HausdorffDisMax

    #Hausdorff distance from target points to source points
    def HausdorffDisMean(self, vCloud, oKDTree):

        #set the zero at the begining of computing
        fTotalMean = 0.0;

        #search the nearest contour point
        for j in range(len(vCloud)):

            #compute distance of two point
            oneneardis, nearestidx = oKDTree.query(vCloud[j],1)

            fTotalMean = fTotalMean + oneneardis

        return fTotalMean / len(vCloud)
    ##########end Function HausdorffDisMean


    #compute the minimum distance based on a dictionary
    def HausdorffDict(self, vCloud, vDisDict):
        # vCloud = vCloud.numpy().tolist()

        #set the max of anything
        fTotalMax = -1.0*sys.float_info.max

        #search the nearest contour point
        for j in range(len(vCloud)):

            #find the voxel index of query point
            fOneDis = self.LocationtoDis(vCloud[j],vDisDict)

            if fTotalMax < fOneDis:
                fTotalMax = fOneDis


        return fTotalMax
    ##########end Function HausdorffDict


    def GeneralHausDis(self,fToTempDis, fToSourceDis):

        #find the maximum value
        gendis = fToTempDis if fToTempDis > fToSourceDis else fToSourceDis
        #normolization
        gendis = gendis/self.m_oVoxel.m_fNeighR
        #in same case of the measured scale of dictionary, accuarcy is low that the distance is larger than 1
        if gendis > 1.0:
            gendis = 1.0

        return gendis

    ##########end Function GeneralHausDis


##########end Function LocationtoDis
