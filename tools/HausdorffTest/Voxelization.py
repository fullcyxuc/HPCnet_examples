#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:53:52 2019

@author: pengdi huang
@email: alualu628628@163.com

a class to implement fast hausdorff distance calculation
this method is based on the situation that oneside of shape is known
"""
import math


class PointXYZ:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z


class Voxel:

    #construction
    def __init__(self, fSpaceR, fVoxelLength):
        self.m_fNeighR = fSpaceR
        self.m_fVoxelLength = fVoxelLength

    ##########end Function __init__

    #i in xnum, j in ynum and k in znum
    def Transfor3DTo1DIdx(self, i, j, k, xnum, ynum):
        return i + j*xnum + k*xnum*ynum

    ##########end Function Transfor3DTo1DIdx

    def SpaceVoxelization(self, vOutCloud):
        """
        Function:SpaceVoxelization
        Parameters:
        points:std::vector<Point3D> & fomually point clouds data set
        m_fVoxelLength: length of each voxel set by Setm_fVoxelLength function
        Output: voxelization of point clouds hehe
        """
        #set the bounding box corner points
        self.m_oMaxCor = PointXYZ(self.m_fNeighR,self.m_fNeighR,self.m_fNeighR)
        fMincorR = -1.0*self.m_fNeighR
        self.m_oMinCor = PointXYZ(fMincorR,fMincorR,fMincorR)


        #compute the size number
        #a example:m_fVoxelLength = 1.0
        #1 2 3 4 r=4.2 5
        #---floor---- + 1
        self.m_xnum = math.floor(abs(self.m_oMaxCor.x - self.m_oMinCor.x) / self.m_fVoxelLength) + 1
        self.m_ynum = math.floor(abs(self.m_oMaxCor.y - self.m_oMinCor.y) / self.m_fVoxelLength) + 1
        self.m_znum = math.floor(abs(self.m_oMaxCor.z - self.m_oMinCor.z) / self.m_fVoxelLength) + 1

        #voxelization and save the corresponding center point in each voxel
        vOutCloud.clear()
        vOutCloud.points.resize(self.m_xnum*self.m_ynum*self.m_znum)

        #voxelization
        for k in range(len(self.m_znum)):

            for j in range(len(self.m_ynum)):

                for i in range(len(self.m_xnum)):

                    #compute the center of each voxel
                    iOneIdx = self.Transfor3DTo1DIdx(i, j, k, self.m_xnum, self.m_ynum, self.m_znum)
                    vOutCloud[iOneIdx][0] = self.m_oMinCor.x + (float(i) + 0.5) * self.m_fVoxelLength
                    vOutCloud[iOneIdx][1] = self.m_oMinCor.y + (float(j) + 0.5) * self.m_fVoxelLength
                    vOutCloud[iOneIdx][2] = self.m_oMinCor.z + (float(k) + 0.5) * self.m_fVoxelLength

    ##########end Function Reloaded SpaceVoxelization

    #reload without any output
    def SpaceVoxelization(self):

        #set the bounding box corner points
        self.m_oMaxCor = PointXYZ(self.m_fNeighR,self.m_fNeighR,self.m_fNeighR)
        fMincorR = -1.0*self.m_fNeighR
        self.m_oMinCor = PointXYZ(fMincorR,fMincorR,fMincorR)

        #compute the size number
        #a example:m_fVoxelLength = 1.0
        #1 2 3 4 r=4.2 5
        #---floor---- + 1
        self.m_xnum = math.floor(abs(self.m_oMaxCor.x - self.m_oMinCor.x) / self.m_fVoxelLength) + 1
        self.m_ynum = math.floor(abs(self.m_oMaxCor.y - self.m_oMinCor.y) / self.m_fVoxelLength) + 1
        self.m_znum = math.floor(abs(self.m_oMaxCor.z - self.m_oMinCor.z) / self.m_fVoxelLength) + 1

    ##########end Function Reloaded SpaceVoxelization

    def LocationTo1DIdx(self, oQueryPoint):

        ith = math.floor(abs(oQueryPoint[0] - self.m_oMinCor.x) / self.m_fVoxelLength)
        jth = math.floor(abs(oQueryPoint[1] - self.m_oMinCor.y) / self.m_fVoxelLength)
        kth = math.floor(abs(oQueryPoint[2] - self.m_oMinCor.z) / self.m_fVoxelLength)

        idx = self.Transfor3DTo1DIdx(ith, jth, kth, self.m_xnum, self.m_ynum)

        if idx == 41591:
            print(ith, jth, kth)
            print(oQueryPoint[0],oQueryPoint[1],oQueryPoint[2])

        return idx
    ##########end Function LocationTo1DIdx
