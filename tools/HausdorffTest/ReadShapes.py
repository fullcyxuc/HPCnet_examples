import os
import os.path
import sys
import torch

"""
Created on Wed Dec 11 22:53:52 2019

@author: pengdi huang
@email: alualu628628@163.com

a class to implement fast hausdorff distance calculation
this method is based on the situation that oneside of shape is known
"""


def get_shapelists(root):
    """
    :param root: a path/folder of shape files
    :return: a dictionary of files' names
    """
    catfile = os.path.join(root, 'ShapeLists.txt')
    filetitles = {}

    with open(catfile, 'r') as f:
        i = 0;
        for line in f:
            filetitles[line.strip('\n')] = i
            i = i + 1

    return filetitles


def read_keyfile(filename):
    """
    :param filename: a full path of target file i.e., /path/name.xxx
    :return: a turple of class PointXYZ, which is a point clouds
    """

    keyshape = []
    #open one key file
    with open(filename, "r") as f:

        for line in f.readlines():
            #read each line of file
            onepointdata = line.strip('\n').split(' ')
            #get the point value
            #onepoint = PointXYZ(float(onepointdata[0]),float(onepointdata[1]),float(onepointdata[2]))
            onepoint = [float(onepointdata[0]),float(onepointdata[1]),float(onepointdata[2])]
            #construct a point clouds
            keyshape.append(onepoint)
    #return a point clouds of given shape
    return keyshape

def read_disdictionary(filename):
    """
    :param filename: a full path of target file i.e., /path/name.xxx
    :return: a turple of class PointXYZ, which is a point clouds
    """
    disdict = []
    #open one key file
    with open(filename, "r") as f:

        for line in f.readlines():
            #read each line of file
            onedisvalue = line.strip('\n')
            #construct a point clouds
            disdict.append(float(onedisvalue))

    return disdict



def LoadGivenShapes(root):
    """
    :param root: a path of files relative to the given shapes
    :return:keyshapes - a lots of point clouds, a point clouds is a tuple of poiclass PointXYZ
            shapedicts - the dictionaries denote the distance from space to given shapes, each dictionary is a tuple of distance value
    """

    shapetitles = get_shapelists(root)
    #key points of shapes
    keyshapes = []
    #dictionaries of shapes
    shapedicts = []
    #sort dictionary
    #order is very important, because it determines the feature vector
    for onetitle in sorted(shapetitles, key = shapetitles.__getitem__):
        #input data from key files
        keyfilename = os.path.join(root, (onetitle + '.key'))
        onekeyshape = read_keyfile(keyfilename)
        keyshapes.append(onekeyshape)

        #input data from files
        dictname = os.path.join(root, (onetitle + '.dic'))
        onedict = read_disdictionary(dictname)
        shapedicts.append(onedict);

    return keyshapes, shapedicts
