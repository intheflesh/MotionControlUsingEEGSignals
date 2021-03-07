import os
import math
import numpy as np
import random
def loadData(excludedFolders,inPath):
    folders = os.listdir(inPath)
    tmpListOfExclusions = []
    for val in excludedFolders:
        tmpListOfExclusions.append(str(val))
    TrainList = []
    TestList = []
    for folder in folders:
        folderNameForCompare = folder.split("S")[1]
        folderNameForCompare = str(int(folderNameForCompare))
        currentPath = os.path.join(inPath,folder)
        files = os.listdir(currentPath)
        for file in files:
            if file.endswith(".npy"):
                curreFile = os.path.join(currentPath,file)
                label = file.split("_")[1]  # change this for spectral\temporal
                if folderNameForCompare not in tmpListOfExclusions:
                    TrainList.append((curreFile,label))
                else:
                    TestList.append((curreFile,label))
    random.shuffle(TrainList)
    random.shuffle(TestList)
    return TrainList,TestList

def loadDataTemporal(excludedFolders,inPath):
    folders = os.listdir(inPath)
    tmpListOfExclusions = []
    for val in excludedFolders:
        tmpListOfExclusions.append(str(val))
    TrainList = []
    TestList = []
    for folder in folders:
        folderNameForCompare = folder.split("S")[1]
        folderNameForCompare = folderNameForCompare.split("0")[-1]
        currentPath = os.path.join(inPath,folder)
        files = os.listdir(currentPath)
        for file in files:
            if file.endswith(".npy"):
                curreFile = os.path.join(currentPath,file)
                label = file.split("_")[1]  # change this for spectral\temporal
                if folderNameForCompare not in tmpListOfExclusions:
                    TrainList.append((curreFile,label))
                else:
                    TestList.append((curreFile,label))
    random.shuffle(TrainList)
    random.shuffle(TestList)
    return TrainList,TestList

def getMaxSize(inList):
    maxSize = 0
    for line in inList:
        X = np.load(line[0])
        val = X.shape[0]    # change this for spectral\temporal
        if val>maxSize:
            maxSize = val
    return maxSize

def getMaxSizeTemporal(inList):
    maxSize = 0
    for line in inList:
        X = np.load(line[0])
        val = X.shape[1]    # change this for spectral\temporal
        if val>maxSize:
            maxSize = val
    return maxSize

def loadDataFromList(inList,maxSize):
    XArr = []
    YArr = []
    for line in inList:
        X = np.load(line[0])
        X = np.swapaxes(X,1,2)
        #X = X.reshape(X.shape[:-2] + (-1,))
        #X = X[:,0:200,:]
        shapeFirst = X.shape[0]
        X = nomalizedDataSetFreq(X)
        X = np.pad(X,((maxSize-shapeFirst,0),(0,0),(0,0)))
        Y = int(line[1])-6
        XArr.append(X)
        YArr.append(Y)
    #XArr = preprocessX(XArr)
    XArr = np.stack(XArr, axis=0)
    # XData = rangeRegularization(XData)
    #XArr = nomalizedDataSet(XArr)
    YArr = np.asarray(YArr)
    return XArr,YArr


def loadDataFromListTemporal(inList,maxSize):
    XArr = []
    YArr = []
    for line in inList:
        X = np.load(line[0])
        X = np.swapaxes(X,0,1)
        X = X.reshape(X.shape[0],X.shape[1],1)
        shapeFirst = X.shape[0]
        X = nomalizedDataSet(X)
        # make sure that the input is padded to keep the same dimentions for stacking later on
#        X = np.pad(X, ((maxSize - shapeFirst, 0), (0, 0)))
        X = np.pad(X, ((maxSize - shapeFirst, 0), (0, 0), (0, 0)))
        Y = int(line[1])-6
        XArr.append(X)
        YArr.append(Y)
    #XArr = preprocessX(XArr)
    XArr = np.stack(XArr, axis=0)
    #XArr = rangeRegularization(XArr)

    YArr = np.asarray(YArr)
    return XArr,YArr


# helper functions

def nomalizedDataSet(dataset):
    meanVals = dataset.mean(axis=(0), keepdims=True)
    dataset = dataset - meanVals
    stdVals =  np.std(dataset, axis=(0), keepdims=True)
    dataset = np.divide(dataset,stdVals)
    return dataset

def nomalizedDataSetFreq(dataset):
    meanVals = dataset.mean(axis=(0), keepdims=True)
    dataset = dataset - meanVals
    stdVals =  np.std(dataset, axis=(0), keepdims=True)
    dataset = np.divide(dataset,stdVals)
    return dataset

def rangeRegularization(dataset):
    minVals = dataset.min(axis=(1,2),keepdims=True)
    maxVals = dataset.max(axis=(1, 2), keepdims=True)
    dataset = dataset - minVals
    dataset = np.divide(dataset, maxVals)
    return dataset
    
def rangeRegularizationForInference(dataset):
    minVals = dataset.min(axis=(0,1),keepdims=True)
    maxVals = dataset.max(axis=(0, 1), keepdims=True)
    dataset = dataset - minVals
    dataset = np.divide(dataset, maxVals)
    return dataset
    
def preprocessX(data,addChannelDim=False):
    data = np.stack(data, axis=0)
    if addChannelDim==True:
       data = np.expand_dims(data, axis=(3))
    return data




