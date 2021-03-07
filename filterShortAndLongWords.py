import os
import math
import numpy as np
import random
import os
if os.path.exists("demofile.txt"):
  os.remove("demofile.txt")

def deleteFiles(inPath):
    folders = os.listdir(inPath)
    wordArrays = [[],[],[],[],[],[]]
    listOfFilesAndLen = []
    for folder in folders:
        folderNameForCompare = folder.split("S")[1]
        folderNameForCompare = folderNameForCompare.split("0")[-1]
        currentPath = os.path.join(inPath,folder)
        files = os.listdir(currentPath)
        for file in files:
            if file.endswith(".npy"):
                curreFile = os.path.join(currentPath,file)
                currentNpyFile = np.load(curreFile)
                shape = currentNpyFile.shape
                length = shape[0]

                word = curreFile.split("_")[-3]
                word = word.split("\\")[-1]
                word = int(word)-6
                wordArrays[word].append(length)
                listOfFilesAndLen.append((curreFile,word,length))

    minVal = []
    maxVal = []
    for arr in wordArrays:
        arr = sorted(arr)
        vals = np.percentile(arr,[15,95])
        minVal.append(vals[0])
        maxVal.append(vals[1])
    for row in listOfFilesAndLen:
        curreFile = row[0]
        word = row[1]
        length = row[2]
        if length <= minVal[word] or length>=maxVal[word]:
            # delete file if too short or to long
            os.remove(curreFile)
        a=0
    a=0


mainPath = r"D:\Studies\DeepLearningCourse\Project\Features_FW80_FS30_Mean_STD_FFT_2048\Features_FW80_FS30_Mean_STD_FFT_2048"
deleteFiles(mainPath)
