import webrtcvad
import soundfile as sf
from librosa import load
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

inPath = r"D:\Studies\DeepLearningCourse\Project\imagined_speech\Parsed\Imagined"
outPath = r"D:\Studies\DeepLearningCourse\Project\Features_FW80_FS40_FFT_512_FullLen"
folders = os.listdir(inPath)
frameWidth = 80 # in samples, where sampling freq is 1024
frameShift = 40 # in samples, where sampling freq is 1024
FFTSize = 1024
HalfFFTSize = int(FFTSize/2)
for folder in folders:
    currentFolder = os.path.join(inPath,folder)
    files = os.listdir(currentFolder)
    for file in files:
        if file.endswith(".npy"):
            fullPathCurrentFile = os.path.join(currentFolder,file)
            mat = np.load(fullPathCurrentFile)
            outFile = fullPathCurrentFile.replace(".npy", "_FFT")
            totalMat = []
            rowCounter = 0
            numOfFrames = int(1 + np.floor((len(mat[0]) - frameWidth) / frameShift))
            shape = np.shape(mat)
            # the resulting matrix is of shape [num of frames (depending on audio), # of sensors (6),  FFT resolution (512)]
            returingMat = np.zeros((numOfFrames,shape[0],HalfFFTSize))
            for i in range(numOfFrames):
                currentChunk = mat[:,int(i * frameShift):int(i * frameShift + frameWidth)]
                totalMat.append([])
                for row in currentChunk:
                    first = np.mean(row)
                    second = np.std(row)
                    row = np.multiply(row,np.hanning(len(row)))
                    tmpRow = np.zeros((HalfFFTSize))
                    fx = np.fft.fft(row,FFTSize)
                    fx = abs(fx)
                    tmpRow[0] = first
                    tmpRow[1] = second
                    tmpRow[0:HalfFFTSize] = fx[0:HalfFFTSize]
                    totalMat[rowCounter].append(tmpRow)
                    a=0
                EEGMatNP = np.array([np.array(xi) for xi in totalMat])
                rowCounter+=1
                returingMat[i] = EEGMatNP[i]
            outFilePath = os.path.join(outPath,folder)
            outFilePath = os.path.join(outFilePath,file)
            np.save(outFilePath,returingMat)

