import scipy.io
import os
import numpy as np
root = r"D:\Studies\DeepLearningCourse\Project\imagined_speech\Base de Datos Habla Imaginada"
outRoot = r"D:\Studies\DeepLearningCourse\Project\imagined_speech\Parsed\Pronounced"
folders = os.listdir(root)
hop = 4096
for folder in folders:
    if "." not in folder:
        files = os.listdir(os.path.join(root,folder))
        for file in files:
            if "EEG" in file:
                speaker = file.split("_")[0]
                completeFile = os.path.join(root,folder)
                completeFile = os.path.join(completeFile,file)
                mat = scipy.io.loadmat(completeFile)
                mat = mat["EEG"]
                # for index -3, imagined is 1, pronounced is 2
                mat = mat[(mat[:,-3])==2]
                # filter out all the vowels and keep the words
                mat = mat[(mat[:,-2]>=6) & (mat[:,-2]<=11)]
                counter = 0
                for row in mat:
                    signalArr = []
                    label = str(int(row[-2]))
                    for i in range(6):
                        signalArr.append([])
                        start = int(i*hop)
                        stop = int((i+1)*hop)
                        signal = row[start:stop]
                        signalArr[i]=signal
                    signalArr = np.asarray(signalArr[:])
                    outPath = os.path.join(outRoot,speaker)
                    outPath = os.path.join(outPath,speaker+"_"+label+"_"+str(counter)+".npy")
                    np.save(outPath,signalArr)
                    counter+=1
