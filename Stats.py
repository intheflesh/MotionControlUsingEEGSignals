import webrtcvad
import soundfile as sf
vad = webrtcvad.Vad(2)
from pyvad import vad, trim, split
from librosa import load
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def drawHistOfAllDurations():
    inPath = r"D:\Studies\DeepLearningCourse\Project\TemporalFeatures\netSpeechOriginalDoNotTouch"
    folders = os.listdir(inPath)
    durationArrTotal = []
    for folder in folders:
        currentFolder = os.path.join(inPath,folder)
        files = os.listdir(currentFolder)

        for file in files:
            if file.endswith(".txt"):
                fullPathCurrentFile = os.path.join(currentFolder,file)
                inObj = open(fullPathCurrentFile,"r")
                for line in inObj:
                    line = line.rstrip()
                    words = line.split("\t")
                    duration = float(words[1])- float(words[0])
                    durationArrTotal.append(duration)

    binVals = np.arange(0.05, 3.2, step=0.15)

    plt2 = drawHist(binVals,durationArrTotal)
    plt2.show()


def drawHistPerWord():
    inPath = r"D:\Studies\DeepLearningCourse\Project\TemporalFeatures\netSpeechOriginalDoNotTouch"
    folders = os.listdir(inPath)
    labels = ["up","down","forward","backward","right","left"]
    durations = [[],[],[],[],[],[]]
    for folder in folders:
        currentFolder = os.path.join(inPath, folder)
        files = os.listdir(currentFolder)

        for file in files:
            if file.endswith(".txt"):
                fullPathCurrentFile = os.path.join(currentFolder, file)
                inObj = open(fullPathCurrentFile, "r")
                for line in inObj:
                    line = line.rstrip()
                    words = line.split("\t")
                    duration = float(words[1]) - float(words[0])
                    word = int(file.split("_")[0])-6
                    durations[word].append(duration)
    binVals = np.arange(0.05, 3.2, step=0.25)
    plt.subplots_adjust(hspace=0.6)

    for val in range(6):
        plt.subplot(3,2,val+1)
        plt2 = drawHist(binVals, durations[val])
        plt2.ylim((0,500))
        plt2.title(labels[val])
    plt2.show()





def drawHistPerWordUpdated():
    inPath = r"D:\Studies\DeepLearningCourse\Project\TemporalFeatures\netSpeechOriginalDoNotTouch"
    folders = os.listdir(inPath)
    labels = ["up","down","forward","backward","right","left"]
    durations = [[],[],[],[],[],[]]
    for folder in folders:
        currentFolder = os.path.join(inPath, folder)
        files = os.listdir(currentFolder)

        for file in files:
            if file.endswith(".txt"):
                fullPathCurrentFile = os.path.join(currentFolder, file)
                inObj = open(fullPathCurrentFile, "r")
                for line in inObj:
                    line = line.rstrip()
                    words = line.split("\t")
                    duration = float(words[1]) - float(words[0])
                    word = int(file.split("_")[0])-6
                    durations[word].append(duration)

    meanArr = []
    stdArr = []
    for val in durations:
        meanArr.append(np.mean(val))
        stdArr.append(np.std(val))
    plt.figure()
    plt.bar(range(len(meanArr)), meanArr,stdArr);
    #plt.xticks(range(len(words.keys())), labels=df_mean_std_length_words.columns.to_numpy(dtype=str));
    plt.title('Mean and STD of Words Duration');
    plt.show()
    a=0
    '''
    plt.figure()
    plt.bar(range(len(words.keys())), df_mean_std_length_words.loc['Mean Length'],
            yerr=df_mean_std_length_words.loc['STDError Length']);
    plt.xticks(range(len(words.keys())), labels=df_mean_std_length_words.columns.to_numpy(dtype=str));
    plt.title('Mean and STD of Words Duration');
    plt.ylabel('ms');
    '''

def drawHist(bins,data):
    result = plt.hist(data,(bins))
    arr = result[:][0]
    plt.grid()
    plt.xticks(bins)
    plt.ylabel("# of word instances")
    plt.xlabel("net-speech [seconds]")
    plt.title("# of word instances VS net-speech ")
    for i in range(len(bins)-1):
        plt.text(x = bins[i] , y = arr[i]+4, s = str (int(arr[i])), size = 10)
    return plt

drawHistPerWordUpdated()