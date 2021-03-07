import webrtcvad
import soundfile as sf

from pyvad import vad, trim, split
from librosa import load
import matplotlib.pyplot as plt
import numpy as np
import os
samplingRate = 44100
EEGSamplingRate = 1024
inPath          = r"C:\Users\romanf\PycharmProjects\DeepLearningCourseProject\imagined_speech\EEGSignalsSegmentsWhileSpeechIsPronounced\netSpeech"
wavFilesPath    = r"C:\Users\romanf\PycharmProjects\DeepLearningCourseProject\imagined_speech\EEGSignalsSegmentsWhileSpeechIsPronounced\FullDuration"
folders = os.listdir(inPath)
for folder in folders:
    currentFolder = os.path.join(inPath,folder)
    files = os.listdir(currentFolder)
    for file in files:
        if file.endswith(".txt"):
            currentFileTXT = os.path.join(currentFolder, file)
            currentFileWAV = currentFileTXT.replace(".txt",".wav")
            currentFileWAV = currentFileWAV.replace("netSpeech","FullDuration")
            currenFileEEG = currentFileWAV.replace(".wav",".npy")
            inFileObj = open(currentFileTXT,"r")
            data,sampleRate = sf.read(currentFileWAV)
            lineCounter = 0
            for line in inFileObj:

                words = line.rstrip().split("\t")[0:2]
                start = float(words[0])
                stop = float(words[1])
                startSampleWAV  = int(start*samplingRate)
                stopSampleWAV   = int(stop*samplingRate)
                startSampleEEG  = int(start*EEGSamplingRate)
                stopSampleEEG = int(stop*EEGSamplingRate)
                EEGMat = np.load(currenFileEEG)
                currentFilename = file.split(".")[0]+"_"+str(lineCounter)
                lineCounter+=1
                outFileWAV = os.path.join(currentFolder,currentFilename+".wav")
                outFileEEG = os.path.join(currentFolder, currentFilename)
                sf.write(outFileWAV,data[startSampleWAV:stopSampleWAV],samplerate=samplingRate)
                subMat = EEGMat[:,startSampleEEG:stopSampleEEG]
                np.save(outFileEEG,subMat)
                a=0

