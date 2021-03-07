import webrtcvad
import soundfile as sf
vad = webrtcvad.Vad(2)
from pyvad import vad, trim, split
from librosa import load
import matplotlib.pyplot as plt
import numpy as np
import os
samplingRate = 44100
inPath = r"C:\Users\romanf\PycharmProjects\DeepLearningCourseProject\imagined_speech\EEGSignalsSegmentsWhileSpeechIsPronounced\FullDuration"
outPath = r"C:\Users\romanf\PycharmProjects\DeepLearningCourseProject\imagined_speech\EEGSignalsSegmentsWhileSpeechIsPronounced\netSpeech"
folders = os.listdir(inPath)
for folder in folders:
    currentFolder = os.path.join(inPath,folder)
    files = os.listdir(currentFolder)
    for file in files:
        if file.endswith(".wav"):
            fullPathCurrentFile = os.path.join(currentFolder,file)
            outFile = fullPathCurrentFile.replace(".wav",".txt")
            outFile = outFile.replace("FullDuration", "netSpeech")
            data, fs = load(fullPathCurrentFile,sr=samplingRate)
            VADVec = vad(data, fs, fs_vad=32000, hop_length=10, vad_mode=3)
            first = VADVec[0]
            counter = 0
            outFile = open(outFile,"w")
            start = 0
            stop = 0
            for val in VADVec[1:]:
                counter+=1
                second = val
                if first == 0 and second == 1:
                    start = counter/samplingRate
                elif first == 1 and second == 0:
                    stop = counter/samplingRate
                    outFile.write (str(start) + "\t" + str(stop) + "\ts\n")
                first = second
