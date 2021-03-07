import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt


inFilePath = r"D:\Studies\DeepLearningCourse\Project\FinalResults.txt"
inObj = open(inFilePath,"r")
speakerArr = []
accuracyArr = []
for line in inObj:
    line = line.rstrip()
    words = line.split(" ")
    speakerArr.append(int(words[0]))
    accuracyArr.append(100*float(words[1]))
plt.xlabel("Subject No.")
plt.ylabel("Accuracy (%)")
plt.ylim([0,100])
plt.minorticks_on()
plt.axhline(16.666,color='r')
plt.axhline(np.average(accuracyArr),color='b',linestyle='--')
plt.bar(speakerArr,accuracyArr)
plt.grid()
plt.show()