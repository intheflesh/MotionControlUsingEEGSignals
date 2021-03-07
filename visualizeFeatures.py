import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt

root = r"D:\Studies\DeepLearningCourse\Project\imagined_speech\Parsed\Imagined\S01"
files = os.listdir(root)
tmpDict = {}
for file in files:
    word = file.split("_")
    word = int(word[1])
    if word not in tmpDict.keys():
        tmpDict[word] = os.path.join(root,file)
titles = ['up','down','forward','backward','right','left']



for i in range(6,12):
    ax= plt.subplot(6, 1,i-5)
    if i!=11:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
    else:
        ynew = []
        hop = 512
        for j in range(0, 4096, hop):
            ynew.append(str(float(j) / 1024))
        ynew.append(4)
        ynew = np.asarray(ynew)

        x = np.arange(0, 4097, hop)
        plt.xticks(x, labels=ynew)
        plt.setp(ax.get_xticklabels(), visible=True)
    plt.yticks(ticks=np.arange(7),fontsize=6)
    plt.setp(ax.get_xticklabels(), visible=True, fontsize=10)
    plt.subplots_adjust(hspace=0.8)

    plt.title(titles[i-6],loc="left")
    mat = np.load(tmpDict[i])
    mat[[3, 1]] = mat[[1, 3]]
    mat[[4, 1]] = mat[[1, 4]]
    mat[[2, 1]] = mat[[1, 2]]
    #mat = mat[:,0:500]
    #plt.gca().axes.get_yaxis().set_visible(False)
    #x = np.arange(6)  # len = 10


    plt.pcolormesh(mat)

    #plt.imshow(mat, aspect = 1.2,extent=[0,4096,0,300])
    #plt.grid(axis='x')
plt.xlabel("Time [seconds]")




plt.show()


