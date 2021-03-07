import operator
import os
import io
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow_addons as tfa
from absl import logging
from ModelLib import chosenCNNModel,effecientNet,multiLayerGRUAndConv,resNet152V2
from loadData import loadData,loadDataFromList,getMaxSize,loadDataFromListTemporal
import keras




# Function which returns subset or r length from n
from itertools import combinations


speakerArr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
r = 1   # this can be 2 if you want to do "keep pair out" or any other number for that matter
combin = list(combinations(speakerArr, r)) # 105 of those
random.shuffle(combin)

logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')

# first decide on the excluded speakers - in our case we use "leave pair out" since we have only 15 speakers
# excluded speakers are inexed by a number from 1 to 15 in accordance to the official folders

# path for the feature files
dataPath = r"D:\Studies\DeepLearningCourse\Project\Features_FW80_FS40_FFT_512\Features_FW80_FS40_FFT_512"

# this file will include the results per fold (that's why it's called big picture)
bigPicture = r"D:\Studies\DeepLearningCourse\Project\Progress3\BigPicture\progress.txt"
# this folder will include txt files that will represent the progress per fold, where train and test accuracy is documented as training proceeds
perFold = r"D:\Studies\DeepLearningCourse\Project\Progress3\PerFold"

batchSize = 32
epochs = 40

listOfFilesAndLabelsTrain,listOfFilesAndLabelsTest = loadData([],dataPath)
maximum = getMaxSize(listOfFilesAndLabelsTrain) # needed for padding later
# here we used callbacks to write out the accuracy
class TrackTrainingModel(tf.keras.callbacks.Callback):

    def __init__(self, currentFold):
        self.currentFold = currentFold

    def on_epoch_end(self, epoch, logs={}):
        accuracyTest = round(logs['val_accuracy'], 4)
        accuracyTrain = round(logs['accuracy'], 4)
        currentFile = os.path.join(perFold,str(currentFold)+".txt")
        tmpFileObj = open(currentFile, "a")
        if accuracyTest>topScoresDict[currentFold]:
            topScoresDict[currentFold] = accuracyTest
        tmpFileObj.write("Epoch_"+str(epoch) + "_AccuracyTrain=" + str(accuracyTrain) + "_AccuracyTest=" + str(accuracyTest)+"\n" )
        tmpFileObj.close()

    def on_train_end(self, logs={}):
        accuracyTest = round(logs['val_accuracy'], 4)
        accuracyTrain = round(logs['accuracy'], 4)
        tmpFileObj = open(bigPicture,"a")
        tmpFileObj.write(str(self.currentFold)  + "_AccuracyTrain=" + str(accuracyTrain) + "_AccuracyTest=" + str(accuracyTest)+"\n" )
        tmpFileObj.close()


topScoresDict = {}
for currentFold in combin:
    exludedSpeakersForEval = currentFold
    topScoresDict [ currentFold ] = 0

    listOfFilesAndLabelsTrain,listOfFilesAndLabelsTest = loadData(exludedSpeakersForEval,dataPath)

    XTrain,YTrain = loadDataFromList(listOfFilesAndLabelsTrain,maximum)
    XTest,YTest = loadDataFromList(listOfFilesAndLabelsTest,maximum)
    shapes = XTrain.shape
    model = multiGRUAndConv5_2.GRUAndConv(shapes[1],shapes[2],shapes[3])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0000001),      # we noticed that a slow rate here is helpful
        loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']

    )
    #print(model.summary())
    # Train the network
    history = model.fit(x=XTrain,y=YTrain,
                        batch_size=batchSize,
                        validation_data=(XTest,YTest),
                        epochs=epochs,
                        shuffle=True,callbacks=[TrackTrainingModel(currentFold)])

# here we just print out the top scoring models per fold
sorted_x = sorted(topScoresDict.items(), key=operator.itemgetter(0))
for val in sorted_x:
    print(str(val))
