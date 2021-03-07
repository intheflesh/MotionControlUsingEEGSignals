import scipy.io
import os
import numpy as np
import soundfile as sf
outputFolder = r"C:\Users\romanf\PycharmProjects\DeepLearningCourseProject\imagined_speech\EEGSignalsSegmentsWhileSpeechIsPronounced\FullDuration"
inPath = r"C:\Users\romanf\PycharmProjects\DeepLearningCourseProject\imagined_speech\Base de Datos Habla Imaginada"

data_tot_dict = {}
for filename in os.listdir(inPath):

    if '.' not in filename:
        folder_name = filename
        data_tot_dict[folder_name] = {'EEG': None, 'Audio': None, 'Data': None}
        for filename in os.listdir(inPath+'/%s' % filename):

            if filename[-4:] == '.txt':
                f = open(inPath+"/%s/%s" % (folder_name, filename), "r", encoding='latin-1')
                f_r = f.read()
                f_r_split = f_r.split('\n')
                list_to_dict_data = np.array([el.split("=") for el in f_r_split])
                dict_data = dict(zip(list_to_dict_data[:, 0], list_to_dict_data[:, 1]))

                if len([dict_data['Sujeto']]) == 2:
                    dict_data['Sujeto'] = dict_data['Sujeto'][0] + '0' + dict_data['Sujeto'][-1]

                data_tot_dict[folder_name]['Data'] = dict_data

            if filename.endswith('Audio.mat'):
                data_tot_dict[folder_name]['Audio'] = \
                scipy.io.loadmat(inPath+"/%s/%s" % (folder_name, filename))['Audio']

            if filename.endswith('EEG.mat'):
                data_tot_dict[folder_name]['EEG'] = \
                scipy.io.loadmat(inPath+"/%s/%s" % (folder_name, filename))['EEG']


for key,value in data_tot_dict.items():
  labels_all=value['Audio'][:,-2]
  just_words=labels_all>=6
  rows_in_EEG=value['Audio'][:,-1]
  ordered_EEG=value['EEG'][rows_in_EEG.astype(int)-1,:]
  data_tot_dict[key]['ordered_EEG']=ordered_EEG[just_words,:-3]
  data_tot_dict[key]['EEG_elec']={}
  data_tot_dict[key]['ordered_Audio']=value['Audio'][just_words,:-2]
  data_tot_dict[key]['ordered_labels']=value['Audio'][just_words,-2]  # the labels

# now export to files, and later divide audio to segments and exports only relevant parts

for key,value in data_tot_dict.items():
    counter = 0
    Audio = value['ordered_Audio']
    Labels = value['ordered_labels']
    EEGVec = value['ordered_EEG']
    for i in range(len(Labels)):
        fileName = str(int(Labels[i]))+"_"+str(counter)
        counter+=1
        currentAudio = Audio[i]
        currentEEG = EEGVec[i]
        EEGMat = []
        hop = 4096
        for j in range(6):
            EEGMat.append(currentEEG[int(j*hop):int((j+1)*hop)])
        EEGMatNP = np.array([np.array(xi) for xi in EEGMat])
        outputPath = os.path.join(outputFolder,key)
        outputPath = os.path.join(outputPath,fileName)
        sf.write(outputPath+".wav",currentAudio,samplerate=44100)
        np.save(outputPath,EEGMatNP)

