# MotionControlUsingEEGSignals
This is a repo for our implementation of a 6 label motion control using EEG signals

This code enables data preparatin and model training. 
# data preperation

1) first download the "Imagined Speech" dataset, these are the relevant links:
article responsible for dataset collection: https://www.researchgate.net/publication/312953157_Open_access_database_of_EEG_signals_recorded_during_imagined_speech
direct dataset download: https://drive.google.com/file/d/0By7apHbIp8ENZVBLRFVlSFhzbHc/view

2) first we would like to take the raw dataset that comes in mat format and turn it into something more python friendly like npy format
for that we need to first run "extractEEGFromDataSet.py" with updated input and output folders.

this step will produce both WAV files and npy files with the same name, only different formats.
the naming is as follows:
x_y.wav
or
x_y.npy
where x is a number from 6 to 11 representing the pronounced words in this 
