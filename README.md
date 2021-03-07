# MotionControlUsingEEGSignals
This work is based on the following article:
https://www.researchgate.net/publication/312953157_Open_access_database_of_EEG_signals_recorded_during_imagined_speech
This is a repo for our implementation of a 6 label motion control using EEG signals
This code enables data preparatin and model training.
# data preperation

1) first download the "Imagined Speech" dataset from the following link:
https://drive.google.com/file/d/0By7apHbIp8ENZVBLRFVlSFhzbHc/view
2) now we would like to take the raw dataset that comes in mat format and turn it into something more python friendly like npy format, and in addition - 
to filter out the vowels and keep the 6 commands "up down forward backward right left" - for that run "extractRelevantEEG.py" 
3) now run "extractEEGFromDataSet.py" with updated input and output folders (the output folder should include also subfolders named "S01" to S15" for the 15 different speakers)
to convert the mat files to .npy that we can easily work with.

in case of pronounced and imagined speech, 
these steps will produce both WAV files and npy files with the same name, only different formats.
the naming is as follows:
x_y.wav
or
x_y.npy
where x is a number from 6 to 11 representing the pronounced words (6-11 is: up down forward backward right left correspondingly)
and y is just a running number - no real meaning, just so that the files will not be overwritten.

each .wav or .npy (EEG) file represents exactly 4 seconds, 
where the .wav is just a stream of samples, sampled with a freq of 44100 Hz,
and the .npy file is a matrix of 6X4096 values, where each row is a different sensor, and the sensor sampling freq is 1024 Hz.
each vector represents a different sensor - creating the follwing order: F3,F4,C3,C4,P3,P4

3) now we would liket to detect when speech occured to extract the relevant segments, for that you need to run "VAD.py" and it would create .txt files with a list of segments
4) now you need to actually slice the segments out - for that - run "ChopSegmentsOut.py" - this in turn will split our x_y format to x_y_z where z is also a running number
5) now finally we can start processing the relevant EEG segments - first we can filter out the segments that are too short or too long - as the VAD can produce errors -
for that run "filterShortAndLongWords.py" - it filters out all files below 15 precentile and over 95 precentile - per word (so we have 6X2 precentiles, and you can change the values as you see fit).
6) now we need to perform framing, windowing and convert our EEG frames to spectrograms - all this is done in "ConvetTimeToFreq.py"

and now the features are ready.

# training

in order to run the actual training - simply update the dataPath variable under "run.py" to the features root folder and run it.
This in turn, will perform multiple training steps with the "leave one out" method, as the number of speakers and data size is limited.

Enjoy!
