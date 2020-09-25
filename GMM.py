import glob
import MFCC as feature
import numpy
import os
from sklearn import mixture
import joblib
import matplotlib.pyplot as plt
import sys
import graph as g

# store the data set in two folders namely Train and Test
# Training:
# Train Folder consist of 86 folders and each folder represent a speaker
# Each speaker has 8 wav files for training
# For every speaker in Train Folder and for every .wav file extract features using MFCC.py
# Store the all the extracted features of each speaker
# create gaussian mixture using sklearn and fit the extracted features into Gmm
# For every speaker create a gaussian mixture model
# Store the model into folder called Train_Data folder of Train_Model folder as .pkl files
# Testing:
# Test folder consist of 196 speaker
# Each folder has 2 wav files for testing
# For every speaker and every audio file extract mfcc
# Get every model stored while training and fit extracted mfcc and calculate the score
# The model with maximum score is the predicted speaker label for given test audio clip
# Accuracy is measured by calculating the percentage of matches of speaker model label and test speaker label
# Plotting the effect of accuracy by varying the number of gaussian components

accuracy = []
number_components = []
speaker_list = {}
key = 0
covariance = ['diag', 'full', 'tied']  # covariance type
initialization = ['kmeans', 'random']  # initialization methods
# for every combination of covariance matrix and initialization method we find accuracy for varying order of model
for matrix in covariance:  # for every covariance we find accuracy
    for initial in initialization:  # for every initialization method we calculate accuracy
        print(matrix)
        print(initial)
        total_speaker = len(glob.glob('Train/*'))  # Number so speakers for training the model
        print('Total Training Speakers', total_speaker)
        for num in range(1, 21):  # Number of gaussian components
            for speaker in (glob.glob('Train/*')):  # pick one speaker at a time in the folder Train
                name_speaker = speaker
                # print(name_speaker)
                speaker_list.update({name_speaker: key})  # store the speaker along with a key in the list

                speaker_data = []  # to store the mel frequency cepstral coefficients for different audio signals of
                # same speaker

                for speaker_train in glob.glob(speaker + '/*.wav'):  # pick all the audio clips of each speaker
                    mfcc = feature.MFCC(speaker_train)  # extract mfcc for each audio clip
                    if len(speaker_data) == 0:
                        speaker_data = mfcc
                    else:
                        speaker_data = numpy.concatenate(
                            (speaker_data, mfcc))  # store mfccs of all audio clips of single speaker

                # print(speaker_data.shape)
                gaussian_mixture_model = mixture.GaussianMixture(n_components=num, covariance_type=matrix, max_iter=100,
                                                                 init_params=initial)  # create gmm using library in
                # sklearn package
                gaussian_mixture_model = gaussian_mixture_model.fit(
                    speaker_data)  # fit the gmm using mfcc of each speaker
                score = gaussian_mixture_model.score(
                    speaker_data)  # calculate per-sample average log-likelihood of mfcc
                # print(score)

                joblib.dump(gaussian_mixture_model,
                            'Train_Model/' + name_speaker + '.pkl')  # to serialize the model and store it in folder
                # in pickel format
                key = key + 1  # updating key

            print('Training Completed Successfully')

            # for speaker in speaker_list:
            # print(speaker)

            match = 0  # initializing number of matches to zero
            num_test = 0  # initializing number of test audio clips to zero

            for test in glob.glob('Test/*'):  # for every speaker in test folder
                # print(test)

                for test_speaker in glob.glob(test + '/*.wav'):  # for every audio clip of each test speaker
                    # print(test_speaker)
                    mfcc = feature.MFCC(test_speaker)  # extracting mfcc for each audio clip
                    # print(mfcc.shape)
                    test_lable = test_speaker.replace('Test', '').replace('9', '').replace('10', '').replace('.wav',
                                                                                                             '')  # saving the name of test speaker
                    threshold = -1000  # it is used to initialize score of each speaker by i picked it randomly as
                    # -1000 since score be cant be less -1000
                    num_test += 1  # update test
                    for model in glob.glob('Train_Model/Train/*.pkl'):  # for every trained model
                        # print(model)
                        gaussian_mixture_model = joblib.load(model)  # retrieve the gmm from the pkl file
                        # print(model)
                        score = gaussian_mixture_model.score(
                            mfcc)  # load the gmm using the mfcc of test speaker audio clip
                        # print(model,score)
                        if score > threshold:
                            threshold = score
                            speaker_label = model.replace('Train_Model/Train', '').replace('.pkl',
                                                                                           '')  # retrieve the
                            # trained speaker label

                    # print(test_lable+'--->'+speaker_label)

                    if test_lable == speaker_label + '/':  # compare the name of test_speaker and speaker that got
                        # high score
                        match += 1

            print('Number of Gaussian Components', num)
            print('Number of Matches:', match)
            print('Number of Test Audios:', num_test)
            percentage = (match / num_test) * 100  # computing accuracy
            print('ACCURACY:', percentage)
            accuracy.append(percentage)  # storing accuracy for different number of gaussian components
            number_components.append(num)  # storing number of gaussian components

        # plotting accuracy against number of gaussian components
        g.plot(matrix, initial, number_components, accuracy)
