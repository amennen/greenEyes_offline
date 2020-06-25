

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s - %(message)s')
import numpy as np
import pickle
import nibabel
import nilearn
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.plotting import show
from nilearn.plotting import plot_roi
from nilearn import image
from nilearn.masking import apply_mask
# get_ipython().magic('matplotlib inline')
import scipy
import matplotlib
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.input_data import NiftiMasker
#from nilearn import plotting
import nibabel
from nilearn.masking import apply_mask
from nilearn.image import load_img
from nilearn.image import new_img_like
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, svm, metrics
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectFwe
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy import interp
import csv
params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, GenericUnivariateSelect, SelectKBest, chi2
from sklearn.feature_selection import RFE
import os
import pandas as pd
import csv
from scipy import stats
import brainiak
import brainiak.funcalign.srm
import sys
from sklearn.utils import shuffle
import random
from datetime import datetime
random.seed(datetime.now())
from sklearn.model_selection import train_test_split,KFold
## SPECIFY PARAMETERS FOR CLASSIFIER HERE

def getNumberofTRs(stationsDict,st):
    this_station_TRs = np.array(stationsDict[st])
    n_station_TRs = len(this_station_TRs)
    return n_station_TRs, this_station_TRs


maskType = 1
removeAvg = 1
filterType = 0
k1 = 0
k2 = 25
constant_classifier = 0
if constant_classifier:
    TR_to_use = 15 + 378  #index begins at 378


# 1. Take original data
logging.debug('parsing original data')
projectDir='/jukebox/norman/amennen/prettymouth_fmriprep2/'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD2.nii.gz'
fmriprep_dir=projectDir + '/derivatives/fmriprep'
# NEW: only train for story TRS!!!

# 2. Load station information
logging.debug('Getting station information ready')
# OLDs
#stationsDict = np.load('stations_iter2.npy').item()
#nStations = len(stationsDict)

# changed 3/14: making new stations dictionary to test all station points
#stationsDict = np.load('mega_testing_stations2.npy').item()

#good_stations = np.array([0,2,5,7,9,12,15,33,39,56]) # these are the good indices to test
# new from 3/20 - looking through each station and taking longer spatiotemporal ones
#good_stations= np.array([0,2,5,7,9,12,15,17,20,21])

# NEXT new one--combining multiple mega and upper right results!!s
#stationsDict = np.load('combined_stations.npy').item()
#stationsDict = np.load('mega_winners_nofilter.npy').item()
stationsDict = np.load('upper_right_winners_nofilter.npy',allow_pickle=True).item()
nStations = len(stationsDict)
good_stations = np.arange(nStations) # because I specified all of the stations here!

nStations = len(good_stations)
# load subject numbers
logging.debug('loading subject numbers')
subInd = 0
nsub=38
allnames = []
allgroups = []
groupInfo={}
# skip subjects 039 and 116
with open(projectDir + 'participants.tsv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if 'sub' in row[0]:
            # now skip the subjects we don't want to analyze
            allInfo = row[0].split('\t')
            subjName=allInfo[0]
            if subjName != 'sub-039' and subjName != 'sub-116':
                if allInfo[3] == 'paranoia':
                    group = 0
                elif allInfo[3] == 'affair':
                    group = 1
                allnames.append(subjName)
                allgroups.append(group)
                subInd+=1
paranoidSubj = allnames[0:19]
cheatingSubj = allnames[19:]
paranoidLabel = allgroups[0:19]
cheatingLabel = allgroups[19:]
n_per_category=19
story_TR_1 = 14
story_TR_2 = 464
nVox = 2414
training_data = np.load('training_data_testing_predict_proba_stations.npy')


# first we have to train the data-- this time don't set proability = True
st = 0
stationInd = good_stations[st]
clf = SVC(kernel='linear')
this_station_TRs = np.array(stationsDict[stationInd])
n_station_TRs = len(this_station_TRs)
training_data_station = training_data[:,:,this_station_TRs]
training_data_reshaped = np.reshape(training_data_station,(nsub,nVox*n_station_TRs))
clf.fit(training_data_reshaped,trainingLabels)
# cross validation -- need to leave one out of each group each time
X_train, X_test, y_train, y_test = train_test_split(training_data_reshaped,trainingLabels,test_size=2/38)


kf = KFold(n_splits=19)

for train_index,test_index in kf.split()

