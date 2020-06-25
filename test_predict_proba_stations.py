# purpose of this script: train a classifier from ALL 38 previous subjects based on given classifier parameters FROM STATIONS!!!
###### PARAMETERS:
# 1. k1
# 2. k2
# 3. remove avg
# 4. mask
# 5. filter
# 6. classifier type



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

# 5. If k2 > 0, build 2 different SRMs and only keep the shared signal within each group

#training_data_classifier_SRM = np.zeros((nSub,vox_num,nTR_new)) # each time point gets its own s x v classifier

# now do second level SRM 2 WITH ALL SUBJECTS!

trainingLabels = np.array([paranoidLabel,cheatingLabel ]).flatten()
stationsDict = np.load('upper_right_winners_nofilter.npy',allow_pickle=True).item()
nStations = len(stationsDict)
good_stations = np.arange(nStations) # because I specified all of the stations here!

all_classifications_cheating = np.zeros((nsub,nStations))
all_classification_prediction = np.zeros((nsub,nStations))
nStations = len(good_stations)
# now build a classifier for each station
for st in np.arange(nStations):
    stationInd = good_stations[st]
    filename = 'saved_classifiers' + '/' 'LOGISTIC_default_UPPERRIGHT_stationInd_' + str(stationInd) + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
    classifier = pickle.load(open(filename, 'rb'))
    n_station_TRs,these_trs = getNumberofTRs(stationsDict,st)
    training_data_station = training_data[:,:,these_trs]
    dataForClassification_reshaped = np.reshape(training_data_station,(nsub,nVox*n_station_TRs))
    all_classifications_cheating[:,st] = classifier.predict_proba(dataForClassification_reshaped)[:,1]
    all_classification_prediction[:,st] = classifier.predict(dataForClassification_reshaped)

s =0 
print(np.shape(training_data))
this_subj_data = training_data[s,:,:]
print(np.shape(this_subj_data))
training_data_station = this_subj_data[:,these_trs]

print(np.shape(training_data_station))
print('*****************************')
print(np.shape(training_data))
this_subj_data = training_data[s,:,these_trs]
print(np.shape(this_subj_data))
### CHECK THE RIGHT SHAPE IN DEBUG MODE FOR CLASSIFICATION *******
# add assertion that the data is voxel x TR shape!!!!!!!
dataForClassification_reshaped_individ_subj = np.reshape(training_data_station,(1,nVox*n_station_TRs))


# first do for real cheating group -- first ones
all_cheating_group = all_classifications_cheating[0:19,:]
plt.figure()
plt.subplot(1,2,1)
for s in np.arange(int(nsub/2)):
    plt.plot(np.arange(nStations),all_cheating_group[s,:], '.', ms=20, color='r')
plt.xlabel('Station number')
plt.ylabel('Probability(cheating)')
plt.title('Real cheating group')
plt.ylim([0,1])
plt.subplot(1,2,2)
all_paranoid_group = all_classifications_cheating[19:,:]
for s in np.arange(int(nsub/2)):
    plt.plot(np.arange(nStations),all_paranoid_group[s,:], '.', ms=20, color='g')
plt.xlabel('Station number')
plt.ylabel('Probability(cheating)')
plt.title('Real paranoid group')
plt.ylim([0,1])
plt.show()

