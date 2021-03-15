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
from sklearn.linear_model import LogisticRegression
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
### CHANGING N STATIONS HERE ###
# For experiment 1 - there were 9 stations; For experiment 2 - there were 7 stations
nStations = 7 #len(stationsDict)
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
logging.debug('configuring mask')
if maskType == 0:
    starterStr = 'aggregate_data'
elif maskType == 1:
    starterStr = 'aggregate_data_TOM_large'
elif maskType == 2:
    starterStr = 'aggregate_data_TOM_cluster'
logging.debug('configuring filters')
if filterType == 0:
    filenameFILT = starterStr + '.npy'
    images_concatenated = np.load(filenameFILT)
elif filterType == 1:
    filenameFILT = starterStr + '_trunc_filtY.npy'
    images_concatenated = np.load(filenameFILT)
    story_TR_1 = story_TR_1 - 2
    story_TR_2 = story_TR_2 - 2
elif filterType == 2:
    filenameFILT = starterStr + '_trunc_filtS.npy'
    images_concatenated = np.load(filenameFILT)
    story_TR_1 = story_TR_1 - 2
    story_TR_2 = story_TR_2 - 2
vox_num, nTR, num_subs = images_concatenated.shape  # Pull out the shape data

logging.info('Participants %i' % num_subs)
logging.info('Voxels per participant %i' % vox_num)
logging.info('TRs per participant %i' % nTR)

# 2. Extract story TRs and zscore
logging.debug('extracting story and zscore')
original_data = []

for sub in range(num_subs):
    original_data.append(images_concatenated[:, story_TR_1:story_TR_2,sub])
for sub in range(num_subs):
    original_data[sub] = stats.zscore(original_data[sub],axis=1,ddof=1)
    original_data[sub] = np.nan_to_num(original_data[sub])
    
nTR = np.shape(original_data)[2] # recalculate to be story TRs instead of entire run
# 3. New version: DON'T loop through every iteration==just calculate after choosing f1/f2
logging.debug('starting new work prechecks')

indTrain = np.arange(n_per_category)

nTR_new = nTR
logging.info('NEW TRs per participant %i' % nTR_new)

# go through first SRM USING ALL SUBJECTS**

logging.info('k1 set to %i' %k1)
n_iter = 15
srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k1)
logging.info('Fitting SRM, may take a few minutes')
#logging.info('Fitting SRM, may take a few minutes')
if k1 > 0:
    srm.fit(original_data)
    S = srm.s_
#logging.info('SRM has been fit')
logging.info('SRM has been fit')
# Fit the SRM data
# the output matrix srm.s_ will be in features x time points space
# then project this output matrix onto each subject's voxel space
nVox = vox_num
nTRs_new = np.shape(original_data[0])[1]
#logging.info('number of new TRS: %i' % nTRs_new)
logging.info('number of new TRS: %i' % nTRs_new)
nSub = 38
#aggregated_SRM_removed = np.zeros((nVox,nTRs_new,nSub))
SRM_removed = []
for s in np.arange(nSub):
    if k1 > 0:
        w = srm.w_[s]
        signal_srm = w.dot(S) # reconstructed signal
    else:
        signal_srm = 0
    signal_original = original_data[s]
    subtracted_signal = signal_original - signal_srm
    #aggregated_SRM_removed[:,:,s] = subtracted_signal
    SRM_removed.append(subtracted_signal)

for sub in range(nSub):
    SRM_removed[sub] = stats.zscore(SRM_removed[sub],axis=1,ddof=1)
    SRM_removed[sub] = np.nan_to_num(SRM_removed[sub])
if removeAvg:
    subject_average = np.mean(SRM_removed,axis=0)
    SRM_removed = SRM_removed - subject_average

# 5. If k2 > 0, build 2 different SRMs and only keep the shared signal within each group

training_data_classifier_SRM = np.zeros((nSub,vox_num,nTR_new)) # each time point gets its own s x v classifier

# now do second level SRM 2 WITH ALL SUBJECTS!

srm1 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k2)
trainingInd = np.arange(n_per_category)
training_data = []
for sub in range(n_per_category):
    training_data.append(SRM_removed[trainingInd[sub]])
if k2 > 0:
    logging.info('Fitting SRM, may take a few minutes')
    srm1.fit(training_data)
    logging.info('SRM has been fit')
    # now we want to bring all subjects back into residual space but with ONLY the shared component
    # training data for the classifier will be a vector nsub x nvoxels*TR
    S1 = srm1.s_
    for s_ind1 in np.arange(n_per_category):
        w = srm1.w_[s_ind1]
        signal_srm = w.dot(S1) # reconstructed shared signal
        signal_transposed = signal_srm.T
        training_data_classifier_SRM[s_ind1,:,:] = signal_srm
else:
    logging.info('Not fitting SRM -- using original data')
    for s_ind1 in np.arange(n_per_category):
        signal_original = training_data[s_ind1]
        signal_transposed = signal_original.T
        training_data_classifier_SRM[s_ind1,:,:] = signal_original

trainingInd = np.arange(n_per_category) + n_per_category
logging.info(trainingInd)
srm2 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k2)
# remake list object for this iteration
training_data = []
for sub in range(n_per_category):
    training_data.append(SRM_removed[trainingInd[sub]])
if k2 > 0:
    logging.info('Fitting SRM, may take a few minutes')
    srm2.fit(training_data)
    logging.info('SRM has been fit')
    S2 = srm2.s_
    for s_ind2 in np.arange(n_per_category):
        w = srm2.w_[s_ind2]
        signal_srm = w.dot(S2) # reconstructed shared signal
        signal_transposed = signal_srm.T
        training_data_classifier_SRM[s_ind2+n_per_category,:,:] = signal_srm
else:
    logging.info('Not fitting SRM -- using original data')
    for s_ind2 in np.arange(n_per_category):
        signal_original = training_data[s_ind2]
        signal_transposed = signal_original.T
        training_data_classifier_SRM[s_ind2+n_per_category,:,:] = signal_original
# 7: Build classifier either by TR or spatiotemporal
# 8. Train classifier and save
trainingLabels = np.array([paranoidLabel,cheatingLabel ]).flatten()
logging.info(trainingLabels)

#np.save('training_data_testing_predict_proba_stations.npy',training_data_classifier_SRM)
# get all station values
good_subset = {key: stationsDict[key] for key in list(range(0,nStations))}
all_station_indices = sum(good_subset.values(), [])
non_station_indices = [i for i in list(range(0,nTR,1)) if i not in all_station_indices] 

# now build a classifier for ENTIRE timecourse OPPOSITE of stations (0-6 for Exp 2)
clf = LogisticRegression(solver='lbfgs',C=1)
training_data_opp_station = training_data_classifier_SRM[:,:,non_station_indices]
training_data_reshaped = np.reshape(training_data_opp_station,(nSub,vox_num*len(non_station_indices)))
clf.fit(training_data_reshaped,trainingLabels)
filename = 'saved_classifiers' + '/' 'LOGISTIC_lbfgs_UPPERRIGHT_OPPOSITEstations_' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
pickle.dump(clf,open(filename, 'wb'))

