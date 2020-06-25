# purpose of this script: train a classifier from ALL 38 previous subjects based on given classifier parameters
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
classifierType = 1 # 1 by TR, 2 spatiotemporal for all TRs

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
if classifierType == 2: # spatiotemporal
	training_data_classifier_SRM = np.zeros((nSub,vox_num*nTR_new))
elif classifierType == 1:
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
		if classifierType == 2:
			training_data_classifier_SRM[s_ind1,:] = signal_transposed.flatten()
		elif classifierType == 1:
			training_data_classifier_SRM[s_ind1,:,:] = signal_srm
else:
	logging.info('Not fitting SRM -- using original data')
	for s_ind1 in np.arange(n_per_category):
		signal_original = training_data[s_ind1]
		signal_transposed = signal_original.T
		if classifierType == 2:
			training_data_classifier_SRM[s_ind1,:] = signal_transposed.flatten()
		elif classifierType == 1:
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
		if classifierType == 2:
			training_data_classifier_SRM[s_ind2+n_per_category,:] = signal_transposed.flatten()
		elif classifierType == 1:
			training_data_classifier_SRM[s_ind2+n_per_category,:,:] = signal_srm
else:
	logging.info('Not fitting SRM -- using original data')
	for s_ind2 in np.arange(n_per_category):
		signal_original = training_data[s_ind2]
		signal_transposed = signal_original.T
		if classifierType == 2:
			training_data_classifier_SRM[s_ind2+n_per_category,:]= signal_transposed.flatten()
		elif classifierType == 1:
			training_data_classifier_SRM[s_ind2+n_per_category,:,:] = signal_original
# 7: Build classifier either by TR or spatiotemporal
# 8. Train classifier and save
trainingLabels = np.array([paranoidLabel,cheatingLabel ]).flatten()
logging.info(trainingLabels)
if classifierType == 1:
	# each separate classifier has one label per subject which is the same
	if constant_classifier:
		# build classifier from a single TR
		this_data_to_train = training_data_classifier_SRM[:,:,TR_to_use]
		clf = SVC(kernel='linear', probability=True)
		clf.fit(this_data_to_train,trainingLabels)
		filename = 'saved_classifiers' + '/' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2) + '_TRconstant_' + str(TR_to_use) + '.sav'
		pickle.dump(clf,open(filename, 'wb'))
	else:
		for t in np.arange(nTR_new):
			this_data_to_train = training_data_classifier_SRM[:,:,t]
			clf = SVC(kernel='linear', probability=True)
			clf.fit(this_data_to_train,trainingLabels)
			filename = 'saved_classifiers' + '/' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2) + '_TR_' + str(t) + '.sav'
			pickle.dump(clf,open(filename, 'wb'))
elif classifierType == 2:
	clf = SVC(kernel='linear', probability=True)
	filename = 'saved_classifiers' + '/' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_classifierType_' + str(classifierType) + '_filter_' + str(filterType)  + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
	clf.fit(training_data_classifier_SRM,trainingLabels)
	pickle.dump(clf,open(filename,'wb'))

# save subject average used to later do the same thing for that subject
if removeAvg:
	average_signal_fn = 'saved_classifiers' + '/' + 'averageSignal' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2) + '.npy'
	np.save(average_signal_fn, subject_average)
