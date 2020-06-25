# updated phase 1 to handle multiple different options

# script:
# 1. takes original data
# 2. extracts story TRs
# 3. zscores within subject
# 4. loops over all possible left out subjects
#	if k1 > 0, trains SRM and removes common signal
# 5. saves final data matrix

import numpy as np
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
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectFwe
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy import interp

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
import seaborn as sns
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

features = np.int(sys.argv[1])
iter_number = np.int(sys.argv[2]) # if you want to draw from a null distribution

# 1. Take original data
projectDir='/jukebox/norman/amennen/prettymouth/'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD.nii.gz'
fmriprep_dir=projectDir + '/derivatives/fmriprep'
# NEW: only train for story TRS!!!
story_TR_1 = 14
story_TR_2 = 464
# load subject numbers
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
images_concatenated = np.load('aggregate_data.npy')
vox_num, nTR, num_subs = images_concatenated.shape  # Pull out the shape data
print('Participants ', num_subs)
print('Voxels per participant ', vox_num)
print('TRs per participant ', nTR)

# 2. Extract story TRs and zscore
prepared_data = []
all_subjects = np.arange(num_subs) # should be 39 for all subjects
fake_subject_list = shuffle(all_subjects)

for sub in range(num_subs):
    prepared_data.append(images_concatenated[:, story_TR_1:story_TR_2,fake_subject_list[sub]])
for sub in range(num_subs):
	prepared_data[sub] = stats.zscore(prepared_data[sub],axis=1,ddof=1)
	prepared_data[sub] = np.nan_to_num(prepared_data[sub])
	

# 3. Loop through every left out iteration and save values
indTrain = np.arange(n_per_category)
for f1 in np.arange(n_per_category):
	trainingSubjectsInd1 = np.concatenate([indTrain[:f1], indTrain[f1+1:]]) # take all indices but fold
	s1 = np.arange(n_per_category)
	trainingInd = np.array([np.int(s1[j]) for j in trainingSubjectsInd1])
	training_data = []
	for sub in range(n_per_category-1):
		training_data.append(prepared_data[trainingInd[sub]])
	
	for f2 in np.arange(n_per_category):
		trainingSubjectsInd2 = np.concatenate([indTrain[:f2], indTrain[f2+1:]]) # take all indices but fold
		s2 = np.arange(19) + 19
		trainingInd = np.array([np.int(s2[j]) for j in trainingSubjectsInd2])
		for sub in range(n_per_category-1):
			training_data.append(prepared_data[trainingInd[sub]])
			
		print(f1)
		print(f2)

		print('k1 set to %i' %features)
		n_iter = 20
		srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
		print('Fitting SRM, may take a few minutes')
		if features > 0:
			srm.fit(training_data)
			S = srm.s_
		print('SRM has been fit')
		# Fit the SRM data
		# the output matrix srm.s_ will be in features x time points space
		# then project this output matrix onto each subject's voxel space
		nVox = 3757
		nTRs_new = np.shape(training_data[0])[1]
		print('number of new TRS: %i' % nTRs_new)
		nSub = 38 - 2
		aggregated_SRM_removed = np.zeros((nVox,nTRs_new,nSub))
		for s in np.arange(nSub):
			if features > 0:
				w = srm.w_[s]
				signal_srm = w.dot(S) # reconstructed signal
			else:
				signal_srm = 0
			signal_original = training_data[s]
			subtracted_signal = signal_original - signal_srm
			aggregated_SRM_removed[:,:,s] = subtracted_signal
		filename = 'new_group_SRM/NULL_aggregated_SRM_' + str(features) + 'f_removed_no_s1_' + str(f1) + '_no_s2_' + str(f2) + '_rep_' + str(iter_number) + '.npz'
		print('saving to %s' % filename)
		np.savez(filename, aggregated_SRM_removed, prepared_data)
    