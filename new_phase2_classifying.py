# updated phase 2 to handle multiple different options

# script
# 1. Randomly choose 2 subjects to leave out
# 2. Load that aggregated data matrix from phase 1
# 3. Zscore within subject
# 4. Extract the time points that you want--> plot it here to check*** should z score again or no?
# 5. If k2 > 0, build 2 different SRMs and only keep the shared signal within each group
# 6. Get testing data ready: load original data, take out story TRs, zscore, then remove time points you want
# 7: Build classifier either by TR or spatiotemporal
# by TR: each TR is a separate classifier --> [nsub # nvox] for each TR
# 8. Train and test classifier
# 9. Save accuracy

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
from sklearn.utils import shuffle
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
import random
from datetime import datetime
random.seed(datetime.now())

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

projectDir='/jukebox/norman/amennen/prettymouth/'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD.nii.gz'
fmriprep_dir=projectDir + '/derivatives/fmriprep'

print(sys.argv)
iter_number = np.int(sys.argv[1])
k1 = np.int(sys.argv[2])
k2 = np.int(sys.argv[3])
lowhigh = np.int(sys.argv[4]) # whether to use low (1), high (2), or all (0)
classifierType = np.int(sys.argv[5]) # whether you want (1) temporal or (2) spatiotemporal classification
# load subject numbers
subInd = 0
nsub=38
allnames = []
allgroups = []
groupInfo={}
# NEW: only train on story TRs!!
story_TR_1 = 14
story_TR_2 = 464
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
print(allnames)

# 1. Randomly choose 2 subjects to leave out
f1 = random.randint(0,n_per_category-1)
f2 = random.randint(0,n_per_category-1)

# 2. Load that aggregated data matrix from phase 1
filename = 'new_group_SRM/aggregated_SRM_' + str(k1) + 'f_removed_no_s1_' + str(f1) + '_no_s2_' + str(f2) + '.npy'
images_SRM_removed = np.load(filename)
vox_num, nTR, num_subs = images_SRM_removed.shape  # Pull out the shape data
# now this will be all subjects - 2
print('Participants ', num_subs)
print('Voxels per participant ', vox_num)
print('TRs per participant ', nTR)
# now it should be nsubjects - 2 already

# 3. Zscore within subject
all_data = []
for sub in range(num_subs):
    all_data.append(images_SRM_removed[:, :, sub])  #
for sub in range(num_subs):
    all_data[sub] = stats.zscore(all_data[sub],axis=1,ddof=1)
    all_data[sub] = np.nan_to_num(all_data[sub])

# 4. Extract the time points that you want--> plot it here to check*** should z score again or no?
if lowhigh == 1: # low difference
	chosenTRs = np.load('all_low_difference_trs.npy')
	nTR_new = len(chosenTRs)
elif lowhigh == 2:
	chosenTRs = np.load('all_high_difference_trs.npy')
	nTR_new = len(chosenTRs)
elif lowhigh == 0:
	nTR_new = nTR
	chosenTRs = np.arange(nTR)
print('NEW TRs per participant ', nTR_new)

prepared_data = []
for sub in range(num_subs):
    prepared_data.append(all_data[sub][:,chosenTRs])

# 5. If k2 > 0, build 2 different SRMs and only keep the shared signal within each group
n_per_category = 19
indTrain = np.arange(n_per_category)
features = k2
n_iter=20
if classifierType == 2: # spatiotemporal
	training_data_classifier_SRM = np.zeros((num_subs,vox_num*nTR_new))
elif classifierType == 1:
	training_data_classifier_SRM = np.zeros((num_subs,vox_num,nTR_new)) # each time point gets its own s x v classifier

# SRM 1
###### if bootstrapping, get fake group 1 and 2 ready! ######

trainingInd = np.arange(n_per_category-1)
srm1 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
training_data = []
for sub in range(n_per_category-1):
	training_data.append(prepared_data[trainingInd[sub]])

if k2 > 0:
	print('Fitting SRM, may take a few minutes')
	srm1.fit(training_data)
	print('SRM has been fit')
	# now we want to bring all subjects back into residual space but with ONLY the shared component
	# training data for the classifier will be a vector nsub x nvoxels*TR
	S1 = srm1.s_
	for s_ind1 in np.arange(n_per_category-1):
		w = srm1.w_[s_ind1]
		signal_srm = w.dot(S1) # reconstructed shared signal
		signal_transposed = signal_srm.T
		if classifierType == 2:
			training_data_classifier_SRM[s_ind1,:] = signal_transposed.flatten()
		elif classifierType == 1:
			training_data_classifier_SRM[s_ind1,:,:] = signal_srm
else:
	print('Not fitting SRM -- using original data')
	for s_ind1 in np.arange(n_per_category-1):
		signal_original = training_data[s_ind1]
		signal_transposed = signal_original.T
		if classifierType == 2:
			training_data_classifier_SRM[s_ind1,:] = signal_transposed.flatten()
		elif classifierType == 1:
			training_data_classifier_SRM[s_ind1,:,:] = signal_original

# SRM 2

trainingInd = np.arange(n_per_category-1) + n_per_category-1
print(trainingInd)
srm2 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
# remake list object for this iteration
training_data = []
for sub in range(n_per_category-1):
	training_data.append(prepared_data[trainingInd[sub]])
if k2 > 0:
	print('Fitting SRM, may take a few minutes')
	srm2.fit(training_data)
	print('SRM has been fit')
	S2 = srm2.s_
	for s_ind2 in np.arange(n_per_category-1):
		w = srm2.w_[s_ind2]
		signal_srm = w.dot(S2) # reconstructed shared signal
		signal_transposed = signal_srm.T
		if classifierType == 2:
			training_data_classifier_SRM[s_ind2+n_per_category-1,:] = signal_transposed.flatten()
		elif classifierType == 1:
			training_data_classifier_SRM[s_ind2+n_per_category-1,:,:] = signal_srm
else:
	print('Not fitting SRM -- using original data')
	for s_ind2 in np.arange(n_per_category-1):
		signal_original = training_data[s_ind2]
		signal_transposed = signal_original.T
		if classifierType == 2:
			training_data_classifier_SRM[s_ind2+n_per_category-1,:]= signal_transposed.flatten()
		elif classifierType == 1:
			training_data_classifier_SRM[s_ind2+n_per_category-1,:,:] = signal_original

# 6. Get testing data ready: load original data, take out story TRs, zscore, then remove time points you want
images_concatenated = np.load('aggregate_data.npy')
left_out_subjects = np.array([f1, f2+19])
original_data = []
for sub in np.arange(2):
	original_data.append(images_concatenated[:,story_TR_1:story_TR_2,left_out_subjects[sub]])
for sub in np.arange(2):
	original_data[sub] = stats.zscore(original_data[sub],axis=1,ddof=1)
	original_data[sub] = np.nan_to_num(original_data[sub])
test_data = []
for sub in np.arange(2):
	test_data.append(original_data[sub][:, chosenTRs])

if classifierType == 2: # spatiotemporal
	testing_data_classifier_SRM = np.zeros((2,vox_num*nTR_new))
elif classifierType == 1:
	testing_data_classifier_SRM = np.zeros((2,vox_num,nTR_new)) # each time point gets its own s x v classifier

for sub in np.arange(2):
	subjdata = test_data[sub]
	if classifierType == 2:
		testing_data_classifier_SRM[sub,:] = subjdata.T.flatten()
	elif classifierType == 1:
		testing_data_classifier_SRM[sub,:,:] = subjdata
	
# 7: Build classifier either by TR or spatiotemporal
# 8. Train and test classifier

trainingLabels = np.array([paranoidLabel[0:18],cheatingLabel[0:18] ]).flatten()
testingLabels = np.array([paranoidLabel[f1],cheatingLabel[f2]])
print(trainingLabels)
print(testingLabels)
if classifierType == 1:
	# each separate classifier has one label per subject which is the same
	accuracy = np.zeros((nTR_new,))
	for t in np.arange(nTR_new):
		this_data_to_train = training_data_classifier_SRM[:,:,t]
		clf = LinearSVC()
		clf.fit(this_data_to_train,trainingLabels)
		this_data_to_test = testing_data_classifier_SRM[:,:,t]
		accuracy[t] = clf.score(this_data_to_test,testingLabels)
elif classifierType == 2:
	accuracy = np.zeros(1)
	clf = LinearSVC()
	clf.fit(training_data_classifier_SRM,trainingLabels)
	accuracy[0] = clf.score(testing_data_classifier_SRM,testingLabels)

# 9. Save accuracy
save_str = 'new_accuracy_phase2/classifierType_' + str(classifierType) + '_lowhigh_' + str(lowhigh) + '_k1_' + str(k1) + '_k2_' + str(k2) + '_rep_' + str(iter_number) + '.npy'
print('saving new file as %s' % save_str)
np.save(save_str,accuracy)
