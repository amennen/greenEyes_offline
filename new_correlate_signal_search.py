# purpose: calculate correlation differences for each held-out comparison
# what this will do:
# get given parameters: filter, k1, k2, mask, removeavg
# if k1 does not equal zero, everyone will be in residual space
# remove shared space component from ALL subjects (but only having trained on the training subjects) --> people are in residual space but it's not circular

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s - %(message)s')
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
logging.info(sys.argv)

array_number=np.int(sys.argv[1]) 
n_iterations=np.int(sys.argv[2]) 
takeAvg = np.int(sys.argv[3])

logging.debug('loading in csv file to see which parameter values to use')
with open('array_combinations_fmriprep2_classifierhunt.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		print(row)
		if line_count > 0:
			i=np.int(row[0])
			if i == array_number:
				k1=np.int(row[1])
				k2=np.int(row[2])
				filterType=np.int(row[3])
				removeAvg=np.int(row[4])
				maskType=np.int(row[5])
		line_count += 1
# 1. Take original data
logging.debug('parsing original data')
projectDir='/jukebox/norman/amennen/prettymouth_fmriprep2/'

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
                    print(group)
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

#logging.debug('Getting station information ready')
#stationsDict = np.load('stations_iter1.npy').item()
#nStations = len(stationsDict)
TRsToSearch = np.arange(378,409+1)
TRsToAvg = np.arange(380,402+1)
#takeAvg = 1 # either take average or search over 40 possible options
logging.debug('starting new work prechecks')

indTrain = np.arange(n_per_category)
nTR_new = nTR
chosenTRs = np.arange(nTR)
logging.info('NEW TRs per participant %i' % nTR_new)

# we're going to output an accuracy for each station for every iteration

logging.info('training big group classifier SRM 1 incase we need it')
n_iter=15
if k1 > 0:
	srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k1)
	srm.fit(original_data)
	S = srm.s_
	logging.info('Group SRM 1 has been fit')
	# now put everyone in residual space before continuing
	nVox = vox_num
	nSub = num_subs
	SRM_removed = []
	for s in np.arange(nSub):
		w = srm.w_[s]
		signal_srm = w.dot(S) # reconstructed signal
		signal_original = original_data[s]
		subtracted_signal = signal_original - signal_srm
		SRM_removed.append(subtracted_signal)
	# everyone is now in residual space
	for sub in range(nSub):
		SRM_removed[sub] = stats.zscore(SRM_removed[sub],axis=1,ddof=1)
		SRM_removed[sub] = np.nan_to_num(SRM_removed[sub])

# now can build training matrix specific to the subjects in the training/testing group

logging.info('starting iteration loop')
nTRs_search = len(TRsToSearch)
if takeAvg:
	predicted_probability_correct = np.zeros((n_iterations,nTR,2))
	predicted_accuracy = np.zeros((n_iterations,nTR,2))
else:
	predicted_probability_correct = np.zeros((n_iterations,nTRs_search,nTR,2))
	predicted_accuracy = np.zeros((n_iterations,nTRs_search,nTR,2))
for ii in np.arange(n_iterations):
	if ii%10==0:
		logging.info('***************************************')
		logging.info('ITERATION OVER PEOPLE: %i' % ii)
	f1 = random.randint(0,n_per_category-1)
	f2 = random.randint(0,n_per_category-1)
	trainingSubjectsInd1 = np.concatenate([indTrain[:f1], indTrain[f1+1:]]) # take all indices but fold
	s1 = np.arange(n_per_category)
	trainingInd = np.array([np.int(s1[j]) for j in trainingSubjectsInd1])
	training_data = []
	for sub in range(n_per_category-1):
		if k1 > 0:
			training_data.append(SRM_removed[trainingInd[sub]])
		else:
			training_data.append(original_data[trainingInd[sub]])

	trainingSubjectsInd2 = np.concatenate([indTrain[:f2], indTrain[f2+1:]]) # take all indices but fold
	s2 = np.arange(19) + 19
	trainingInd = np.array([np.int(s2[j]) for j in trainingSubjectsInd2])
	for sub in range(n_per_category-1):
		if k1 > 0:
			training_data.append(SRM_removed[trainingInd[sub]])
		else:
			training_data.append(original_data[trainingInd[sub]])

	nSub = num_subs - 2
	# now both groups have nSub - 2
	if removeAvg:
		subject_average = np.mean(training_data,axis=0)
		training_data = training_data - subject_average

	training_data_classifier_SRM = np.zeros((nSub,vox_num,nTR_new)) # each time point gets its own s x v classifier

	# now the individual group SRM's - should definitely not include the people you're leaving out
	trainingInd = np.arange(n_per_category-1)
	srm1 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k2)
	training_data_group1 = []
	for sub in range(n_per_category-1):
		training_data_group1.append(training_data[trainingInd[sub]])

	if k2 > 0:
		logging.info('Fitting SRM, may take a few minutes')
		srm1.fit(training_data_group1)
		logging.info('SRM has been fit')
		# now we want to bring all subjects back into residual space but with ONLY the shared component
		# training data for the classifier will be a vector nsub x nvoxels*TR
		S1 = srm1.s_
		for s_ind1 in np.arange(n_per_category-1):
			w = srm1.w_[s_ind1]
			signal_srm = w.dot(S1) # reconstructed shared signal
			signal_transposed = signal_srm.T
			training_data_classifier_SRM[s_ind1,:,:] = signal_srm
	else:
		#logging.info('Not fitting SRM -- using original data')
		logging.info('Not fitting SRM -- using original data')
		for s_ind1 in np.arange(n_per_category-1):
			signal_original = training_data_group1[s_ind1]
			training_data_classifier_SRM[s_ind1,:,:] = signal_original

	# SRM 2

	trainingInd = np.arange(n_per_category-1) + n_per_category-1
	#logging.info(trainingInd)
	logging.info(trainingInd)
	srm2 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k2)
	# remake list object for this iteration
	training_data_group2 = []
	for sub in range(n_per_category-1):
		training_data_group2.append(training_data[trainingInd[sub]])
	if k2 > 0:
		logging.info('Fitting SRM, may take a few minutes')
		srm2.fit(training_data_group2)
		logging.info('SRM has been fit')
		S2 = srm2.s_
		for s_ind2 in np.arange(n_per_category-1):
			w = srm2.w_[s_ind2]
			signal_srm = w.dot(S2) # reconstructed shared signal
			signal_transposed = signal_srm.T
			training_data_classifier_SRM[s_ind2+n_per_category-1,:,:] = signal_srm
	else:
		logging.info('Not fitting SRM -- using original data')
		for s_ind2 in np.arange(n_per_category-1):
			signal_original = training_data_group2[s_ind2]
			training_data_classifier_SRM[s_ind2+n_per_category-1,:,:] = signal_original

	# 6. Get testing data ready: load original data, take out story TRs, zscore, then remove time points you want
	left_out_subjects = np.array([f1, f2+19])
	test_data = []
	for sub in np.arange(2):
		if k1 > 0:
			test_data.append(SRM_removed[left_out_subjects[sub]][:,chosenTRs])
		else:
			test_data.append(original_data[left_out_subjects[sub]][:, chosenTRs])

	if removeAvg:
		test_data = test_data - subject_average
	testing_data_classifier_SRM = np.zeros((2,vox_num,nTR_new)) # each time point gets its own s x v classifier
	for sub in np.arange(2):
		subjdata = test_data[sub]
		testing_data_classifier_SRM[sub,:,:] = subjdata


	trainingLabels = np.array([paranoidLabel[0:18],cheatingLabel[0:18] ]).flatten()
	testingLabels = np.array([paranoidLabel[f1],cheatingLabel[f2]])
	logging.info(trainingLabels)
	logging.info(testingLabels)
	# now build classifier for each TR on its own
	
	if takeAvg:
		# just train one classifier on average signal across those given TRs
		clf = SVC(kernel='linear',probability=True)
		training_data_TR = np.mean(training_data_classifier_SRM[:,:,TRsToAvg],axis=2)
		clf.fit(training_data_TR,trainingLabels)
		testing_data_TR = np.mean(testing_data_classifier_SRM[:,:,TRsToAvg],axis=2)
		# to make comparable, let's just score everything
		predictions_correct_1 = clf.predict(testing_data_classifier_SRM[0,:,:].T) == paranoidLabel[0]
		predictions_correct_2 = clf.predict(testing_data_classifier_SRM[1,:,:].T) == cheatingLabel[0]
		#predicted_probability_correct[ii,:,0] = clf.predict_proba(testing_data_classifier_SRM[0,:,:].T)[:,1]
		#predicted_probability_correct[ii,:,1] = clf.predict_proba(testing_data_classifier_SRM[1,:,:].T)[:,0]
		predicted_accuracy[ii,predictions_correct_1,0] = 1
		predicted_accuracy[ii,predictions_correct_2,1] = 1
	else:
		for tt in np.arange(nTRs_search):
			# build classifier for this TR
			clf = SVC(kernel='linear', probability=True)
			this_TR = TRsToSearch[tt]
			# now go through training and testing data and reshape
			training_data_TR = training_data_classifier_SRM[:,:,this_TR]
			testing_data_TR = testing_data_classifier_SRM[:,:,this_TR]
			# now isolate only relevant TR and build separate training and testing signals
			clf.fit(training_data_TR,trainingLabels)
			# now we want to test on all nTRs total
			#testing_data_classifier_SRM -- the first subject is from the first group; the second subject is from the second group
			# i got the labels wrong!!! so this was actually probability INCORRECT
			np.repeat(paranoidLabel[0],nTR)
			predictions_correct_1 = clf.predict(testing_data_classifier_SRM[0,:,:].T) == paranoidLabel[0]
			predictions_correct_2 = clf.predict(testing_data_classifier_SRM[1,:,:].T) == cheatingLabel[0]
			predicted_accuracy[ii,tt,predictions_correct_1,0] = 1
			predicted_accuracy[ii,tt,predictions_correct_2,1] = 1
			#predicted_probability_correct[ii,tt,:,0] = clf.predict_proba(testing_data_classifier_SRM[0,:,:].T)[:,1]
			#predicted_probability_correct[ii,tt,:,1] = clf.predict_proba(testing_data_classifier_SRM[1,:,:].T)[:,0]
			# next you want to check the accuarcy at each TR for each subject


#########  AFTER FINAL LOOP ##############
#avg_predicted_probabiltiy_correct = np.mean(predicted_probability_correct,axis=0)
#save_str = 'classifier_hunt/PROBA2' + '_ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) +  '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_' + str(array_number) + '.npy'
avg_predicted_accuracy = np.mean(predicted_accuracy,axis=0)
save_str = 'classifier_hunt/accuracy_averagedTRs_' + str(takeAvg) + '_ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) +  '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_' + str(array_number) + '.npy'
logging.info('saving new file as %s' % save_str)
np.save(save_str,avg_predicted_accuracy)