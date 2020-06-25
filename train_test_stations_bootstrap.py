# purpose: train all potential spatiotemporal stations and see which ones give highest accuracy
# bootstrap for winning classifier 1000 times

# updated phase 1 AND 2 for saving memory space
# script:
# 1. takes original data
# 2. extracts story TRs
# 3. zscores within subject
# 4. loops over all possible left out subjects
#	if k1 > 0, trains SRM and removes common signal
# 5. saves final data matrix

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


#logging.info(sys.argv)
logging.info(sys.argv)
array_number=np.int(sys.argv[1]) 
n_iterations=np.int(sys.argv[2]) 
lowhigh=0
logging.debug('loading in csv file to see which parameter values to use')
with open('array_combinations_fmriprep2.csv') as csv_file:
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
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD2.nii.gz'
fmriprep_dir=projectDir + '/derivatives/fmriprep'
# NEW: only train for story TRS!!!

# 2. Load station information
logging.debug('Getting station information ready')
# take second version for continuous station # 20, 58 -- we just want to test 2 stations
# 3/20 --changing using different station
#stationsDict = np.load('mega_testing_stations2.npy').item()
#stationsDict = np.load('stations_upper_right_nofilter.npy').item()
#stationsDict = np.load('mega_testing_stations_nofilter.npy').item()
stationsDict = np.load('upper_right_winners_nofilter.npy',allow_pickle=True).item() # updating for thesis plots - 3/39/20

nStations = len(stationsDict)
logging.debug('number of stations found is %i' % nStations)
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
#logging.info('Participants ', num_subs)
#logging.info('Voxels per participant ', vox_num)
#logging.info('TRs per participant ', nTR)'


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
if lowhigh == 1: # low difference
	chosenTRs = np.load('all_low_difference_trs.npy')
	nTR_new = len(chosenTRs)
elif lowhigh == 2:
	chosenTRs = np.load('all_high_difference_trs.npy')
	nTR_new = len(chosenTRs)
elif lowhigh == 0:
	nTR_new = nTR
	chosenTRs = np.arange(nTR)
logging.info('NEW TRs per participant %i' % nTR_new)

accuracy = np.zeros((nStations,n_iterations))


logging.info('starting iteration loop')
for ii in np.arange(n_iterations):
	if ii%10==0:
		logging.info('***************************************')
		logging.info('ITERATION OVER PEOPLE: %i' % ii)
	f1 = random.randint(0,n_per_category-1)
	f2 = random.randint(0,n_per_category-1)
	trainingSubjectsInd1 = np.concatenate([indTrain[:f1], indTrain[f1+1:]]) # take all indices but fold
	s1 = np.arange(n_per_category)
	trainingInd = np.array([np.int(s1[j]) for j in trainingSubjectsInd1])
	trainingInd_BOOTSTRAP = trainingInd[np.random.choice(len(trainingInd),size=len(trainingInd))]
	training_data = []
	for sub in range(n_per_category-1):
		training_data.append(original_data[trainingInd_BOOTSTRAP[sub]])

	trainingSubjectsInd2 = np.concatenate([indTrain[:f2], indTrain[f2+1:]]) # take all indices but fold
	s2 = np.arange(19) + 19
	trainingInd = np.array([np.int(s2[j]) for j in trainingSubjectsInd2])
	trainingInd_BOOTSTRAP = trainingInd[np.random.choice(len(trainingInd),size=len(trainingInd))]
	for sub in range(n_per_category-1):
		training_data.append(original_data[trainingInd_BOOTSTRAP[sub]])
		
	#logging.info('k1 set to %i' %k1)
	logging.info('k1 set to %i' %k1)
	n_iter = 15
	srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k1)
	logging.info('Fitting SRM, may take a few minutes')
	#logging.info('Fitting SRM, may take a few minutes')
	if k1 > 0:
		srm.fit(training_data)
		S = srm.s_
	#logging.info('SRM has been fit')
	logging.info('SRM has been fit')
	# Fit the SRM data
	# the output matrix srm.s_ will be in features x time points space
	# then project this output matrix onto each subject's voxel space
	nVox = vox_num

	nTRs_new = np.shape(training_data[0])[1]
	#logging.info('number of new TRS: %i' % nTRs_new)
	logging.info('number of new TRS: %i' % nTRs_new)
	nSub = 38 - 2
	#aggregated_SRM_removed = np.zeros((nVox,nTRs_new,nSub))
	SRM_removed = []
	for s in np.arange(nSub):
		if k1 > 0:
			w = srm.w_[s]
			signal_srm = w.dot(S) # reconstructed signal
		else:
			signal_srm = 0
		signal_original = training_data[s]
		subtracted_signal = signal_original - signal_srm
		#aggregated_SRM_removed[:,:,s] = subtracted_signal
		SRM_removed.append(subtracted_signal)

	for sub in range(nSub):
		SRM_removed[sub] = stats.zscore(SRM_removed[sub],axis=1,ddof=1)
		SRM_removed[sub] = np.nan_to_num(SRM_removed[sub])
		# SRM removed shape: 36 subjects x nVoxels x 450 TRs

	if removeAvg:
		subject_average = np.mean(SRM_removed,axis=0)
		SRM_removed = SRM_removed - subject_average

	# 5. If k2 > 0, build 2 different SRMs and only keep the shared signal within each group
	training_data_classifier_SRM = np.zeros((nSub,vox_num,nTR_new)) # each time point gets its own s x v classifier

	# SRM 1
	###### if bootstrapping, get fake group 1 and 2 ready! ######

	trainingInd = np.arange(n_per_category-1)
	srm1 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k2)
	training_data = []
	for sub in range(n_per_category-1):
		training_data.append(SRM_removed[trainingInd[sub]])

	if k2 > 0:
		#logging.info('Fitting SRM, may take a few minutes')
		logging.info('Fitting SRM, may take a few minutes')
		srm1.fit(training_data)
		#logging.info('SRM has been fit')
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
			signal_original = training_data[s_ind1]
			signal_transposed = signal_original.T
			training_data_classifier_SRM[s_ind1,:,:] = signal_original

	# SRM 2

	trainingInd = np.arange(n_per_category-1) + n_per_category-1
	#logging.info(trainingInd)
	logging.info(trainingInd)
	srm2 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k2)
	# remake list object for this iteration
	training_data = []
	for sub in range(n_per_category-1):
		training_data.append(SRM_removed[trainingInd[sub]])
	if k2 > 0:
		#logging.info('Fitting SRM, may take a few minutes')
		logging.info('Fitting SRM, may take a few minutes')
		srm2.fit(training_data)
		#logging.info('SRM has been fit')
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
			signal_original = training_data[s_ind2]
			signal_transposed = signal_original.T
			training_data_classifier_SRM[s_ind2+n_per_category-1,:,:] = signal_original

	# 6. Get testing data ready: load original data, take out story TRs, zscore, then remove time points you want
	left_out_subjects = np.array([f1, f2+19])
	test_data = []
	for sub in np.arange(2):
		test_data.append(original_data[left_out_subjects[sub]][:, chosenTRs])

	if removeAvg:
		test_data = test_data - subject_average

	testing_data_classifier_SRM = np.zeros((2,vox_num,nTR_new)) # each time point gets its own s x v classifier

	for sub in np.arange(2):
		subjdata = test_data[sub]
		testing_data_classifier_SRM[sub,:,:] = subjdata
	
	# 7: Build classifier either by TR or spatiotemporal
	# 8. Train and test classifier

	trainingLabels = np.array([paranoidLabel[0:18],cheatingLabel[0:18] ]).flatten()
	testingLabels = np.array([paranoidLabel[f1],cheatingLabel[f2]])
	logging.info(trainingLabels)
	logging.info(testingLabels)
	#stationsToTest = [20,58]
	for stationInd in np.arange(nStations):
		#if stationInd in stationsToTest: # I ran this when I wanted to rerun another iteration with just 2 cases
		clf = SVC(kernel='linear', probability=True)
		this_station_TRs = np.array(stationsDict[stationInd])
		n_station_TRs = len(this_station_TRs)
		training_data_station = training_data_classifier_SRM[:,:,this_station_TRs]
		testing_data_station = testing_data_classifier_SRM[:,:,this_station_TRs]
		training_data_reshaped = np.reshape(training_data_station,(nSub,vox_num*n_station_TRs))
		testing_data_reshaped = np.reshape(testing_data_station,(2,vox_num*n_station_TRs))
		clf.fit(training_data_reshaped,trainingLabels)
		accuracy[stationInd,ii] = clf.score(testing_data_reshaped,testingLabels)

# 9. Save accuracy
# saving as MEGASTATIONDATA2 for stations 20 and 58!
#save_str = 'new_bothphases/MEGASTATIONDATA2' + '_ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_' + str(array_number) + '.npy'
#save_str = 'new_bothphases/UPPER_RIGHT' + '_ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_' + str(array_number) + '.npy'
save_str = 'new_bothphases/THESIS_PLOT_TEST' + '_ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_' + str(array_number) + '.npy'

logging.info('saving new file as %s' % save_str)
np.save(save_str,accuracy)

for stationInd in np.arange(nStations):
	np.array(stationsDict[stationInd])