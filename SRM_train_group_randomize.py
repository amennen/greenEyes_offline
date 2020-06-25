# purpose of script: for given k1 and k2, randomize which two subjects are left out 
# inputs: iter number (for saving), k1, and k2

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
nfolds=19
print(allnames)
filename = 'aggregated_SRM_' + str(k1) + 'f_removed_storyTRz.npy'
images_SRM_removed = np.load(filename)
# this is in shape voxels x TRs x subjects

vox_num, nTR, num_subs = images_SRM_removed.shape  # Pull out the shape data

print('Participants ', num_subs)
print('Voxels per participant ', vox_num)
print('TRs per participant ', nTR)

# make processed data once for training/testing
prepared_data = []
for sub in range(num_subs):
    prepared_data.append(images_SRM_removed[:,:,sub])
for sub in range(num_subs):
    prepared_data[sub] = stats.zscore(prepared_data[sub],axis=1,ddof=1)
    prepared_data[sub] = np.nan_to_num(prepared_data[sub])
    
# this makes prepared data in subjects x voxels x TRs

n_per_category = 19
indTrain = np.arange(n_per_category)
features = k2
n_iter=20
# choose two different subjects to hold out at random--indices 
f1 = random.randint(0,n_per_category-1)
f2 = random.randint(0,n_per_category-1)

trainingSubjectsInd1 = np.concatenate([indTrain[:f1], indTrain[f1+1:]]) # take all indices but fold
s1 = np.arange(n_per_category)
trainingInd = np.array([np.int(s1[j]) for j in trainingSubjectsInd1])
print(trainingInd)
srm1 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
# remake list object for this iteration
training_data = []
for sub in range(n_per_category-1):
	training_data.append(prepared_data[trainingInd[sub]])
print('Fitting SRM, may take a few minutes')
srm1.fit(training_data)
print('SRM has been fit')
# now we want to bring all subjects back into residual space but with ONLY the shared component
# training data for the classifier will be a vector nsub x nvoxels*TR
training_data_classifier_SRM = np.zeros((num_subs-2,vox_num*nTR))
S1 = srm1.s_
for s_ind1 in np.arange(n_per_category-1):
	w = srm1.w_[s_ind1]
	signal_srm = w.dot(S1) # reconstructed shared signal
	signal_transposed = signal_srm.T
	training_data_classifier_SRM[s_ind1,:] = signal_transposed.flatten()
accuracy = np.zeros(1)
# repeat with other condition
trainingSubjectsInd2 = np.concatenate([indTrain[:f2], indTrain[f2+1:]]) # take all indices but fold
s2 = np.arange(19) + 19
trainingInd = np.array([np.int(s2[j]) for j in trainingSubjectsInd2])
print(trainingInd)
srm2 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
# remake list object for this iteration
training_data = []
for sub in range(n_per_category-1):
	training_data.append(prepared_data[trainingInd[sub]])
print('Fitting SRM, may take a few minutes')
srm2.fit(training_data)
print('SRM has been fit')
S2 = srm2.s_
for s_ind2 in np.arange(n_per_category-1):
	w = srm2.w_[s_ind2]
	signal_srm = w.dot(S2) # reconstructed shared signal
	signal_transposed = signal_srm.T
	training_data_classifier_SRM[s_ind2+n_per_category-1,:] = signal_transposed.flatten()

trainingLabels = [paranoidLabel[j] for j in trainingSubjectsInd1]+[cheatingLabel[j] for j in trainingSubjectsInd2]
testingLabels = np.array([paranoidLabel[f1],cheatingLabel[f2]])
print(trainingLabels)
print(testingLabels)
testingInd = np.array([f1,f2+19])
print(testingInd)
testing_data_classifier = np.zeros((2,vox_num*nTR))
for sub in np.arange(len(testingInd)):
	subjdata = prepared_data[testingInd[sub]].T
	testing_data_classifier[sub,:] = subjdata.flatten()
clf = LinearSVC()
clf.fit(training_data_classifier_SRM,trainingLabels)
accuracy[0] = clf.score(testing_data_classifier,testingLabels)


save_str = 'accuracy_SRM_randomized/accuracy_k1_' + str(k1) + '_k2_' + str(k2) + '_rep_' + str(iter_number) + 'storyTRz.npy'
np.save(save_str,accuracy)


