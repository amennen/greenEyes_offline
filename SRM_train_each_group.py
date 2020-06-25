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

projectDir='/jukebox/norman/amennen/prettymouth/'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD.nii.gz'
fmriprep_dir=projectDir + '/derivatives/fmriprep'

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
nfolds=19
print(allnames)

images_SRM_removed = np.load('aggregate_SRM_removed.npy')

vox_num, nTR, num_subs = images_SRM_removed.shape  # Pull out the shape data

print('Participants ', num_subs)
print('Voxels per participant ', vox_num)
print('TRs per participant ', nTR)

# make matrix and z score once--check that this works
train_data = []
for sub in range(num_subs):
    train_data.append(images_SRM_removed[:,:,sub])
for sub in range(num_subs):
    train_data[sub] = stats.zscore(train_data[sub],axis=1,ddof=1)
    train_data[sub] = np.nan_to_num(train_data[sub])

# now go through each category and train SRM on N subjects - 1
# FIRST: paranoid category
n_category = 19
nfolds = n_category
indTrain = np.arange(n_paranoid)
f=0
trainingSubjectsInd = np.concatenate([indTrain[:f],indTrain[f+1:]])
s1 = np.arange(n_category)
trainingInd = [s1[j] for j in trainingSubjectsInd]
# now train SRM on those indices
features = 100
n_iter = 20
srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
print('Fitting SRM, may take a few minutes')
srm.fit(train_data[trainingInd,:,:])
print('SRM has been fit')
# NOW transfer all the data back into voxel space

S = srm.s_
nVox = 3757
nTRs = 475
group_2_SRM = np.zeros((nVox,nTRs,n_category))
for s in np.arange(n_category):
    w = srm.w_[s]
    signal_srm = w.dot(S) # reconstructed signal
    group_2_SRM[:,:,s] = signal_srm
# now train the classifier on that data (same classifier) and then test on the residual space of left out subject



# SECOND: cheating category
trainingSubjectsInd = np.concatenate([indTrain[:f],indTrain[f+1:]])
s2 = np.arange(n_category) + n_category
trainingInd = [s2[j] for j in trainingSubjectsInd]



features = 10
n_iter = 20
srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
print('Fitting SRM, may take a few minutes')
srm.fit(train_data)
print('SRM has been fit')
# Fit the SRM data
# the output matrix srm.s_ will be in features x time points space
# then project this output matrix onto each subject's voxel space
S = srm.s_
nVox = 3757
nTRs = 475
nSub = 38
aggregated_SRM_removed = np.zeros((nVox,nTRs,nSub))
for s in np.arange(num_subs):
    w = srm.w_[s]
    signal_srm = w.dot(S) # reconstructed signal
    signal_original = train_data[s]
    subtracted_signal = signal_original - signal_srm
    aggregated_SRM_removed[:,:,s] = subtracted_signal

np.save('aggregated_SRM_removed.npy', aggregated_SRM_removed)




