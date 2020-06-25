# purpose: compare real-time processing pipeline with original processing

import numpy as np
import pickle
import nibabel as nib
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
import scipy.stats as sstats  # type: ignore

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

# first get that subject's data ready for preprocessing in the same way
projectDir = '/jukebox/norman/amennen/RT_prettymouth/data/bids/Norman/Mennen/5516_greenEyes/'
fmriprep_dir=projectDir + '/derivatives/fmriprep'
preproc_dir = projectDir + '/derivatives/preproc'
TOM_large = '/jukebox/norman/amennen/prettymouth_fmriprep2/ROI/TOM_large_resampled_maskedbybrain.nii.gz'
TOM_cluster = '/jukebox/norman/amennen/prettymouth_fmriprep2/ROI/TOM_cluster_resampled_maskedbybrain.nii.gz'
fmriprep2_path = '/jukebox/norman/amennen/prettymouth_fmriprep2/code'
offline_path = '/jukebox/norman/amennen/prettymouth_fmriprep2/code/saved_classifiers'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD2.nii.gz'

maskType = 1
removeAvg = 1
filterType = 0
k1 = 0
k2 = 25

allSubjects = np.array([101,102])
nSub = len(allSubjects)
story_TR_1 = 14
story_TR_2 = 464
run_TRs = 450
nRuns = 4 # for subject 101 nRuns = 3, for subject 102 nRuns = 4


zscore_data = 1
if maskType == 0:
    nVoxels = 3757
elif maskType == 1:
    nVoxels = 2414
allData = np.zeros((nVoxels,run_TRs,nRuns,nSub))
originalData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))
zscoredData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))
removedAvgData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))

# get all subject data first
for s in np.arange(nSub):
    subjectNum = allSubjects[s]
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(1)
    data_dir = fmriprep_dir + '/' + bids_id + '/' + ses_id + '/' + 'func'

    if subjectNum == 101:
        nRuns = 3
    else:
        nRuns = 4
    
    for r in np.arange(nRuns):
        subjectRun = r + 1
        run_id = 'run-{0:02d}'.format(subjectRun)
        subjData = data_dir + '/' + bids_id + '_' + ses_id + '_' + 'task-story' + '_' + run_id + '_' + 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        print(subjData)
        if maskType == 0:
            masked_data = apply_mask(subjData,DMNmask)
        elif maskType == 1:
            masked_data = apply_mask(subjData,TOM_large)
        elif maskType == 2:
            masked_data = apply_mask(subjData,TOM_cluster)
        masked_data_remove10 = masked_data[10:,:]
        originalData_all[:,:,r,s] = masked_data_remove10[story_TR_1:story_TR_2,:].T # now should be voxels x TR

        # B. zscore within that subject across all time points
        if zscore_data:
            originalData_all[:,:,r,s] = stats.zscore(originalData_all[:,:,r,s],axis=1,ddof = 1)
            originalData_all[:,:,r,s] = np.nan_to_num(originalData_all[:,:,r,s])
            zscoredData_all[:,:,r,s] = originalData_all[:,:,r,s]
        # C. remove average signal
        if removeAvg:
            average_signal_fn = offline_path + '/' +  'averageSignal' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2) + '.npy'
            average_signal = np.load(average_signal_fn)
            #average_signal = 0
            SRM_removed = originalData_all[:,:,r,s] - average_signal
            removedAvgData_all[:,:,r,s] = SRM_removed
        else:
            removedAvgData_all[:,:,r,s] = originalData_all[:,:,r,s]
        allData[:,:,r,s] = removedAvgData_all[:,:,r,s]

stationsDict = np.load('/jukebox/norman/amennen/prettymouth_fmriprep2/code/upper_right_winners_nofilter.npy',allow_pickle=True).item()
nStations = len(stationsDict)

nTRs = run_TRs
nRuns = 4
correct_prob = np.zeros((nStations,nRuns,nSub))
# load spatiotemporal pattern and test for each separate station
runOrder = {}
runOrder[0] = ['P', 'P', 'C']
runOrder[1] = ['C', 'C', 'P', 'P']
for s in np.arange(nSub):
    subjectNumber = allSubjects[s]
    print(subjectNumber)
    if subjectNumber == 101:
        nRuns = 3
    else:
        nRuns = 4
    for r in np.arange(nRuns):
        for st in np.arange(nStations):
            stationInd = st
            filename = offline_path + '/' + 'UPPERRIGHT_stationInd_' + str(stationInd) + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'

            loaded_model = pickle.load(open(filename, 'rb'))
            # test whole spatiotemporal pattern
            this_station_TRs = np.array(stationsDict[stationInd])
            print(this_station_TRs)
            print('***')
            n_station_TRs = len(this_station_TRs)
            testing_data = allData[:,this_station_TRs,r,s]
            testing_data_reshaped = np.reshape(testing_data,(1,nVoxels*n_station_TRs))
            cheating_probability = loaded_model.predict_proba(testing_data_reshaped)[0][1]
            if runOrder[s][r] == 'C':
                correct_prob[st,r,s] = cheating_probability
            elif runOrder[s][r] == 'P':
                correct_prob[st,r,s] = 1 - cheating_probability
        print(r,s)


# now load the same information

cmap=plt.get_cmap('cool')
colors=cmap(np.linspace(0,1,nStations))
brainiak_path='/jukebox/norman/amennen/github/brainiak/rt-cloud/projects/greenEyes/data/'
# now load newestfile
sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
from rtCommon.utils import findNewestFile, loadMatFile
s = 0
subject_path = brainiak_path + 'sub-' + str(allSubjects[s]) + '/' + 'ses-02' + '/'
run = 1
filePattern = 'patternsData_r{}*'.format(run)
fn = findNewestFile(subject_path,filePattern)
test_data = loadMatFile(fn)
test_prob = test_data.correct_prob

x = correct_prob[:,run-1,s]
y = test_prob[0,:]

corr = sstats.pearsonr(x,y)[0]
plt.figure(figsize=(10,10))
for st in np.arange(nStations):
    plt.plot(x[st],y[st], '.', ms=20, color=colors[st],label=st)
plt.plot([0,1],[0,1], '--', color='r', lw=3)
plt.title('Subj %i, Run %i, Total corr = %3.3f' % (allSubjects[s],run,corr))
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Offline prediction')
plt.legend()
plt.ylabel('Brainiak RT prediction')
plt.show()