# plot green eyes results to make classifier
# used with myclone environment
# updated from plot_greenEyes_clf to include station information clcassification too
# assumes logistic = 1 to show the information for the classifier used

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
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy import interp
# params = {'legend.fontsize': 'large',
#           'figure.figsize': (5, 3),
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large'}
font = {'weight': 'normal',
        'size': 22}
plt.rc('axes',linewidth=5)

plt.rc('font', **font)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, GenericUnivariateSelect, SelectKBest, chi2
from sklearn.feature_selection import RFE
import os
import seaborn as sns
import pandas as pd
import csv
from scipy import stats
import glob
import matplotlib.patches as patches

def makeColorPalette(colors):
  # Create an array with the colors you want to use
  # Set your custom color palette
  customPalette = sns.color_palette(colors)
  return customPalette

projectDir='/jukebox/norman/amennen/prettymouth_fmriprep2/'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD2.nii.gz'
fmriprep_dir=projectDir + '/derivatives/fmriprep'
# load subject numbers
subInd = 0
nsub=38
allnames = []
allgroups = []
groupInfo={}
performance_ST = {}
performance_TR = {}
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
logistic = 1
# load all the possible
if not logistic:
    n_options = 156
    all_k1 = np.zeros((n_options))
    all_k2 = np.zeros((n_options))
    all_filters = np.zeros((n_options))
    all_removeAvg = np.zeros((n_options))
    all_masks = np.zeros((n_options))
    with open('array_combinations_fmriprep2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            #print(row)
            if line_count > 0:
                i=np.int(row[0])
                all_k1[i-1]=np.int(row[1])
                all_k2[i-1]=np.int(row[2])
                all_filters[i-1]=np.int(row[3])
                all_removeAvg[i-1]=np.int(row[4])
                all_masks[i-1]=np.int(row[5])
            line_count += 1
else: # get logistic csv file loaded
    # for logistic classifier rows are: k1, k2, remove avg, C
    n_options = 156
    all_k1 = np.zeros((n_options))
    all_k2 = np.zeros((n_options))
    all_removeAvg = np.zeros((n_options))
    all_C = np.zeros((n_options))
    with open('array_combinations_fmriprep2_logistic.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            #print(row)
            if line_count > 0:
                i=np.int(row[0])
                all_k1[i-1]=np.int(row[1])
                all_k2[i-1]=np.int(row[2])
                all_removeAvg[i-1]=np.int(row[3])
                all_C[i-1]=np.float(row[4])
            line_count += 1

# load in vector scoressegment_score_vector = np.load('segment_score_vector.npy')
beliefs_vector = np.load('beliefs_vector.npy')
emotions_vector = np.load('emotions_vector.npy')
intentions_vector = np.load('intentions_vector.npy')
segment_score_vector = np.load('segment_score_vector.npy')

# load station information
stationsDict = np.load('upper_right_winners_nofilter.npy',allow_pickle=True).item()
nStations = len(stationsDict)
nStations = 7 # set to this instead!
# now see how many have finished if exists, plot
lowhigh = 0
boostrapped = 1 # whether or not to randomly sample training data w/ replacement

# now for the ones that have been completed:
# 1. average accuracy over 1000 samples
# 2. average accuracy over all TRs
# 3. plot
# 4. determine which one has the best one

n_iter = 1000
n_completed = 156
nTR = 450
# build dictionaries to get averages for each classifier type completed and take average of both
all_accuracy_data_TR = np.zeros((n_iter,n_completed))
all_accuracy_data_ST = np.zeros((n_iter,n_completed))
performance_TR = np.zeros((n_completed,))
performance_ST = np.zeros((n_completed,))
clf_types = np.array([1,2])
for classifierType in clf_types:
    if classifierType == 1:
        all_accuracy_data = np.zeros((nTR,n_iter,n_completed))
    else:
        all_accuracy_data = np.zeros((n_iter,n_completed))
    for cc in np.arange(n_completed):
        perm = cc + 1
        index = perm - 1
        k1 = np.int(all_k1[index])
        k2 = np.int(all_k2[index])
        if not logistic:
            ROI = np.int(all_masks[index])
            filterType = np.int(all_filters[index])
        else:
            ROI = 1
            C_param = str(all_C[index])
            filterType = 0
        removeAvg = np.int(all_removeAvg[index])
        if not logistic:
            if not boostrapped:
                filename_data = 'new_bothphases/ARRAYDATA' + '_ROI_' + str(ROI) + '_AVGREMOVE_' + str(removeAvg) +  '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_lowhigh_' + str(lowhigh) + '_k1_' + str(k1) + '_k2_' + str(k2) + '_perm_' + str(perm) + '.npy'
            else:
                filename_data = 'new_bothphases/ARRAYDATA_BS_' + '_ROI_' + str(ROI) + '_AVGREMOVE_' + str(removeAvg) +  '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_lowhigh_' + str(lowhigh) + '_k1_' + str(k1) + '_k2_' + str(k2) + '_perm_' + str(perm) + '.npy'
        else:
            filename_data = 'logistic_test/LOGISTIC_ARRAYDATA_BS_no_replace' + '_ROI_' + str(ROI) + 'C_param' + str(C_param) + '_AVGREMOVE_' + str(removeAvg) + '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_lowhigh_' + str(lowhigh) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_' + str(perm) + '.npy'

        all_accuracies = np.load(filename_data) # this is 450 TRs x 1000 examples
        if classifierType == 1:
            all_accuracy_data[:,:,cc] = all_accuracies
            performance_TR[index] = np.mean(all_accuracies)
            all_accuracy_data_TR[:,cc] = np.mean(all_accuracies,axis=0)

        else:
            all_accuracy_data[:,cc] = all_accuracies
            performance_ST[index] = np.mean(all_accuracies)
            all_accuracy_data_ST[:,cc] = all_accuracies
# all_accuracy_data is 450 TRs x 1000 iterations x number of types completed

# SORT ACCURACY BY COMBO
best_combination = np.zeros((n_options,))
for index in np.arange(n_options):
    best_combination[index] = (performance_TR[index] + performance_ST[index])/2

# SORT ACCURACY BY TR
best_TR = np.zeros((n_options,))
for index in np.arange(n_options):
    best_TR[index] = performance_TR[index] 
large_small_ind_TR = np.argsort(best_TR)[::-1]

# SORT ACCURACY BY STATION\TEMPORAL SCORE
best_ST = np.zeros((n_options,))
for index in np.arange(n_options):
    best_ST[index] = performance_ST[index] 
large_small_ind_ST = np.argsort(best_ST)[::-1]

# now get station accuracy 
ROI = 1
removeAvg = 1
filterType = 0
k1 = 0
k2 = 25
C_param = 1.0
fn5r = glob.glob('logistic_test/rtAtten_UPPER_RIGHT_winners_nofilter_no_replace' + '_ROI_' + str(ROI) + 'C_param' + str(C_param) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_*.npz')
station_data5r = np.load(fn5r[0]) # 58 x 1000 
log_accuracy2r = station_data5r['x'] # 9 stations by 1000 iterations

# reload accuracy data
winning_index=118
perm = winning_index+1
classifierType=1
filename_data = 'logistic_test/LOGISTIC_ARRAYDATA_BS_no_replace' + '_ROI_' + str(ROI) + 'C_param' + str(C_param) + '_AVGREMOVE_' + str(removeAvg) + '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_lowhigh_' + str(lowhigh) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_' + str(perm) + '.npy'
accuracy_tr = np.load(filename_data).T
nTR = np.shape(accuracy_tr)[1]
x = np.arange(nTR-3)
y = accuracy_tr[:,3:]
yerr = scipy.stats.sem(y,axis=0)
mean_acc = np.mean(y,axis=0)
all_station_err = scipy.stats.sem(log_accuracy2r,axis=1,ddof=1)
TR_vec = np.tile(x,n_iter)
data = y.flatten()
iter_number = np.repeat(np.arange(n_iter),nTR-3)
matrix = np.concatenate((TR_vec[:,np.newaxis],data[:,np.newaxis],iter_number[:,np.newaxis]),axis=1)
df = pd.DataFrame(data=matrix, columns = ['TR', 'accuracy','iter'])


fig,ax1= plt.subplots(figsize=(25,15))

sns.set_style("white")
sns.despine()
err_band = {}
err_band['alpha'] = 1

#plt.errorbar(np.arange(len(yerr)),mean_acc,yerr=yerr,color='k', label='TR accuracy')
sns.lineplot(x="TR",y="accuracy",color='#636363',data=df,ci=95,err_kws=err_band,label='TR accuracy')
for st in np.arange(nStations):
    this_station_TRs = np.array(stationsDict[st])
    n_station_TRs = len(this_station_TRs)
    this_station_avg = np.mean(log_accuracy2r[st,:])*np.ones((n_station_TRs,))
    # plot with error bars
    # put only one label
    if st == 0:
        plt.errorbar(this_station_TRs-3,this_station_avg,yerr=all_station_err[st],color='r', ecolor='r',alpha=1,label='station accuracy')
    else:
        plt.errorbar(this_station_TRs-3,this_station_avg,yerr=all_station_err[st],color='r', ecolor='r',alpha=1)
plt.plot([0 ,500], [.55, .55], 'k--', lw=6, label='threshold')

plt.legend()
plt.xlim([0,nTR])
plt.ylabel('classification accuracy',fontsize=20)
plt.xlabel('time in story (TR #)',fontsize=20)
ax2 = ax1.twinx()
sns.despine()
ax2.set_ylabel('segment score',color='#3182bd',fontsize=20)
ax2.plot(segment_score_vector[0:-3]/3 , '--', color='#3182bd', label='segment_score',linewidth=5,alpha=0.7)
ax2.set_ylim([-1.5,1.5])
plt.savefig('thesis_plots/station_search_logistic.pdf')
plt.show()

# get correlation of TR accuracy and story vector
r,p = scipy.stats.pearsonr(segment_score_vector[0:-3]/3,mean_acc)
#r = 0.15 p =0.0014

# now get average station accuracy
# get average station accuracy across each station for each iteration
all_station_acc = np.mean(log_accuracy2r,axis=0)

# MAKE INTO A DATAFRAME
large_small_ind = np.argsort(best_combination)[::-1]
array_options = np.arange(n_options)
combo_sorted = best_combination[large_small_ind]
print('the best combo accuracy is %4.4f' % combo_sorted[0])
combo = np.arange(n_options)
all_data = np.concatenate((combo_sorted[:,np.newaxis],combo[:,np.newaxis]),axis=1)
df = pd.DataFrame(data=all_data,columns=['accuracy', 'combo'])

# make a dataframe that includes the iterations to get error bars
mega_data_frame = {}
tr_sorted = all_accuracy_data_TR[:,large_small_ind]
# check that this is right
np.all(np.mean(tr_sorted,axis=0) - performance_TR[large_small_ind] < 0.0000001)
st_sorted = all_accuracy_data_ST[:,large_small_ind]
# now put into dataframe
tr_sorted_vector = tr_sorted.flatten() # goes for first iteration, then all combo options
st_sorted_vector = st_sorted.flatten()
avg_accuracy_sorted_vector = (tr_sorted_vector+st_sorted_vector)/2
iteration_vector = np.repeat(np.arange(n_iter),n_completed)
combo_vector = np.tile(np.arange(n_completed),n_iter)
mega_data_frame['tr_acc'] = tr_sorted_vector
mega_data_frame['st_acc'] = st_sorted_vector
mega_data_frame['combo_acc'] = avg_accuracy_sorted_vector
mega_data_frame['iteration'] = iteration_vector
mega_data_frame['rank'] = combo_vector
df_iter = pd.DataFrame.from_dict(data=mega_data_frame)
# we want to sort each of the datasets in the right order

# PLOT RESULTS - COMBINATION AVERAGE CLASSIFICATION FIRST
winning_index=118






fig,ax= plt.subplots(figsize=(20,15))

ax=plt.subplot(2,1,1)
pl = sns.barplot(data=df_iter,x="rank",y="tr_acc",ci=95)
sns.set_style("white")
sns.despine()
#sns.stripplot(data=df,x="combo",y="accuracy",jitter=True,split=True,hue="datanull",color="k")
#sns.violinplot(data=df,x="combo",y="accuracy",hue="datanull",split=True)
# n_options=156 to show all
n_options_show=51
for i in np.arange(n_options_show):
    perm = array_options[large_small_ind[i]] + 1
    c_index = perm - 1
    k1 = np.int(all_k1[c_index])
    k2 = np.int(all_k2[c_index])
    wasAvgRemove = np.int(all_removeAvg[c_index])
    combo_k1k2 = [k1,k2]
    str_avg = '%2.2f' % combo_sorted[i]
    shift = .05
    ytext=0.8
    if not logistic:
        filterType = np.int(all_filters[c_index])
        ROI = np.int(all_masks[c_index])

        #pl.text(i, total_average_sorted[i]+0.1, str_avg, horizontalalignment='center', size='small', color='black', weight='light')
        pl.text(i, 1, str(ROI), horizontalalignment='center', size='small', color='black',weight='bold')
        pl.text(i, .95, str(wasAvgRemove), horizontalalignment='center', size='small', color='black',weight='light')
        pl.text(i, 0.9 , str(filterType), horizontalalignment='center', size='small', color='black',weight='bold')
        pl.text(i, 0.85, str(k1), horizontalalignment='center', size='small', color='black',weight='light')
        pl.text(i, ytext, str(k2), horizontalalignment='center', size='small', color='black',weight='bold')
    else:
        C = np.float(all_C[c_index])
        pl.text(i, 1, str(C), horizontalalignment='center', fontsize=5, color='black',weight='bold')
        pl.text(i, 0.95, str(wasAvgRemove), horizontalalignment='center', fontsize=5, color='black',weight='light')
        pl.text(i, 0.85, str(k1), horizontalalignment='center', fontsize=5, color='black',weight='light')
        pl.text(i, ytext, str(k2), horizontalalignment='center', fontsize=5, color='black',weight='bold')

if not logistic:
    pl.text(i+.5, 1, 'ROI', horizontalalignment='left', size='small', color='black',weight='bold')
    pl.text(i+.5, 0.9 , 'HP filter', horizontalalignment='left', size='small', color='black',weight='bold')
else:
    pl.text(i+.5, 1, 'C', horizontalalignment='left', size='small', color='black',weight='bold')
pl.text(i+.5, .87, 'AVG removed', horizontalalignment='left', size='small', color='black',weight='light')
pl.text(i+.5, 0.73, 'k1', horizontalalignment='left', size='small', color='black',weight='light')
pl.text(i+.5, ytext, 'k2', horizontalalignment='left', size='small', color='black',weight='bold')
rect = patches.Rectangle((10.5,0.5),1,0.5,linewidth=5,edgecolor='r',facecolor='none')
ax.add_patch(rect)
# chosen for non-logistic - row 97, index = 96 - best TR
# make text labels
plt.plot([-10 ,n_options+5], [.5, .5], 'c--', lw=6)
plt.ylim([0.5,1.03])
plt.xlabel('')
plt.ylabel('TR clf')
plt.title('Mean accuracy over all classifier types')
plt.xlim([-.5,n_options_show-1+.5])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

ax=plt.subplot(2,1,2)
pl = sns.barplot(data=df_iter,x="rank",y="st_acc",ci=95)
sns.set_style("white")
sns.despine()
#sns.stripplot(data=df,x="combo",y="accuracy",jitter=True,split=True,hue="datanull",color="k")
#sns.violinplot(data=df,x="combo",y="accuracy",hue="datanull",split=True)
# n_options=156 to show all
# for i in np.arange(n_options_show):
#     perm = array_options[large_small_ind[i]] + 1
#     c_index = perm - 1
#     k1 = np.int(all_k1[c_index])
#     k2 = np.int(all_k2[c_index])
#     filterType = np.int(all_filters[c_index])
#     wasAvgRemove = np.int(all_removeAvg[c_index])
#     combo_k1k2 = [k1,k2]
#     ROI = np.int(all_masks[c_index])
#     str_avg = '%2.2f' % combo_sorted[i]
#     shift = .05
    # pl.text(i, 1, str(ROI), horizontalalignment='center', size='small', color='black',weight='bold')
    # pl.text(i, .95, str(wasAvgRemove), horizontalalignment='center', size='small', color='black',weight='light')
    # pl.text(i, 0.9 , str(filterType), horizontalalignment='center', size='small', color='black',weight='bold')
    # pl.text(i, 0.85, str(k1), horizontalalignment='center', size='small', color='black',weight='light')
    # ytext=0.8
    # pl.text(i, ytext, str(k2), horizontalalignment='center', size='small', color='black',weight='bold')

plt.plot([-10 ,n_options+5], [.5, .5], 'c--', lw=6)
plt.ylim([0.5,1.03])
plt.xlabel('Classifier parameter combination')
plt.title('')
plt.xlim([-.5,n_options_show-1+.5])
plt.ylabel('spatiotemporal clf')
rect = patches.Rectangle((10.5,0.5),1,0.3,linewidth=5,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.savefig('thesis_plots/classifier_search_test_logistic.pdf')

plt.show()

# new version: just plot all TR and spatiotemporal classification accuracy as plots
winning_index=118
winning_dict = {}
z = np.concatenate((all_accuracy_data_TR[:,winning_index],all_accuracy_data_ST[:,winning_index],all_station_acc),axis=0)
#winning_dict['temporal'] = all_accuracy_data_TR[:,winning_index]
#winning_dict['spatiotemporal'] = all_accuracy_data_ST[:,winning_index]
winning_dict['accuracy'] = z
winning_dict['clf'] = ['TR']*n_iter + ['spatiotemporal'] * n_iter + ['spatiotemporal: \n stations'] * n_iter
df = pd.DataFrame.from_dict(data=winning_dict)
fig,ax= plt.subplots(figsize=(20,15))
COLORS=['#636363','#bdc9e1', 'r'] # grey for TR, blue grey, red
P = makeColorPalette(COLORS)
pl = sns.barplot(data=df,x="clf",y="accuracy",ci=95,palette=P)
plt.ylim([0.5,.8])
sns.set_style("white")
plt.ylabel('accuracy',fontsize=20)
plt.xlabel('type of classifier',fontsize=20)
plt.title('Classifier accuracy',fontsize=30)
sns.despine()
plt.savefig('thesis_plots/winning_clf.pdf')

plt.show()