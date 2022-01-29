# plot green eyes results to make classifier
# used with myclone environment
# THIS TIME: DO SVM AND NO REPLACEMENT
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
# font = {'weight': 'normal',
#         'size': 22}
# plt.rc('font', **font)
font = {'size': 22,
        'weight': 'normal'}
plt.rc('axes', linewidth=3)
plt.rc('xtick.major', size=0, width = 3)
plt.rc('ytick.major', size=0, width = 3)
matplotlib.rc('font',**font)
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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

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
# skip subjects 039 and 116 because they didn't have the same start times
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

# load all possible parameter combinations
# these are saved as CSV files -- the first one was made for the original SVM
# classifier. the second CSV was made once I was testing combinations
# only for logistic classification
# here, we're going back to SVM because we based these bootstrap results in 
# desinging experiment 1
logistic = 0
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


# classification parameters
lowhigh = 0 # this was from previous testing when I split up the TRs into different training and testing data
bootstrapped = 1 # whether or not to randomly sample training data w/ replacement
n_iter = 1000 # number of bootstrap loops
n_completed = 156
nTR = 450
# build dictionaries to get averages for each classifier type completed and take average of both
all_accuracy_data_TR = np.zeros((n_iter,n_completed)) # for TR classifier type
all_accuracy_data_ST = np.zeros((n_iter,n_completed)) # for spatiotemporal classifier type
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
            if not bootstrapped:
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
# all_accuracy_data is 450 TRs x 1000 iterations x number of parameter options

# combo accuracy is the average TR accuracy and spatiotemporal accuracy
# calculate the average of each accuracy type
best_combination = np.zeros((n_options,))
for index in np.arange(n_options):
    best_combination[index] = (performance_TR[index] + performance_ST[index])/2

# sort combo accuracy from best to worst
### change to sorting by TR accuracy only
large_small_ind = np.argsort(performance_TR)[::-1]
array_options = np.arange(n_options)
combo_sorted = best_combination[large_small_ind]
print('the best combo accuracy is %4.4f' % combo_sorted[0])


# make a dataframe that includes the iterations to get error bars
mega_data_frame = {}
tr_sorted = all_accuracy_data_TR[:,large_small_ind]
st_sorted = all_accuracy_data_ST[:,large_small_ind]
# check to make sure the sorting by averaging over all TRs was the same as after averaging
# np.all(np.mean(tr_sorted,axis=0) - performance_TR[large_small_ind] < 0.0000001)
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


# (1) create a plot that plots the mean accuracy over top 20 average classifier performances
# n_options=156 if we wanted to plot all options
n_options_show=20 # show only the top 20 performing options
# *** the chosen classifier for non-logistic - row 97, index = 96 - best TR
fig,ax= plt.subplots(figsize=(20,15))
# in the top subplot, we plot sorted TR classifier accuracy
ax=plt.subplot(1,1,1)
pl = sns.barplot(data=df_iter,x="rank",y="tr_acc",ci=95)
sns.set_style("white")
sns.despine()
# all of the pl.text commands below are to show what the parameter values are for that classifier
for i in np.arange(n_options_show):
    perm = array_options[large_small_ind[i]] + 1
    c_index = perm - 1
    k1 = np.int(all_k1[c_index])
    k2 = np.int(all_k2[c_index])
    wasAvgRemove = np.int(all_removeAvg[c_index])
    combo_k1k2 = [k1,k2]
    str_avg = '%2.2f' % combo_sorted[i]
    shift = .05
    ytext=0.58
    if not logistic:
        filterType = np.int(all_filters[c_index])
        ROI = np.int(all_masks[c_index])
        #pl.text(i, total_average_sorted[i]+0.1, str_avg, horizontalalignment='center', size='small', color='black', weight='light')
        pl.text(i, ytext+0.1, str(ROI), horizontalalignment='center', verticalalignment='center', size='small', color='black',weight='bold')
        pl.text(i, ytext+.075, str(wasAvgRemove), horizontalalignment='center',  verticalalignment='center',size='small', color='black',weight='light')
        pl.text(i, ytext+0.05 , str(filterType), horizontalalignment='center',  verticalalignment='center',size='small', color='black',weight='bold')
        pl.text(i, ytext+0.025, str(k1), horizontalalignment='center', size='small',  verticalalignment='center',color='black',weight='light')
        pl.text(i, ytext, str(k2), horizontalalignment='center', size='small',  verticalalignment='center',color='black',weight='bold')
    else:
        C = np.float(all_C[c_index])
        pl.text(i, ytext+0.2, str(C), horizontalalignment='center', fontsize=5, verticalalignment='center', color='black',weight='bold')
        pl.text(i, ytext+0.15, str(wasAvgRemove), horizontalalignment='center',  verticalalignment='center',fontsize=5, color='black',weight='light')
        pl.text(i, ytext+0.1, str(k1), horizontalalignment='center', fontsize=5,  verticalalignment='center',color='black',weight='light')
        pl.text(i, ytext, str(k2), horizontalalignment='center', fontsize=5,  verticalalignment='center',color='black',weight='bold')
if not logistic:
    pl.text(i+.5, ytext+0.1, 'ROI', horizontalalignment='left', fontsize=17,  verticalalignment='center', color='black',weight='bold')
    pl.text(i+.5, ytext+0.05 , 'HP filter', horizontalalignment='left',  fontsize=17,  verticalalignment='center', color='black',weight='bold')
else:
    pl.text(i+.5, ytext+0.2, 'C', horizontalalignment='left', fontsize=17,  verticalalignment='center',color='black',weight='bold')
pl.text(i+.5, ytext+0.075, 'mean removed', horizontalalignment='left',fontsize=17, verticalalignment='center', color='black',weight='light')
pl.text(i+.5, ytext+0.025, 'k1', horizontalalignment='left',fontsize=17,  verticalalignment='center',color='black',weight='light')
pl.text(i+.5, ytext, 'k2', horizontalalignment='left', fontsize=17,  verticalalignment='center',color='black',weight='bold')
# this plots a red rectangle around index 11 (from 10.5 - 11.5, width = 1)
rect = patches.Rectangle((-0.5,0.5),1,0.2,linewidth=5,edgecolor='r',facecolor='none')
ax.add_patch(rect)

plt.ylim([0.5,0.7])
plt.yticks([0.5, 0.55, 0.6],fontsize=20,weight='normal')
#plt.xticks([])
#plt.xlabel('')
plt.ylabel('Average accuracy',fontsize=25,weight='normal')
plt.title('Mean accuracy over all TRs',fontsize=30)
plt.xlim([-.5,n_options_show-1+.5])

# in the bottom subplot we plot the spatiotemporal classifier performance
#### COMMENTING OUT FROM HERE
#ax=plt.subplot(2,1,2)
#pl = sns.barplot(data=df_iter,x="rank",y="st_acc",ci=95)
#sns.set_style("white")
#sns.despine()
#plt.plot([-10 ,n_options+5], [.5, .5], 'c--', lw=6)
#plt.ylim([0.5,0.8])
#plt.yticks([0.5,0.75],fontsize=20)
plt.xlabel('rank: average accuracy',fontsize=25)
#plt.title('')
plt.xlim([-.6,n_options_show-1+.6])
plt.xticks(fontsize=20)
#plt.ylabel('spatiotemporal clf',fontsize=25)
# here's where we add the rectangle in the same spot
#rect = patches.Rectangle((10.5,0.5),1,0.3,linewidth=5,edgecolor='r',facecolor='none')
#ax.add_patch(rect)
plt.savefig('thesis_plots_checked/classifier_search_test_SVM_replacement_TR_only.pdf')
#plt.show()


# next, we plot how this specific classifier does for TR and station classifiers

# these are the parameters for the chosen classifier
maskType = 1
removeAvg = 1
filterType = 0
k1 = 0
k2 = 25
array_number = 97
# first load in the station classifier performance (made with train_test_stations_bootstrap)
fn5r = glob.glob('new_bothphases/THESIS_PLOT_TEST' + '_ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_perm_*.npy')
station_acc = np.load(fn5r[0]) # 9 stations x 1000 bootstrap iterations
all_station_acc = np.mean(station_acc,axis=0) # averge over all stations first
print('shape of station performance')
print(np.shape(station_acc))
# next combine all data for this classifier to plot average accuracy
winning_index=96
winning_dict = {}
z = np.concatenate((all_accuracy_data_TR[:,winning_index],all_accuracy_data_ST[:,winning_index],all_station_acc),axis=0)
winning_dict['accuracy'] = z
winning_dict['clf'] = ['TR']*n_iter + ['spatiotemporal'] * n_iter + ['spatiotemporal:\n stations'] * n_iter
df = pd.DataFrame.from_dict(data=winning_dict)
COLORS=['#636363','#bdc9e1', 'r'] # grey for TR, blue grey, red
P = makeColorPalette(COLORS)

# (2) plot average accuracy across the three classifier types
fig,ax= plt.subplots(figsize=(12,9))
pl = sns.barplot(data=df,x="clf",y="accuracy",ci=95,palette=P)
plt.ylim([0.5,.75])
sns.set_style("white")
plt.ylabel('accuracy',fontsize=25)
plt.xlabel('classifier type',fontsize=25)
plt.title('Classifier accuracy',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
sns.despine()
plt.savefig('thesis_plots_checked/winning_clf_SVM.pdf')
print('PRINTING ACCURACY RESULTS')
print('TR')
print(np.mean(all_accuracy_data_TR[:,winning_index]))
print(np.std(all_accuracy_data_TR[:,winning_index]))
print('spatiotemporal')
print(np.mean(all_accuracy_data_ST[:,winning_index]))
print(np.std(all_accuracy_data_ST[:,winning_index]))
print('sp - station')
print(np.mean(all_station_acc))
print(np.std(all_station_acc))
#plt.show()

# now plot each classifier performance over different points in the story

# first, load in beliefs at each point in the story
beliefs_vector = np.load('beliefs_vector.npy')
emotions_vector = np.load('emotions_vector.npy')
intentions_vector = np.load('intentions_vector.npy')
segment_score_vector = np.load('segment_score_vector.npy')

# load station information
stationsDict = np.load('upper_right_winners_nofilter.npy',allow_pickle=True).item()
nStations = len(stationsDict)


# load TR accuracy - this get each TR
perm = winning_index + 1
classifierType = 1
filename_data = glob.glob('new_bothphases/ARRAYDATA_BS_' + '_ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) +  '_classifierType_' + str(classifierType) + '_filter_' + str(filterType) + '_lowhigh_' + str(lowhigh) + '_k1_' + str(k1) + '_k2_' + str(k2) + '_perm_*.npy')
accuracy_tr = np.load(filename_data[0]).T
nTR = np.shape(accuracy_tr)[1]
x = np.arange(nTR-3)
y = accuracy_tr[:,3:]
yerr = scipy.stats.sem(y,axis=0, nan_policy='omit')
mean_acc = np.mean(y,axis=0)
TR_vec = np.tile(x,n_iter)
data = y.flatten()
iter_number = np.repeat(np.arange(n_iter),nTR-3)
matrix = np.concatenate((TR_vec[:,np.newaxis],data[:,np.newaxis],iter_number[:,np.newaxis]),axis=1)
df = pd.DataFrame(data=matrix, columns = ['TR', 'accuracy','iter'])
# also define station error bars
all_station_err = scipy.stats.sem(station_acc,axis=1,ddof=1,nan_policy='omit')

# (3) make large plot with TR, segment scores, and station classifier information
fig,ax1= plt.subplots(figsize=(17,10))
sns.set_style("white")
sns.despine(right=False)
err_band = {}
err_band['alpha'] = 0.45 # this controls the opacity of the error bars of TR classifier (closer to 1 --> darker)
lw = 3
# this plots the TR accuracy
g = sns.lineplot(x="TR",y="accuracy",color='k', data=df,ci=95,err_kws=err_band, label='TR accuracy', **{'linewidth': lw})
#plt.setp(g.artists, linewidth=lw)
# why is 9,10 now a label?
# next, plot the average accuracy for each station
for st in np.arange(nStations):
    this_station_TRs = np.array(stationsDict[st])
    n_station_TRs = len(this_station_TRs)
    this_station_acc = station_acc[st,:]
    m,l,h = mean_confidence_interval(this_station_acc, confidence=0.95)
    this_station_avg = m*np.ones((n_station_TRs,))
    z = np.array([l,h])
    z = z[:,np.newaxis]
    alle = np.tile(np.array([l,h]),(n_station_TRs,1)) 
    # plot with error bars
    # put only one label
    if st == 0:
        plt.errorbar(this_station_TRs-3,this_station_avg,yerr=m-l,color='r', ecolor='r',alpha=1,label='station accuracy')
    else:
        plt.errorbar(this_station_TRs-3,this_station_avg,yerr=m-l,color='r', ecolor='r',alpha=1)
plt.plot([0 ,500], [.55, .55], 'k--', lw=6, label='accuracy threshold')
plt.legend(loc=2)
plt.xlim([0,nTR])
plt.ylabel('classification accuracy',fontsize=25)
plt.xlabel('time in story (TR #)',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Classification accuracy over time',fontsize=30)
# plot the segment scores on a separate axis
ax2 = ax1.twinx()
sns.despine(right=False)
ax2.set_ylabel('segment score',color='#3182bd',fontsize=25)
# shift the segment scores to plot because of HRF lag
ax2.plot(segment_score_vector[0:-3]/3 , '--', color='#3182bd', label='segment_score',linewidth=3,alpha=0.7)
plt.yticks([-1.5,0,1.5],fontsize=20)
plt.legend(loc=4)
ax2.set_ylim([-2,2])
plt.savefig('thesis_plots_checked/station_search_SVM.pdf')
#plt.show()

r,p = scipy.stats.pearsonr(segment_score_vector[0:-3]/3,mean_acc)
print('********')
print('Pearson r TR accuracy and segment scores : ',(r,p))

#### things not used/redudant
# load in vector scores segment_score_vector = np.load('segment_score_vector.npy')
# beliefs_vector = np.load('beliefs_vector.npy')
# emotions_vector = np.load('emotions_vector.npy')
# intentions_vector = np.load('intentions_vector.npy')
# load station information
# stationsDict3 = np.load('upper_right_winners_nofilter.npy',allow_pickle=True).item()
# nStations3 = len(stationsDict3)

# now we make a data frame and save results to check
# combo = np.arange(n_options)
# perm = array_options[large_small_ind]
# all_data = np.concatenate((combo_sorted[:,np.newaxis],perm[:,np.newaxis]),axis=1)
# df = pd.DataFrame(data=all_data,columns=['accuracy', 'perm'])
# df.to_csv('thesis_plots/SVM_results.csv')
