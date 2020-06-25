import numpy as np
import matplotlib.pyplot as plt
from brainiak.funcalign.srm import SRM
from time_segment_classification import (
    time_segment_correlation, correlation_classification)
from isc import isc, isfc, compute_summary_statistic
from scipy.stats import zscore
from time import time

train_stories = ['pieman', 'milkyway-vodka', 'prettymouth-affair',
                 'notthefall', 'slumlordreach']

test_stories = ['slumlordreach']
assert set(test_stories) < set(train_stories)

# Length of segments for time-segment classification
segment_length = 10

# Switch for time-series SRM?
timeseries_srm = False

# Load ROI data from posterior cingulate
roi_datasets = np.load('posterior_cingulate_bh_clean_datasets.npy').item()

# Aggregate and z-score the ROI data
train_roi, test_roi = {}, {}
for story in train_stories:
    roi_data = roi_datasets[story]['data']
    if story in test_stories:
        roi_trs = roi_data.shape[0]
        train_roi[story] = np.nan_to_num(zscore(roi_data[:roi_trs//2, ...], axis=0))
        test_roi[story] = np.nan_to_num(zscore(roi_data[roi_trs//2:, ...], axis=0))
    else:
        train_roi[story] = np.nan_to_num(zscore(roi_data, axis=0))
    
# Time-series SRM on ROI data
if timeseries_srm:
    test_shared = {}
    for story in stories:
        srm = SRM(n_iter=n_iter, features=n_features)

        # Change subjects to list for SRM
        train_list = [train.T for train in np.moveaxis(train_roi[story], 2, 0)]
        test_list = [test.T for test in np.moveaxis(test_roi[story], 2, 0)]

        # Train SRM and apply
        srm.fit(train_list)

        test_transformed = srm.transform(test_list)
        test_shared[story] = np.dstack([test.T for test in test_transformed])

# Load in the whole-brain surface data
train_target, test_target = {}, {}
for story in train_stories:
    
    # Stack left and right hemispheres
    target_data = np.hstack((np.load(f'{story}_cortex_lh_data.npy'),
                             np.load(f'{story}_cortex_lh_data.npy')))
    
    # Remove zeroed vertices (medial wall)
    target_data = target_data[:, ~np.all(target_data == 0, axis=(0, 2))]
    
    # Split targets into training and test sets
    target_trs = target_data.shape[0]
    if story in test_stories:
        train_target[story] = np.nan_to_num(zscore(target_data[:target_trs//2, ...], axis=0))
        test_target[story] = np.nan_to_num(zscore(target_data[target_trs//2:, ...], axis=0))
    else:
        train_target[story] = np.nan_to_num(zscore(target_data, axis=0))
    
    print(f"Finished loading surface data for {story}")
    
# Compute ISCs for target data
load_iscs = False
if load_iscs:
    train_target_iscs = np.load('train_target_iscs.npy').item()
    
else:
    train_target_iscs = {}
    for story in train_stories:
        train_target_iscs[story] = isc(train_target[story])
        print(f"Finished computing ISC for {story}")

    np.save('train_target_iscs_split-story.npy', train_target_iscs)

# Get mean ISC across stories and create mask
train_target_mean_isc = compute_summary_statistic([compute_summary_statistic(
                                                    train_target_iscs[story], axis=0)
                                                   for story in train_stories], axis=0)
target_mask = train_target_mean_isc > .1

# Mask train target data
train_target_masked = {}
for story in train_stories:
    train_target_masked[story] = train_target[story][:, target_mask, :]

# Compute ISFC between voxels of interest and connectivity targets
load_isfcs = False
if load_isfcs:
    train_isfcs = {}
    for story in train_stories:
        train_isfcs[story] = np.load(f'train_isfcs_{story}_split-story.npy')
else:
    train_isfcs = {}
    for story in train_stories:
        assert train_roi[story].shape[0] == train_target_masked[story].shape[0]
        isfcs = isfc(train_roi[story], train_target_masked[story], pairwise=False)
        train_isfcs[story] = isfcs
        np.save('train_isfcs_{story}_split-story.npy', isfcs)
        print(f"Finished computing ROI-target ISFCs for {story}")
        
# Z-score ISFCs individually
train_isfcs_z = {}
for story in train_stories:
    train_isfcs_z[story] = np.nan_to_num(zscore(np.arctanh(
                                train_isfcs[story]), axis=1))
    np.save(f'train_isfcs_zscored_{story}_split-story.npy', train_isfcs_z[story])
    print(f"Finished z-scoring ISFCs for {story}")
    
train_isfcs_z = {}
for story in stories:
    train_isfcs_z[story] = np.load(f'train_isfcs_zscored_{story}.npy')
    
# Stack ISFCs now that we're in the same space
train_isfcs_stack = np.dstack([train_isfcs_z[story] for story in stories])
train_isfcs_all = np.arange(train_isfcs_stack.shape[2])
train_isfcs_ids = {}
for story in stories:
    train_isfcs_n = train_isfcs_z[story].shape[2]
    train_isfcs_ids[story] = train_isfcs_all[:train_isfcs_n]
    train_isfcs_all = train_isfcs_all[train_isfcs_n:]
np.save('train_isfcs_zscored_stack.npy', train_isfcs_stack)
np.save('train_isfcs_ids.npy', train_isfcs_ids)

train_isfcs_ids = np.load('train_isfcs_ids.npy').item()

# Change subjects to list for SRM
train_list = [train for train in np.moveaxis(train_isfcs_stack, 2, 0)]

# Set k shared features for SRM (and number of iterations)
n_features = 800
n_iter = 10

# Train connectivity SRM
c_srm = SRM(n_iter=n_iter, features=n_features)

# Train SRM and apply
start = time()
c_srm.fit(train_list)
print(f"SRM timer: {time() - start}")
    
# Transform test date into shared space
test_shared = {}
for story in stories:
    test_list = [test.T for test in np.moveaxis(test_roi[story], 2, 0)]
    
    # Loop though subjects and apply transformation
    assert len(train_isfcs_ids[story]) == len(test_list)
    transformed_tests = []
    for train_id, test_subject in zip(train_isfcs_ids[story], test_list):
        
        # Apply this subject's W transformation
        test_transformed = c_srm.w_[train_id].T.dot(test_subject)
        transformed_tests.append(test_transformed)
    
    test_shared[story] = np.dstack([test.T for test in transformed_tests])
np.save(f'test_shared_k{n_features}.npy', test_shared)
    
# Run time-segment classification
test_data = test_roi
for story in stories:
    correlations = time_segment_correlation(test_data[story], segment_length)
    accuracies = correlation_classification(correlations)
    print(f"Mean accuracy for {story}: {np.mean(accuracies['accuracies'])*100:.2f}% "
          f"(chance = {accuracies['chance']:.2f})") 
    
####

# Reformat training and test data into lists for SRM
#train_list = [train.T for train in np.moveaxis(train_data, 2, 0)]
train_list_t = [train.T for train in np.moveaxis(train_roi_masked, 2, 0)]
train_list = [train for train in np.moveaxis(isfcs, 2, 0)]
test_list = [test.T for test in np.moveaxis(test_roi_masked, 2, 0)]

    
# Set k shared features for SRM (and number of iterations)
n_features = 50
n_iter = 20

# Fit SRM on training data
srm = SRM(n_iter=n_iter, features=n_features)
srm = SRM(n_iter=n_iter, features=n_features)
srm.fit(train_list)
srm.fit(train_list_t)

# Apply SRM transformations to test data
test_roi_shared = srm.transform(test_list)
test_roi_shared = np.dstack([test.T for test in test_roi_shared])

orig_correlations = time_segment_correlation(test_roi_masked, 10)
orig_accuracies = correlation_classification(orig_correlations)

srm_correlations = time_segment_correlation(test_roi_shared, 10)
srm_accuracies = correlation_classification(srm_correlations)

print(f"Original accuracy: {np.mean(orig_accuracies['accuracies'])}; "
      f"SRM accuracy: {np.mean(srm_accuracies['accuracies'])}" )

# Compute ISFC between voxels of interest and connectivity targets
isfcs = isfc(train_data, train_targets, pairwise=False)

n_segments = test_data.shape[0] // segment_length

# Reformat training and test data into lists for SRM
#train_list = [train.T for train in np.moveaxis(train_data, 2, 0)]
train_list = [train for train in np.moveaxis(isfcs, 2, 0)]
test_list = [test.T for test in np.moveaxis(test_data, 2, 0)]

# Set k shared features for SRM (and number of iterations)
n_features = 50
n_iter = 20

# Fit SRM on training data
srm = SRM(n_iter=n_iter, features=n_features)
srm.fit(train_list)

# Apply SRM transformations to test data
shared_test = srm.transform(test_list)
shared_test = np.dstack([test.T for test in shared_test])

# Time-segment classification on original data
correlations = time_segment_correlation(test_data, 10)
accuracies = correlation_classification(correlations)
print(f"Accuracy without SRM = {np.mean(accuracies):.2f};",
      f"(chance accuracy = {1 / n_segments})")

# Time-segment classification on data in shared space
correlations = time_segment_correlation(shared_test, 10)
accuracies = correlation_classification(correlations)
print(f"Accuracy with SRM = {np.mean(accuracies):.2f};",
      f"(chance accuracy = {1 / n_segments})")

n_TRs = 600
n_voxels = 200
n_subjects = 20
noise = 10

segment_length = 10

signal = np.random.randn(n_TRs, n_voxels)
data = []
for subject in np.arange(n_subjects):
    data.append(np.hstack((signal[:, :n_voxels // 2],
                np.hstack(np.random.permutation(np.split(
                   signal[:, n_voxels // 2:], n_voxels // 2, axis=1)))))
     + np.random.randn(n_TRs, n_voxels) * noise)
data = np.dstack(data)
random_data = np.random.randn(n_TRs, n_voxels, n_subjects)

targets = np.random.randn(n_TRs, 1000, n_subjects)

assert data.shape == (n_TRs, n_voxels, n_subjects)

train_data, test_data = data[:300, ...], data[300:, ...]
train_targets = targets[:300, ...]
