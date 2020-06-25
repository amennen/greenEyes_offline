#!/usr/bin/env python

# Script to convert and populate data in BIDS format
# Run with something like:
# ./code/BIDS_prettymouth.py 

from os.path import exists, join
from os import makedirs
from glob import glob
from shutil import copyfile
import nibabel as nib
import json
import csv
import pandas as pd

# Set to True to actually copy files
copy_files = True

# Source of raw data on Princeton server
source_dir = '/jukebox/hasson/formerLabMembers/yaara/understandExp/Raw'

# Base directory for data collection
base_dir = '/jukebox/hasson/snastase/narratives'
example_dir = join(base_dir, 'dcm_examples', 'prettymouth')

# Path to current BIDS directory
bids_dir = '/jukebox/hasson/snastase/narratives/prettymouth'
if not exists(bids_dir):
    makedirs(bids_dir)
    
# Session label to BIDS ID mapping
with open(join(base_dir, 'bids_ids.json')) as f:
    bids_ids = json.load(f)

# Set number of BOLD TRs to check
n_trs = 475

# Original subject / session IDs and demographics
exclude = []
df = pd.read_table(join(base_dir, 'subjects_spreadsheet_new.tsv'))
df = df.loc[df['Story'] == 'Pretty Mouth and Green My Eyes'][['Session', 'Age', 'Sex', 'Manipulation', 'Comprehension (score)']]
### GET COMPREHENSION AND MANIPULATION IN HERE!!!
df = df.loc[~df['Session'].isin(exclude)]
df = df.reset_index()
df['participant_id'] = None
for i, row in df.iterrows():
    df['participant_id'].iloc[i] = 'sub-' + bids_ids[row['Session']]

source_ids = df['Session'].tolist()
df = df.drop(labels=['Session', 'index'], axis=1)
rename = {'Age': 'age', 'Sex': 'sex', 'Comprehension (score)': 'comprehension', 'Manipulation': 'group'}
df = df.rename(columns=rename)
df = df[['participant_id', 'age', 'sex', 'group', 'comprehension']]
df = df.fillna('n/a')

bids_ids = df['participant_id'].tolist()
assert len(source_ids) == len(bids_ids)

df.to_csv(join(bids_dir, 'participants.tsv'), sep='\t', index=False)

# Top-level json dataset descriptor
description_fn = join(bids_dir, 'dataset_description.json')
if not exists(description_fn):
    description = {u'Funding': [u'TODO'], 
                   u'Name': u'Pretty Mouth and Green My Eyes',
                   u'License': u'PDDL (http://opendatacommons.org/licenses/pddl/)',
                   u'HowToAcknowledge': u'TODO',
                   u'Authors': [u'Yaara Yeshurun', 'Yun-Fei Liu', 'Samuel A. Nastase', 'Uri Hasson'],
                   u'ReferencesAndLinks': [u'TODO'],
                   u'DatasetDOI': u'TODO',
                   u'BIDSVersion': u'1.1.0',
                   u'Acknowledgements': u'We thank the administrative staff of the Princeton Neuroscience Institute.'}
    with open(description_fn, 'w') as f:
        json.dump(description, f, sort_keys=True, indent=2)

# Get func and anat DICOM metadata from example files
with open(join(example_dir, 'prettymouth_func_example.json')) as f:
    func_example = json.load(f)
    
with open(join(example_dir, 'prettymouth_anat_example.json')) as f:
    anat_example = json.load(f)
        
dcm_get = ['RepetitionTime', 'Manufacturer',
           'ManufacturersModelName', 'MagneticFieldStrength',
           'DeviceSerialNumber', 'StationName', 'SoftwareVersions',
           'ReceiveCoilName', 'ReceiveCoilActiveElements',
           'ScanningSequence', 'SequenceVariant', 'SequenceName',
           'ScanOptions', 'PulseSequenceDetails',
           'ParallelReductionFactorInPlane',
           'ParallelReductionTechnique', 'PartialFourier',
           'PhaseEncodingDirection', 'EffectiveEchoSpacing',
           'TotalReadoutTime', 'EchoTime', 'SliceTiming', 
           'DwellTime', 'FlipAngle', 'MultibandAccelerationFactor',
           'InversionTime', 'PixelBandwidth']

func_meta = {'TaskName': 'prettymouth',
             'ParellelReductionType': 'TODO', 
             'PulseSequenceType': 'Gradient Echo EPI',                    
             'NumberOfVolumesDiscardedByScanner': 'TODO',                    
             'InstitutionName': 'Princeton University',                    
             'InstitutionAddress': ('Washington Rd, Building 25, '
                                    'Princeton, NJ 08540, USA'),
             'InstitutionalDepartmentName': 'Princeton Neuroscience Institute',
             'TaskDescription': ("Passively listened to audio story "
                                 "'Pretty Mouth and Green My Eyes' "
                                 "by J. D. Salinger"),
             'CogAtlasID': 'https://www.cognitiveatlas.org/task/id/trm_4c8991fadfe01'}


anat_meta = {'ParellelReductionType': 'TODO', 
             'PulseSequenceType': 'MPRAGE',                    
             'InstitutionName': 'Princeton University',                    
             'InstitutionAddress': ('Washington Rd, Building 25, '
                                    'Princeton, NJ 08540, USA'),
             'InstitutionalDepartmentName': 'Princeton Neuroscience Institute'}

for key in dcm_get:
    # Try to grab func metadata from example json
    try:
        func_meta[key] = func_example[key]
    except KeyError:
        print(f'{key} not found in example functional json!!!')
        
    # Try to grab func metadata from example json
    try:
        anat_meta[key] = anat_example[key]
    except KeyError:
        print(f'{key} not found in example anatomical json!!!')

# Create top level task file (a little redundant but oh well)
task_fn = join(bids_dir, 'task-prettymouth_bold.json')                              
with open(task_fn, 'w') as f:
    json.dump(func_meta, f, sort_keys=True, indent=2)

n_func, n_anat = [], []
for source_id, bids_id in zip(source_ids, bids_ids):
    if not exists(join(bids_dir, bids_id)):
        makedirs(join(bids_dir, bids_id))

    func_dir = join(bids_dir, bids_id, 'func')
    anat_dir = join(bids_dir, bids_id, 'anat')
    
    if not exists(func_dir):
        makedirs(func_dir)

    if not exists(anat_dir):
        makedirs(anat_dir)
    
    func_fns = glob(join(source_dir, source_id, 'NII', '*epi*greenEyes.nii.gz'))
    func_keep = []
    for fn in func_fns:
        if nib.load(fn).shape[-1] == n_trs:
            func_keep.append(fn)
        else:
            print("Found EPI with wrong number of TRs--skipping it")
    if len(func_fns) == 0:
        # Some files with 475 were improperly named with "piemanaudio" suffix
        print("WARNING: Found no matching EPIs for subject {0}--using number of TRs!".format(source_id))
        misnamed_fns = glob(join(source_dir, source_id, 'NII', '*epi*.nii.gz'))
        for m_fn in misnamed_fns:
            if nib.load(m_fn).shape[-1] == n_trs:
                func_fns.append(m_fn)
                func_keep.append(m_fn)
    assert len(func_keep) == 1
    func_in = func_fns[0]
    n_func.append(func_in)
    func_out = join(func_dir, bids_id + '_task-prettymouth_bold.nii.gz')
    func_json = join(func_dir, bids_id + '_task-prettymouth_bold.json')
    print("Copying {0}\n\tto {1}\n\twith {2}".format(
            func_in, func_out, func_json))
    if copy_files:
        copyfile(func_in, func_out)
        with open(func_json, 'w') as f:
            json.dump(func_meta, f, sort_keys=True, indent=2)
    
    anat_fns = glob(join(source_dir, source_id, 'NII', 'o*mprage*.nii.gz'))
    # In case subject only has "co" (i.e., cropped) anatomical
    if source_id == 'BC_031813':
        anat_fns = [join('/jukebox/hasson/janice/Pieman/piesky',
                         'subjects/BC_022513/data/nifti',
                         'BC_022513_t1_mprage_192.nii.gz')]
    if len(anat_fns) == 0:
        print("WARNING: Found no uncropped anatomical for subject {0}--using cropped version!".format(source_id))
        anat_fns.append(glob(join(source_dir, source_id, 'NII', 'co*mprage*.nii.gz'))[0])    
    assert len(anat_fns) == 1
    anat_in = anat_fns[0]
    n_anat.append(anat_in)
    anat_out = join(anat_dir, bids_id + '_T1w.nii.gz')
    anat_json = join(anat_dir, bids_id + '_T1w.json')
    print("Copying {0}\n\tto {1}\n\twith {2}".format(
            anat_in, anat_out, anat_json))
    
    if copy_files:
        copyfile(anat_in, anat_out)
        with open(anat_json, 'w') as f:
            json.dump(anat_meta, f, sort_keys=True, indent=2)

assert len(bids_ids) == len(n_func) == len(n_anat)
