#!/bin/bash

# Purpose check registration from a new file to the example functional used for a given subject

module load ANTs/2.3.1 


wf_dir=/jukebox/norman/amennen/prettymouth_fmriprep2/derivatives/work/fmriprep_wf/single_subject_031_wf/
ref_BOLD=${wf_dir}/func_preproc_task_prettymouth_wf/bold_reference_wf/validate_ref/ref_image_valid.nii.gz
BOLD_to_T1=${wf_dir}/func_preproc_task_prettymouth_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt
T1_to_MNI=${wf_dir}/anat_preproc_wf/t1_2_mni/ants_t1_to_mniComposite.h5
MNI_ref=/jukebox/norman/amennen/MNI_things/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii.gz
# now make another reference image in BOLD space
# fslmaths mni_icbm152_t1_tal_nlin_asym_09c_BOLD.nii -mas mni_icbm152_t1_tal_nlin_asym_09c_BOLD_mask.nii mni_icbm152_t1_tal_nlin_asym_09c_BOLD_brain.nii
MNI_ref_BOLD=/jukebox/norman/amennen/MNI_things/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_BOLD_brain.nii.gz
# need to remake these to make sure same number of slices as BOLD data **
# BOLD reference
func_dir=/jukebox/norman/amennen/prettymouth_fmriprep2/sub-031/func

# in mni/mni/mni directory**
#python resample.py mni_icbm152_t1_tal_nlin_asym_09c_brain.nii.gz /jukebox/norman/amennen/prettymouth_fmriprep2/derivatives/fmriprep/sub-031/func/sub-031_task-prettymouth_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

# fslroi sub-031_task-prettymouth_bold.nii.gz sub-031_task-prettymouth_bold_TR0.nii.gz 0 1
# should brain extract BOLD first
BOLD_ex=${func_dir}/sub-031_task-prettymouth_bold_TR0.nii.gz

# fixed = MNI ref
# moving = BOLD ex
# output is in MNI space**
antsApplyTransforms --default-value 0 --float 1 --interpolation LanczosWindowedSinc -d 3 -e 3 --input $BOLD_ex --reference-image $MNI_ref --output testBOLD_to_MNI_1m_LW.nii.gz --transform $BOLD_to_T1 --transform $T1_to_MNI -v 1

# try MNI_ref in BOLD space now
# USE THIS ONE!!
antsApplyTransforms --default-value 0 --float 1 --interpolation LanczosWindowedSinc -d 3 -e 3 --input $BOLD_ex --reference-image $MNI_ref_BOLD --output testBOLD_to_MNI_BOLD_3m_LW.nii.gz --transform $BOLD_to_T1 --transform $T1_to_MNI -v 1

fslview_deprecated testBOLD_to_MNI_BOLD.nii.gz /jukebox/norman/amennen/prettymouth_fmriprep2/derivatives/fmriprep/sub-031/func/sub-031_task-prettymouth_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz &

# check reference image is the right one
# to do LanczosWindowedSinc, try 1 mm
