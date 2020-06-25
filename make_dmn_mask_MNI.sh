#!/bin/bash
#Author: Anne
#Purpose: register DMN mask from Yeo 2011 to MNI2009cAsym space
# Things it does
# 1. skull strip data
# 2. register to standard space
# 3. invert transformation


# We have Yeo in fsaverage space, want to convert to MNI space

project_path=/jukebox/norman/amennen/prettymouth
freesurfer_path=${project_path}/derivatives/freesurfer
Yeo_path=${freesurfer_path}/fsaverage/label
template_path=/jukebox/norman/amennen/MNI_things/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c
template_img=${template_path}/mni_icbm152_t1_tal_nlin_asym_09c.nii
output_dir=/jukebox/norman/amennen/MNI_things/Yeo_MNI_transforms/
fsaverage_dir=${project_path}/derivatives/freesurfer/fsaverage/mri




# okay the bottom I didn't like because it seemed too small though it fit well
# what I wound up doing was just orienting the given mask from the MNI Yeo catory reorient2std and that seemed to work
# then just used fslmaths to take 7
# then used jupyter notebook nilearn to resample everything, chose example subject for bold reslicing (subject 39) but shouldn't matter because just reslicing


# 1. Convert from .annot to .mgz

mri_label2vol --annot ${Yeo_path}/lh.Yeo2011_7Networks_N1000.annot --subject fsaverage --hemi lh --fillthresh 0 --temp ${fsaverage_dir}/aparc+aseg.mgz --identity --o ${output_dir}/lh.Yeo2011_7Networks_N1000.mgz
mri_label2vol --annot ${Yeo_path}/rh.Yeo2011_7Networks_N1000.annot --subject fsaverage --hemi rh --fillthresh 0 --temp ${fsaverage_dir}/aparc+aseg.mgz --identity --o ${output_dir}/rh.Yeo2011_7Networks_N1000.mgz


# 2. Combine mgz
mri_concat --i ${output_dir}/lh.Yeo2011_7Networks_N1000.mgz --i ${output_dir}/rh.Yeo2011_7Networks_N1000.mgz --max --o ${output_dir}/Yeo2011_7Networks_N1000.mgz

# 3. Convert from .mgz to .nii.gz
mri_convert -rl $template_img -rt nearest ${output_dir}/Yeo2011_7Networks_N1000.mgz ${output_dir}/Yeo2011_7Networks_N1000_convert2MNI.nii.gz

# Make DMN mask
fslmaths ${output_dir}/Yeo2011_7Networks_N1000_convert2MNI.nii.gz -thr 7 -uthr 7 -bin ${output_dir}/Yeo_DMN_mask.nii.gz


# then I reoriented 

# didnt' work--rotated
#mri_label2vol --annot ${Yeo_path}/lh.Yeo2011_7Networks_N1000.annot --subject fsaverage --hemi lh --fillthresh 0.2 --temp $template_img --identity --o ${output_dir}/lh.Yeo2011_7Networks_N1000.mgz

#mri_label2vol --annot ${Yeo_path}/rh.Yeo2011_7Networks_N1000.annot --subject fsaverage --hemi rh --fillthresh 0.2 --temp $template_img --identity --o ${output_dir}/Yeo_7_MNI_rh.nii.gz