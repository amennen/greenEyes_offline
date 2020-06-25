#!/bin/bash

# purpose: make sphere based on ROI coordinates
# supply coordinates from thresholded--let's go with threshold of 5

ROIDIR=/jukebox/norman/amennen/prettymouth/ROI
ROI_LARGE=theory_mind_association_z_FDR_0.01.nii.gz
STANDARD=/jukebox/pkgs/FSL/5.0.9/data/standard/MNI152_T1_2mm.nii.gz

c=1
ROI_IND=(49 -58 23) # the (X,Y,Z) = (20,34,47) # but the mask is 0 (mistake -- this should have been )

c=2
ROI_IND=(68 33 46) # this is left TPJ - cluster 2 from tabale S1 (-45,-57,22)

c=3
ROI_IND=(46 35 56) # this is cluster 3 from table S1 - precuneus

c=4
ROI_IND=(45 36 49) # this is cluster 4 from table S1 - posterior cingulate

c=5
ROI_IND=(43 90 33) # this is cluster 5 from table S1 - vmPFC

c=6
ROI_IND=(44 82 51) # this is cluster 6 from table S1 - dmPFC

c=7
ROI_IND=(20 79 35) # this is cluster 7 from table S1 - rIFG

c=8
ROI_IND=(69 78 37) # this is cluster 8 from table S1 - lIFG

c=9
ROI_IND=(20 35 46) # this is cluster 10 from table S1 - lSTS

c=10
ROI_IND=(71 71 26) # this is cluster 12 from table S1 - l temporal pole


fslmaths $STANDARD -mul 0 -add 1 -roi ${ROI_IND[0]} 1 ${ROI_IND[1]} 1 ${ROI_IND[2]} 1 0 1 ${ROIDIR}/cluster${c}.nii.gz -odt float
fslmaths ${ROIDIR}/cluster${c}.nii.gz -kernel box 10 -fmean ${ROIDIR}/cluster${c}sphere.nii.gz
fslmaths ${ROIDIR}/cluster${c}sphere.nii.gz -bin ${ROIDIR}/cluster${c}sphere_bin.nii.gz


# now add all of the clusters into one ROI
fslmaths ${ROIDIR}/cluster1sphere_bin.nii.gz -add ${ROIDIR}/cluster2sphere_bin.nii.gz -add ${ROIDIR}/cluster3sphere_bin.nii.gz \
	-add ${ROIDIR}/cluster4sphere_bin.nii.gz -add ${ROIDIR}/cluster5sphere_bin.nii.gz -add ${ROIDIR}/cluster6sphere_bin.nii.gz \
	-add ${ROIDIR}/cluster7sphere_bin.nii.gz  -add ${ROIDIR}/cluster8sphere_bin.nii.gz  -add ${ROIDIR}/cluster9sphere_bin.nii.gz \
	-add ${ROIDIR}/cluster10sphere_bin.nii.gz \
	-bin ${ROIDIR}/top10clusters.nii.gz


# now mask by the brain
