# load DMN ROI and plot with pysurfer

import os
from surfer import Brain, project_volume_data

print(__doc__)

brain = Brain("fsaverage", "lh", "inflated")
volume_file = '/Volumes/norman/amennen/prettymouth_fmriprep2/ROI/TOM_large_resampled_maskedbybrain.nii.gz'
reg_file = os.path.join(os.environ["FREESURFER_HOME"],
                        "average/mni152.register.dat")
zstat = project_volume_data(volume_file, "lh", reg_file).astype(bool)
brain.add_data(zstat, min=0, max=1,colormap="bone", alpha=.6, colorbar=False)
brain.show_view("medial")