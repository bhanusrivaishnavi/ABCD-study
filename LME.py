from nilearn import plotting
from nilearn.masking import unmask
import pandas as pd
import numpy as np
import warnings
import statsmodels.formula.api as smf
import nibabel as nib

warnings.filterwarnings("ignore")

masked_data = np.load("masked_data_ica.npy")
meta_data_df= pd.read_csv("meta_data_cog_beh.csv")  

meta_data_df = meta_data_df[["demo_sex_v2", "cbcl_scr_syn_anxdep_t_x","site_id_l","interview_age"]]

n_voxels = masked_data.shape[1]

voxel_coefficients = np.zeros(n_voxels)
p_values = np.zeros(n_voxels)
gender_coefficients = np.zeros(n_voxels)

for v in range(n_voxels):
    print(v)
    df = meta_data_df.copy()
    df["voxel"] = masked_data[:, v]  
    
    model = smf.mixedlm("cbcl_scr_syn_anxdep_t_x ~ voxel + demo_sex_v2 + interview_age", df, groups=df["site_id_l"])
    result = model.fit(method="lbfgs", maxiter=1000)
    
    voxel_coefficients[v] = result.params["voxel"]
    p_values[v] = result.pvalues["voxel"]

mask_img = nib.load("mask.nii")
coeff_img = unmask(voxel_coefficients, mask_img)
pval_img = unmask(p_values,mask_img)

coeff_img.to_filename("voxel_anxdep.nii")
pval_img.to_filename("voxel_p_anxdep.nii")
