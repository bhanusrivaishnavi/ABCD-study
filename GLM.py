import numpy as np
from nilearn.masking import unmask
import numpy as np
import statsmodels.api as sm
import nibabel as nib
import pandas as pd

masked_data=np.load("masked_data_abcd_full.npy")
print(masked_data.shape)
meta_data_df=pd.read_csv("Corr_Data.csv")
print(meta_data_df.shape)

y = meta_data_df["cbcl_scr_syn_anxdep_r"].values   
gender = meta_data_df["demo_sex_v2"].values  
age = meta_data_df["interview_age"].values
print(set(list(y)))
print(meta_data_df.shape)

n_voxels = masked_data.shape[1]  

print(n_voxels)

voxel_coefficients = np.zeros(n_voxels)  
gender_coefficients = np.zeros(n_voxels)
age_coefficients = np.zeros(n_voxels)

voxel_p_values = np.zeros(n_voxels)  
gender_p_values = np.zeros(n_voxels)
age_p_values = np.zeros(n_voxels)

for v in range(n_voxels):
    print(v)
    X = np.column_stack((gender, age, masked_data[:, v]))  
    X_with_intercept = sm.add_constant(X)  
    
    neg_bin_model = sm.GLM(y, X_with_intercept, family=sm.families.NegativeBinomial(alpha=0.1))
    neg_bin_results = neg_bin_model.fit()
    
    gender_column_index = X_with_intercept.shape[1] - 3  
    age_column_index = X_with_intercept.shape[1] - 2  
    voxel_column_index = X_with_intercept.shape[1] - 1  

    voxel_coefficients[v] = neg_bin_results.params[voxel_column_index]  
    gender_coefficients[v] = neg_bin_results.params[gender_column_index]  
    age_coefficients[v] = neg_bin_results.params[age_column_index]  

    voxel_p_values[v] = neg_bin_results.pvalues[voxel_column_index]
    gender_p_values[v] = neg_bin_results.pvalues[gender_column_index]
    age_p_values[v] = neg_bin_results.pvalues[age_column_index]

mask_img = nib.load("mask.nii")

voxel_coeff_img = unmask(voxel_coefficients, mask_img)
gender_coeff_img = unmask(gender_coefficients, mask_img)
age_coeff_img = unmask(age_coefficients, mask_img)

voxel_pval_img = unmask(voxel_p_values, mask_img)
gender_pval_img = unmask(gender_p_values, mask_img)
age_pval_img = unmask(age_p_values, mask_img)

voxel_coeff_img.to_filename("voxel_rscore_nc.nii")
gender_coeff_img.to_filename("gender_rscore_nc.nii")
age_coeff_img.to_filename("age_rscore_nc.nii")

voxel_pval_img.to_filename("voxel_p_rscore_nc.nii")
gender_pval_img.to_filename("gender_p_rscore_nc.nii")
age_pval_img.to_filename("age_p_rscore_nc.nii")

print("Coefficient and p-value images saved successfully!")


y = meta_data_df["cbcl_scr_syn_anxdep_t"].values   
gender = meta_data_df["demo_sex_v2"].values  
age = meta_data_df["interview_age"].values
print(set(list(y)))
print(meta_data_df.shape)

n_voxels = masked_data.shape[1]  

print(n_voxels)

voxel_coefficients = np.zeros(n_voxels)  
gender_coefficients = np.zeros(n_voxels)
age_coefficients = np.zeros(n_voxels)

voxel_p_values = np.zeros(n_voxels)  
gender_p_values = np.zeros(n_voxels)
age_p_values = np.zeros(n_voxels)

for v in range(n_voxels):
    print(v)
    X = np.column_stack((gender, age, masked_data[:, v]))  
    X_with_intercept = sm.add_constant(X)  
    
    neg_bin_model = sm.GLM(y, X_with_intercept, family=sm.families.NegativeBinomial(alpha=0.1))
    neg_bin_results = neg_bin_model.fit()
    
    gender_column_index = X_with_intercept.shape[1] - 3  
    age_column_index = X_with_intercept.shape[1] - 2  
    voxel_column_index = X_with_intercept.shape[1] - 1  

    voxel_coefficients[v] = neg_bin_results.params[voxel_column_index]  
    gender_coefficients[v] = neg_bin_results.params[gender_column_index]  
    age_coefficients[v] = neg_bin_results.params[age_column_index]  

    voxel_p_values[v] = neg_bin_results.pvalues[voxel_column_index]
    gender_p_values[v] = neg_bin_results.pvalues[gender_column_index]
    age_p_values[v] = neg_bin_results.pvalues[age_column_index]

mask_img = nib.load("mask.nii")

voxel_coeff_img = unmask(voxel_coefficients, mask_img)
gender_coeff_img = unmask(gender_coefficients, mask_img)
age_coeff_img = unmask(age_coefficients, mask_img)

voxel_pval_img = unmask(voxel_p_values, mask_img)
gender_pval_img = unmask(gender_p_values, mask_img)
age_pval_img = unmask(age_p_values, mask_img)

voxel_coeff_img.to_filename("voxel_tscore_nc.nii")
gender_coeff_img.to_filename("gender_tscore_nc.nii")
age_coeff_img.to_filename("age_tscore_nc.nii")

voxel_pval_img.to_filename("voxel_p_tscore_nc.nii")
gender_pval_img.to_filename("gender_p_tscore_nc.nii")
age_pval_img.to_filename("age_p_tscore_nc.nii")

print("Coefficient and p-value images saved successfully!")




import numpy as np
import pandas as pd
from nilearn.masking import unmask
import numpy as np
import statsmodels.api as sm
import nibabel as nib

masked_data=np.load("masked_data_mdd_ds.npy")
print(masked_data.shape)
meta_data_df=pd.read_csv("Corr_Data_mdd_ds.csv")
print(meta_data_df.shape)

y = meta_data_df["current_score"].values   
gender = meta_data_df["demo_sex_v2"].values  
age = meta_data_df["interview_age"].values
print(set(list(y)))

print(meta_data_df.shape)


n_voxels = masked_data.shape[1]  

print(n_voxels)

voxel_coefficients = np.zeros(n_voxels)  
gender_coefficients = np.zeros(n_voxels)
age_coefficients = np.zeros(n_voxels)

voxel_p_values = np.zeros(n_voxels)  
gender_p_values = np.zeros(n_voxels)
age_p_values = np.zeros(n_voxels)

for v in range(n_voxels):
    #print(v)
    X = np.column_stack((gender, age, masked_data[:, v]))  
    X_with_intercept = sm.add_constant(X)  
    
    neg_bin_model = sm.GLM(y, X_with_intercept, family=sm.families.NegativeBinomial(alpha=0.1))
    neg_bin_results = neg_bin_model.fit()
    
    gender_column_index = X_with_intercept.shape[1] - 3  
    age_column_index = X_with_intercept.shape[1] - 2  
    voxel_column_index = X_with_intercept.shape[1] - 1  

    voxel_coefficients[v] = neg_bin_results.params[voxel_column_index]  
    gender_coefficients[v] = neg_bin_results.params[gender_column_index]  
    age_coefficients[v] = neg_bin_results.params[age_column_index]  

    voxel_p_values[v] = neg_bin_results.pvalues[voxel_column_index]
    gender_p_values[v] = neg_bin_results.pvalues[gender_column_index]
    age_p_values[v] = neg_bin_results.pvalues[age_column_index]

mask_img = nib.load("mask_mdd_ds.nii")

voxel_coeff_img = unmask(voxel_coefficients, mask_img)
gender_coeff_img = unmask(gender_coefficients, mask_img)
age_coeff_img = unmask(age_coefficients, mask_img)

voxel_pval_img = unmask(voxel_p_values, mask_img)
gender_pval_img = unmask(gender_p_values, mask_img)
age_pval_img = unmask(age_p_values, mask_img)

voxel_coeff_img.to_filename("voxel_current_nc.nii")
gender_coeff_img.to_filename("gender_current_nc.nii")
age_coeff_img.to_filename("age_current_nc.nii")

voxel_pval_img.to_filename("voxel_p_current_nc.nii")
gender_pval_img.to_filename("gender_p_current_nc.nii")
age_pval_img.to_filename("age_p_current_nc.nii")

print("Coefficient and p-value images saved successfully!")


y = meta_data_df["lifetime_score"].values   
gender = meta_data_df["demo_sex_v2"].values  
age = meta_data_df["interview_age"].values
print(set(list(y)))

print(meta_data_df.shape)


n_voxels = masked_data.shape[1]  

print(n_voxels)

voxel_coefficients = np.zeros(n_voxels)  
gender_coefficients = np.zeros(n_voxels)
age_coefficients = np.zeros(n_voxels)

voxel_p_values = np.zeros(n_voxels)  
gender_p_values = np.zeros(n_voxels)
age_p_values = np.zeros(n_voxels)

for v in range(n_voxels):
    #print(v)
    X = np.column_stack((gender, age, masked_data[:, v]))  
    X_with_intercept = sm.add_constant(X)  
    
    neg_bin_model = sm.GLM(y, X_with_intercept, family=sm.families.NegativeBinomial(alpha=0.1))
    neg_bin_results = neg_bin_model.fit()
    
    gender_column_index = X_with_intercept.shape[1] - 3  
    age_column_index = X_with_intercept.shape[1] - 2  
    voxel_column_index = X_with_intercept.shape[1] - 1  

    voxel_coefficients[v] = neg_bin_results.params[voxel_column_index]  
    gender_coefficients[v] = neg_bin_results.params[gender_column_index]  
    age_coefficients[v] = neg_bin_results.params[age_column_index]  

    voxel_p_values[v] = neg_bin_results.pvalues[voxel_column_index]
    gender_p_values[v] = neg_bin_results.pvalues[gender_column_index]
    age_p_values[v] = neg_bin_results.pvalues[age_column_index]

mask_img = nib.load("mask_mdd_ds.nii")

voxel_coeff_img = unmask(voxel_coefficients, mask_img)
gender_coeff_img = unmask(gender_coefficients, mask_img)
age_coeff_img = unmask(age_coefficients, mask_img)

voxel_pval_img = unmask(voxel_p_values, mask_img)
gender_pval_img = unmask(gender_p_values, mask_img)
age_pval_img = unmask(age_p_values, mask_img)

voxel_coeff_img.to_filename("voxel_lifetime_nc.nii")
gender_coeff_img.to_filename("gender_lifetime_nc.nii")
age_coeff_img.to_filename("age_lifetime_nc.nii")

voxel_pval_img.to_filename("voxel_p_lifetime_nc.nii")
gender_pval_img.to_filename("gender_p_lifetime_nc.nii")
age_pval_img.to_filename("age_p_lifetime_nc.nii")

print("Coefficient and p-value images saved successfully!")

