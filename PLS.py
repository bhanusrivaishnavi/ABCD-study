import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from brainspace.null_models import compute_mem, moran_randomization
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# ---- 1) Data Preparation ----

voxel_img = nib.load("average_map_3475.nii")   
mask_img = nib.load("mask.nii")        
atlas_img = nib.load("Merged_Atlas.nii.gz")
voxel_data = voxel_img.get_fdata()
mask_data = mask_img.get_fdata()
atlas_data = atlas_img.get_fdata().astype(int)

masked_voxels = np.where(mask_data > 0, voxel_data, np.nan)

print(masked_voxels.shape)
region_ids = np.unique(atlas_data)
region_ids = region_ids[region_ids > 0]   

t=0
Y = []
for rid in region_ids:
    region_mask = (atlas_data == rid)
    region_values = masked_voxels[region_mask]
    t+=region_values.shape[0]
    region_values = region_values[~np.isnan(region_values)]
    Y.append(region_values.mean())

Y = np.array(Y) 
print(voxel_data.shape,mask_data.shape,atlas_data.shape)
print(Y.shape)


df = pd.read_csv("Expression_output.csv")
X= df.select_dtypes(include=[float, int]).values
print(X.shape)


Y=Y.reshape(-1,1)
X_scaled = StandardScaler().fit_transform(X)
Y_scaled = StandardScaler().fit_transform(Y)
print(X_scaled.shape,Y_scaled.shape)


X=X_scaled
Y=Y_scaled


n_perm = 1000
n_components = 63

print("Y shape:", Y.shape)

df = pd.read_csv("region_middle_voxels.csv")

coords = df["MiddleVoxelMNI"].str.strip("()").str.split(",", expand=True).astype(float).values

D = cdist(coords, coords, metric="euclidean")  # (486, 486)

W = 1 / (D + np.eye(D.shape[0]))   
np.fill_diagonal(W, 0)             

mem, evals = compute_mem(W)

# ---- 2) True model ----
pls = PLSRegression(n_components=n_components)
pls.fit(X, Y)
Y_pred = pls.predict(X).ravel()
true_r2 = r2_score(Y, Y_pred)
true_corrs = [pearsonr(pls.x_scores_[:, i], pls.y_scores_[:, i])[0] for i in range(n_components)]

# ---- 3) Null distribution via Moran randomization ----
null_r2 = np.zeros(n_perm)
null_corrs = np.zeros((n_perm, n_components))
print(mem.shape,evals.shape)

for p in range(n_perm):
    Y_null = moran_randomization(Y.reshape(-1, 1), mem, n_rep=1).ravel()

    pls_null = PLSRegression(n_components=n_components)
    pls_null.fit(X, Y_null)
    Y_pred_null = pls_null.predict(X).ravel()

    null_r2[p] = r2_score(Y_null, Y_pred_null)
    null_corrs[p, :] = [pearsonr(pls_null.x_scores_[:, i], pls_null.y_scores_[:, i])[0]
                        for i in range(n_components)]

# ---- 4) Compute empirical p-values ----
p_r2 = (1 + np.sum(null_r2 >= true_r2)) / (1 + n_perm)
p_corrs = [(1 + np.sum(null_corrs[:, j] >= true_corrs[j])) / (1 + n_perm)
           for j in range(n_components)]

print({
    "true_r2": true_r2,
    "true_corrs": true_corrs,
    "p_r2": p_r2,
    "p_corrs": p_corrs
})


null_corr_array = np.array(null_corrs)  
print(null_corr_array)
final_p=[]
for i in range(n_components):
    true_corr = true_corrs[i]
    null_dist = null_corr_array[:, i]
    p_val = np.mean(null_dist >= true_corr)
    mask = null_dist >= true_corr  
    count = np.sum(mask)
    final_p.append(p_val)
    print(f"Component {i+1}: true corr={true_corr:.3f}, p={p_val:.4f}, count={count}")

# ---- 5) Significant components and genes ----
alpha = 0.01  # 1% 

significant_components = [
    i for i, p in enumerate(final_p) if p <= alpha
]

print("P-values:", final_p)
print("Significant components (1% level):", significant_components)


gene_names = list(df.columns)   
all_results = []

for comp in [0,1,4,5]:  
    weights = pls.x_weights_[:, comp]

    z_weights = zscore(weights)

    sig_idx = np.where(np.abs(z_weights) > 2.5)[0]
    
    for i in sig_idx:
        all_results.append({
            "Component": comp,
            "Gene": gene_names[i],
            "Weight": weights[i],
            "Z-score": z_weights[i]
        })

df = pd.DataFrame(all_results)
df.to_csv("significant_genes_pls.csv", index=False)

genes_by_component = df.groupby("Component")["Gene"].apply(set)


# ---- 6) Visualization ----

atlas_img = nib.load("Merged_Atlas.nii.gz")
atlas_data = atlas_img.get_fdata()
region_ids = np.unique(atlas_data)[1:]  
components_to_plot = [0, 1, 4, 5]

for comp in components_to_plot:
    weights = pls.x_weights_[:, comp]
    weights_z = (weights - np.mean(weights)) / np.std(weights)
    region_to_weight = dict(zip(region_ids, weights_z))

    mapped_data = np.vectorize(region_to_weight.get)(atlas_data)
    mapped_data = np.asarray(mapped_data, dtype=np.float32)

    # Threshold small values for visualization
    thresholded_data = np.where(np.abs(mapped_data) < 2.5, 0, mapped_data)

    # Create NIfTI image for display
    img = nib.Nifti1Image(thresholded_data, atlas_img.affine, atlas_img.header)
    nib.save(img, f"PLS_Component_{comp+1}.nii.gz")
    
    print(f"Displaying PLS Component {comp+1}")
    display = plotting.plot_stat_map(
        img,
        title=f"PLS Component {comp+1}",
        display_mode="ortho",
        threshold=2,
        colorbar=True,
        cmap="cold_hot"
    )
    plt.show()
