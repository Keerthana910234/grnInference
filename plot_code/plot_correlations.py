#%%
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import os
import subprocess
os.system("module load texlive")

os.environ['PATH'] = "/software/texlive/2020/bin/x86_64-linux:" + os.environ['PATH']

matplotlib.rcParams['text.usetex'] = True
#Set all tick label fonts to be 16 and labels to be size 20 
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16

param_df = pd.read_csv('/home/mzo5929/Keerthana/grnInference/simulationData/parameter_sweep.csv', index_col=0)

#%%
path_to_correlations = "/home/mzo5929/Keerthana/grnInference/analysisData2/"
correlation_regulation = pd.read_csv(f"{path_to_correlations}/correlation_k_on_tf.csv")
correlation_no_regulation = pd.read_csv(f"{path_to_correlations}/correlation_df_k_on_tf_no_regulation.csv")

#%%
#Aggregate the dataframesbased on k_on_TF value by taking the mean and std of the 3 correlation columns
def aggregate_correlation(correlation_df):
    # Group by 'k_on_TF' and aggregate the correlation columns
    agg_df = correlation_df.groupby('k_on_TF').agg({
        "t1_gene_gene_correlation": ['mean', 'std'],
        "t1_twin_pair_correlation": ['mean', 'std'],
        "t1_random_pair_correlation": ['mean', 'std']
    }).reset_index()
    return agg_df
correlation_regulation_agg = aggregate_correlation(correlation_regulation)
correlation_no_regulation_agg = aggregate_correlation(correlation_no_regulation)

#%%
#Plot k_on_TF vs t1_gene_gene_correlation for both regulation and no regulation
plt.figure(figsize=(10, 6))
plt.errorbar(correlation_regulation_agg['k_on_TF'], 
             correlation_regulation_agg['t1_gene_gene_correlation']['mean'], 
             yerr=correlation_regulation_agg['t1_gene_gene_correlation']['std'], 
             label='Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_regulation_agg['k_on_TF'],
                    correlation_regulation_agg['t1_gene_gene_correlation']['mean'] - correlation_regulation_agg['t1_gene_gene_correlation']['std'],
                    correlation_regulation_agg['t1_gene_gene_correlation']['mean'] + correlation_regulation_agg['t1_gene_gene_correlation']['std'],
                    alpha=0.2)
plt.errorbar(correlation_no_regulation_agg['k_on_TF'],
                correlation_no_regulation_agg['t1_gene_gene_correlation']['mean'], 
                yerr=correlation_no_regulation_agg['t1_gene_gene_correlation']['std'], 
                label='No Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_no_regulation_agg['k_on_TF'],
                    correlation_no_regulation_agg['t1_gene_gene_correlation']['mean'] - correlation_no_regulation_agg['t1_gene_gene_correlation']['std'],
                    correlation_no_regulation_agg['t1_gene_gene_correlation']['mean'] + correlation_no_regulation_agg['t1_gene_gene_correlation']['std'],
                    alpha=0.2)
plt.ylim(-0.01, 0.2)
# plt.axhline(y=1e-3, color='black', linestyle='--', label=r'$\rho_{baseline} = 0.001$')
plt.axhline(y=0, color='black')
plt.yticks(np.arange(0, 0.21, 0.05))
plt.xlabel('$k_{on}^{A}$', fontsize=24)
plt.ylabel(r'$\rho$', fontsize=24)
plt.legend()
#save the figure
# plt.savefig('/home/mzo5929/Keerthana/grnInference/plots/plots_present/gene_gene_correlation_k_on_TF.png', bbox_inches='tight', dpi=300)
plt.show()


# %%
#Now for the mixed population case
mixed_corr_df = pd.read_csv("/home/mzo5929/Keerthana/grnInference/analysisData2/correlation_mixed_population_k_on_tf_no_regulation_with_lower_quartile_burst_param.csv")
#Aggregate the dataframes based on k_on_TF value by taking the mean and std of the 3 correlation columns

mixed_corr_agg = aggregate_correlation(mixed_corr_df)
#%%
#Plot the 3 cases together - regulation, no regulation and mixed population
plt.figure(figsize=(10, 6))
plt.errorbar(correlation_regulation_agg['k_on_TF'], 
             correlation_regulation_agg['t1_gene_gene_correlation']['mean'], 
             yerr=correlation_regulation_agg['t1_gene_gene_correlation']['std'], 
             label='Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_regulation_agg['k_on_TF'],
                    correlation_regulation_agg['t1_gene_gene_correlation']['mean'] - correlation_regulation_agg['t1_gene_gene_correlation']['std'],
                    correlation_regulation_agg['t1_gene_gene_correlation']['mean'] + correlation_regulation_agg['t1_gene_gene_correlation']['std'],
                    alpha=0.2)
plt.errorbar(correlation_no_regulation_agg['k_on_TF'],
                correlation_no_regulation_agg['t1_gene_gene_correlation']['mean'], 
                yerr=correlation_no_regulation_agg['t1_gene_gene_correlation']['std'], 
                label='No Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_no_regulation_agg['k_on_TF'],
                    correlation_no_regulation_agg['t1_gene_gene_correlation']['mean'] - correlation_no_regulation_agg['t1_gene_gene_correlation']['std'],
                    correlation_no_regulation_agg['t1_gene_gene_correlation']['mean'] + correlation_no_regulation_agg['t1_gene_gene_correlation']['std'],
                    alpha=0.2)
plt.errorbar(mixed_corr_agg['k_on_TF'],
                mixed_corr_agg['t1_gene_gene_correlation']['mean'], 
                yerr=mixed_corr_agg['t1_gene_gene_correlation']['std'], 
                label='2 States', fmt='o-', capsize=5)
plt.fill_between(mixed_corr_agg['k_on_TF'],
                    mixed_corr_agg['t1_gene_gene_correlation']['mean'] - mixed_corr_agg['t1_gene_gene_correlation']['std'],
                    mixed_corr_agg['t1_gene_gene_correlation']['mean'] + mixed_corr_agg['t1_gene_gene_correlation']['std'],
                    alpha=0.2)
plt.ylim(-0.02, 0.3)
# plt.axhline(y=1e-3, color='black', linestyle='--', label=r'$\rho_{baseline} = 0.001$')
# plt.axhline(y=-1*1e-3, color='black', linestyle='--')
plt.axhline(y=0, color='black')
plt.yticks(np.arange(0, 0.31, 0.05))
plt.xlabel('$k_{on}^{A}$', fontsize=24)
plt.ylabel(r'$\rho$', fontsize=24)
plt.legend()
#save the figure
# plt.savefig('/home/mzo5929/Keerthana/grnInference/plots/plots_present/gene_gene_correlation_k_on_TF_all_3.png', bbox_inches='tight', dpi=300)
plt.show()


# %%
#Plotting just the mixed population case
plt.figure(figsize=(10, 6))
plt.errorbar(mixed_corr_agg['k_on_TF'],
                mixed_corr_agg['t1_gene_gene_correlation']['mean'], 
                yerr=mixed_corr_agg['t1_gene_gene_correlation']['std'], 
                label='2 states', fmt='o-', capsize=5, color='#2ca02c')
plt.fill_between(mixed_corr_agg['k_on_TF'],
                    mixed_corr_agg['t1_gene_gene_correlation']['mean'] - mixed_corr_agg['t1_gene_gene_correlation']['std'],
                    mixed_corr_agg['t1_gene_gene_correlation']['mean'] + mixed_corr_agg['t1_gene_gene_correlation']['std'],
                    alpha=0.2, color='#2ca02c')
plt.ylim(-0.02, 0.3)
# plt.axhline(y=1e-3, color='black', linestyle='--', label=r'$\rho_{baseline} = 0.001$')
# plt.axhline(y=-1*1e-3, color='black', linestyle='--')
plt.axhline(y=0, color='black')
plt.axvline(x= 0.47, color='black', linestyle='--', label=r'$k_{on}^{A}$ for state 1')
plt.yticks(np.arange(0, 0.31, 0.05))
plt.xlabel('$k_{on}^{A}$ for state 2', fontsize=24)
plt.ylabel(r'$\rho$', fontsize=24)
plt.legend(loc = 'upper right')
#save the figure
# plt.savefig('/home/mzo5929/Keerthana/grnInference/plots/plots_present/gene_gene_correlation_k_on_TF_mixed_population.png', bbox_inches='tight', dpi=300)
plt.show()

#%%
#Similar plots for twin pair correlaions
#Plot the 3 cases together - regulation, no regulation and mixed population
plt.figure(figsize=(10, 6))
plt.errorbar(correlation_regulation_agg['k_on_TF'], 
             correlation_regulation_agg['t1_twin_pair_correlation']['mean'], 
             yerr=correlation_regulation_agg['t1_twin_pair_correlation']['std']/2, 
             label='Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_regulation_agg['k_on_TF'],
                    correlation_regulation_agg['t1_twin_pair_correlation']['mean'] - correlation_regulation_agg['t1_twin_pair_correlation']['std']/2,
                    correlation_regulation_agg['t1_twin_pair_correlation']['mean'] + correlation_regulation_agg['t1_twin_pair_correlation']['std']/2,
                    alpha=0.2)
plt.errorbar(correlation_no_regulation_agg['k_on_TF'],
                correlation_no_regulation_agg['t1_twin_pair_correlation']['mean'], 
                yerr=correlation_no_regulation_agg['t1_twin_pair_correlation']['std'], 
                label='No Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_no_regulation_agg['k_on_TF'],
                    correlation_no_regulation_agg['t1_twin_pair_correlation']['mean'] - correlation_no_regulation_agg['t1_twin_pair_correlation']['std']/2,
                    correlation_no_regulation_agg['t1_twin_pair_correlation']['mean'] + correlation_no_regulation_agg['t1_twin_pair_correlation']['std']/2,
                    alpha=0.6)
plt.errorbar(mixed_corr_agg['k_on_TF'],
                mixed_corr_agg['t1_twin_pair_correlation']['mean'], 
                yerr=mixed_corr_agg['t1_twin_pair_correlation']['std'], 
                label='2 States', fmt='o-', capsize=5)
plt.fill_between(mixed_corr_agg['k_on_TF'],
                    mixed_corr_agg['t1_twin_pair_correlation']['mean'] - mixed_corr_agg['t1_twin_pair_correlation']['std']/2,
                    mixed_corr_agg['t1_twin_pair_correlation']['mean'] + mixed_corr_agg['t1_twin_pair_correlation']['std']/2,
                    alpha=0.2)
plt.ylim(-0.0, 0.2)
plt.yticks(np.arange(0, 0.21, 0.05))
# plt.axhline(y=1e-3, color='black', linestyle='--', label=r'$\rho_{baseline} = 0.001$')
# plt.axhline(y=-1*1e-3, color='black', linestyle='--')
plt.axhline(y=0, color='black')
plt.xlabel('$k_{on}^{A}$', fontsize=24)
plt.ylabel(r'$\hat{\rho}_{\Delta}$', fontsize=28)
plt.legend()
#save the figure
# plt.savefig('/home/mzo5929/Keerthana/grnInference/plots/plots_present/twin_pair_correlation_k_on_all_3.png', bbox_inches='tight', dpi=300)
plt.show()

#%%
#Same plot for random pairs
plt.figure(figsize=(10, 6))
plt.errorbar(correlation_regulation_agg['k_on_TF'], 
             correlation_regulation_agg['t1_random_pair_correlation']['mean'], 
             yerr=correlation_regulation_agg['t1_random_pair_correlation']['std'], 
             label='Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_regulation_agg['k_on_TF'],
                    correlation_regulation_agg['t1_random_pair_correlation']['mean'] - correlation_regulation_agg['t1_random_pair_correlation']['std'],
                    correlation_regulation_agg['t1_random_pair_correlation']['mean'] + correlation_regulation_agg['t1_random_pair_correlation']['std'],
                    alpha=0.2)
plt.errorbar(correlation_no_regulation_agg['k_on_TF'],
                correlation_no_regulation_agg['t1_random_pair_correlation']['mean'], 
                yerr=correlation_no_regulation_agg['t1_random_pair_correlation']['std'], 
                label='No Regulation', fmt='o-', capsize=5)
plt.fill_between(correlation_no_regulation_agg['k_on_TF'],
                    correlation_no_regulation_agg['t1_random_pair_correlation']['mean'] - correlation_no_regulation_agg['t1_random_pair_correlation']['std'],
                    correlation_no_regulation_agg['t1_random_pair_correlation']['mean'] + correlation_no_regulation_agg['t1_random_pair_correlation']['std'],
                    alpha=0.6)
plt.errorbar(mixed_corr_agg['k_on_TF'],
                mixed_corr_agg['t1_random_pair_correlation']['mean'], 
                yerr=mixed_corr_agg['t1_random_pair_correlation']['std'], 
                label='2 states', fmt='o-', capsize=5)
plt.fill_between(mixed_corr_agg['k_on_TF'],
                    mixed_corr_agg['t1_random_pair_correlation']['mean'] - mixed_corr_agg['t1_random_pair_correlation']['std'],
                    mixed_corr_agg['t1_random_pair_correlation']['mean'] + mixed_corr_agg['t1_random_pair_correlation']['std'],
                    alpha=0.2)
plt.ylim(-0.01, 0.2)
# plt.axhline(y=1e-3, color='black', linestyle='--')
# plt.axhline(y=-1*1e-3, color='black', linestyle='--')
plt.axhline(y=0, color='black')
plt.xlabel('$k_{on}^{A}$', fontsize=24)
plt.ylabel(r'$\rho_{\Delta}$', fontsize=28)
plt.yticks(np.arange(0, 0.21, 0.05))
plt.legend()
#save the figure
# plt.savefig('/home/mzo5929/Keerthana/grnInference/plots/plots_present/random_pair_correlation_k_on_TF_all3.png', bbox_inches='tight', dpi=300)
plt.show()
#
#%%
#Plot A vs B correlation
A = np.random.lognormal(mean=0, sigma=1, size=2000)
# A = np.concatenate((A_1, A_2))
B = np.random.lognormal(mean=0, sigma=1, size=2000) + abs(np.random.normal(loc=0, scale=0.5, size=2000))
corr = spearmanr(A, B)
plt.figure(figsize=(10, 6))
plt.scatter(A, B, alpha=0.5)
plt.xlabel(r'$B_{t_1}$', fontsize=32)
plt.ylabel(r'$A_{t_2}$', fontsize=32)
# if corr and corr.correlation is not None:
#     plt.title(rf'$\bf{{\hat{{\rho}}_{{\Delta}}}}$ =  {corr.correlation:.2e}', fontsize=28)
# else:
#     plt.title(r'$\hat{\rho}_\Delta $: Not Available', fontsize=24)
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.xticks(np.arange(0, 21, 5), fontsize=24)
plt.yticks(np.arange(0, 21, 5), fontsize=24)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()