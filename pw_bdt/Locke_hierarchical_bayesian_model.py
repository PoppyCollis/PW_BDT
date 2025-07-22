
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
current_dir = Path(__file__).resolve().parent
save_csv_path = current_dir.parent / "data" / "locke"
data_path = current_dir.parent / "data" / "sensitivity_per_subject_per_session.csv"
df = pd.read_csv(data_path)

# Preprocess subject indices
subjects = sorted(df['subject'].unique())
N_subjects = len(subjects)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
df['subject_idx'] = df['subject'].map(subject_to_idx)

# Extract observed values
subject_idx = df['subject_idx'].values

d_prime_values = df['d_prime'].values
meta_d_prime_values = df['meta_d_prime'].values


# Build and sample model
with pm.Model() as model:
    # Hyperpriors
    hyper_prior_mu_d = 1.0
    sigma_type1 = pm.Uniform('sigma_type1', lower=0.1, upper=5)
    sigma_type2 = pm.Uniform('sigma_type2', lower=0.1, upper=5)

    # Bounded subject-level parameters
    # d_subj = pm.TruncatedNormal(
    #     'd_subj', mu=hyper_prior_mu_d, sigma=sigma_type1, lower=0, upper=3,
    #     shape=N_subjects
    # )
    d_subj = pm.Normal("d_subj", mu=hyper_prior_mu_d, sigma=sigma_type1, shape=N_subjects)
    
    # meta_d_subj = pm.TruncatedNormal(
    #     'meta_d_subj', mu=0.8 * d_subj, sigma=sigma_type2, lower=0, upper=3,
    #     shape=N_subjects
    # )
    
    meta_d_subj = pm.Normal("meta_d_subj",
                            mu=0.8 * d_subj,
                            sigma=sigma_type2,
                            shape=N_subjects)
    
    

    # Subject-specific noise
    sigma_subj = pm.Uniform('sigma_subj', lower=0.1, upper=5.0, shape=N_subjects)

    # Likelihoods
    pm.Normal('d_obs', mu=d_subj[subject_idx], sigma=sigma_subj[subject_idx],
              observed=d_prime_values)
    pm.Normal('meta_d_obs', mu=meta_d_subj[subject_idx], sigma=sigma_subj[subject_idx],
              observed=meta_d_prime_values)

    # Sampling
    trace = pm.sample(2000, tune=2000, chains=4, return_inferencedata=True)

# Save trace
save_path = current_dir.parent / 'data' / 'sensitivity_hierarchical_trace.nc'
az.to_netcdf(trace, save_path)

# 4. Extract posterior summaries
summary = az.summary(trace, hdi_prob=0.95)
print("_______")
print(summary.loc['sigma_type2'])


# Identify HDI columns dynamically
hdi_cols = [c for c in summary.columns if c.startswith('hdi_')]
if len(hdi_cols) < 2:
    raise ValueError(f"No HDI columns found in summary: {summary.columns}")
# parse numeric percentiles for sorting (e.g., 'hdi_2.5%', 'hdi_97.5%', or 'hdi_3%', 'hdi_97%')
def _pct(x):
    val = x.split('_')[1].strip('%')
    return float(val)
hdi_cols_sorted = sorted(hdi_cols, key=_pct)
hdi_low, hdi_high = hdi_cols_sorted[0], hdi_cols_sorted[-1]

# Helper to extract subject-level parameters
def extract_df(varname, est_col, low_col, high_col):
    # Match exactly varname with index pattern varname[n]
    pattern = rf'^{varname}\['
    df_var = summary.loc[summary.index.str.match(pattern), ['mean', hdi_low, hdi_high]].copy()
    df_var.columns = [est_col, low_col, high_col]
    return df_var.reset_index(drop=True)

# Subject-level tables
df_d     = extract_df('d_subj',      'muEst',       'muLow95CI',     'muHigh95CI')
df_md    = extract_df('meta_d_subj', 'mu2Est',      'mu2Low95CI',    'mu2High95CI')
df_sigma = extract_df('sigma_subj',  'sigmaEst',    'sigmaLow95CI',  'sigmaHigh95CI')

# Population-level parameters
df_pop = summary.loc[['sigma_type1', 'sigma_type2'], ['mean', hdi_low, hdi_high]].copy()
df_pop = df_pop.rename(index={'sigma_type1': 'sigmaPop', 'sigma_type2': 'sigmaMCE'},
                       columns={'mean':'mean', hdi_low:'low95CI', hdi_high:'high95CI'})

# Assemble and write subject-level output
df_out = pd.concat([df_d, df_md, df_sigma], axis=1)
# Now df_out has N_subjects rows; assign sID safely
df_out['sID'] = subjects
# Reorder columns
df_out = df_out[['sID', 'muEst','muLow95CI','muHigh95CI',
                   'mu2Est','mu2Low95CI','mu2High95CI',
                   'sigmaEst','sigmaLow95CI','sigmaHigh95CI']]

# Save subject-level results
df_out.to_csv(save_csv_path / 'my_fit_dPrime_hierarchicalBayes_fitData.txt', sep='\t', index=False)
df_pop.to_csv(save_csv_path / 'my_fit_dPrime_hierarchicalBayes_population_mine.txt',
               sep='\t', index=False)