import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pw_bdt.helpers.plots import plot_shrinkage

# Load data
current_dir = Path(__file__).resolve().parent
data_path = current_dir.parent / "data" / "sensitivity_per_subject_per_session.csv"
df = pd.read_csv(data_path, sep=",")

# Preprocess
# Create a mapping from subject labels to integer indices
subjects = df['subject'].unique()
print(subjects)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
df['subject_idx'] = df['subject'].map(subject_to_idx)

# Extract values needed for the model
N_subjects = len(subjects)
subject_idx = df['subject_idx'].values
d_prime_values = df['d_prime'].values
meta_d_prime_values = df['meta_d_prime'].values


with pm.Model() as model:
    
    #------------ d prime -----------#
    #  Define Group-Level Hyperpriors
    mu = 1.0  # fixed from thresholding
    #sigma_type1 = pm.HalfNormal("sigma_type1", sigma=1.0)  # group variability
    sigma_type1 = pm.Uniform("sigma_type1", lower=0.1, upper=5.0)

    # Define Subject-Level Latent Variables
    d_subj = pm.Normal("d_subj", mu=mu, sigma=sigma_type1, shape=N_subjects)
    
    # Define the Observation Noise Model
    #sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
    sigma_obs = pm.Uniform("sigma_obs", lower=0.1, upper=5.0)
    
    # Define the Likelihood
    d_obs = pm.Normal("d_obs", mu=d_subj[subject_idx], sigma=sigma_obs, observed=d_prime_values)
    
    #------------ meta d prime -----------#
    # Hyperprior on Type 2 (meta-d′) group-level variability
    sigma_type2 = pm.Uniform("sigma_type2", lower=0.1, upper=5.0)
    
    # Latent meta-d′ per subject, linked to d′ via prior
    meta_d_subj = pm.Normal("meta_d_subj",
                            mu=0.8 * d_subj,
                            sigma=sigma_type2,
                            shape=N_subjects)

    # Likelihood for observed session-level meta-d′
    meta_d_obs = pm.Normal("meta_d_obs",
                        mu=meta_d_subj[subject_idx],
                        sigma=sigma_obs,  # same observation noise
                        observed=meta_d_prime_values)

    # Run inference
    trace = pm.sample(2000, tune=2000, chains=4, return_inferencedata=True)
    print(trace)
    save_path = current_dir.parent / "data" / "sensitivity_hierarchical_trace.nc"
    az.to_netcdf(trace, save_path)
    
    # summary = az.summary(trace, var_names=["d_subj", "sigma_type1", "sigma_obs"])
    # print(summary)
    summary = az.summary(trace, var_names=["d_subj", "meta_d_subj", "sigma_type1", "sigma_type2", "sigma_obs"])
    print(summary)

    # az.plot_posterior(trace, var_names=["d_subj"], coords={"d_subj_dim_0": [0, 1, 2]})
    az.plot_posterior(trace, var_names=["d_subj", "meta_d_subj"])
    plt.show()
    
  
    # # Plot shrinkage
    # # Compute empirical mean d′ per subject
    # empirical_means = df.groupby("subject")["d_prime"].mean().reset_index()
    # empirical_means.columns = ["subject", "empirical_d_prime"]

    # # Extract posterior means for d_subj[i]
    # posterior_means = az.summary(trace, var_names=["d_subj"])["mean"].reset_index()
    # posterior_means["subject_index"] = posterior_means["index"].str.extract(r"(\d+)").astype(int)
    # posterior_means["subject"] = posterior_means["subject_index"] + 1  # Fix the mismatch
    # posterior_means.columns = ["index", "posterior_d_prime", "subject_index", "subject"]

    # # Merge the two
    # comparison_df = pd.merge(empirical_means, posterior_means[["subject", "posterior_d_prime"]], on="subject")

    # plot_shrinkage(comparison_df)


