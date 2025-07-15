import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
current_dir = Path(__file__).resolve().parent
data_path = current_dir.parent / "data" / "dprime_per_subject_per_session.csv"
df = pd.read_csv(data_path, sep=",")


# Preprocess
# Create a mapping from subject labels to integer indices
subjects = df['subject'].unique()
subject_to_idx = {s: i for i, s in enumerate(subjects)}
df['subject_idx'] = df['subject'].map(subject_to_idx)

# Extract values needed for the model
N_subjects = len(subjects)
subject_idx = df['subject_idx'].values
d_prime_values = df['d_prime'].values

with pm.Model() as model:
    
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

    # Run inference
    trace = pm.sample(1000, tune=1000, chains=4, return_inferencedata=True)
    print(trace)
    save_path = current_dir.parent / "data" / "dprime_hierarchical_trace.nc"
    az.to_netcdf(trace, save_path)

    
    summary = az.summary(trace, var_names=["d_subj", "sigma_type1", "sigma_obs"])
    print(summary)
    az.plot_posterior(trace, var_names=["d_subj"], coords={"d_subj_dim_0": [0, 1, 2]})
    plt.show()

    