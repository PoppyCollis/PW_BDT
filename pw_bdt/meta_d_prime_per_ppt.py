import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path

# Load data
current_dir = Path(__file__).resolve().parent
data_path = current_dir.parent / "data" / "rawChoiceData.txt"
save_path = current_dir.parent / "data" / "meta_dprime_per_ppt_per_session.csv"
df = pd.read_csv(data_path, sep=",")

# Make sure confidence and correctness columns are present
df["correct"] = (df["stimulus"] == df["r1"]).astype(int)

# Store results
results = []
skipped = []

# Iterate over subjects and sessions
for (subj, sess), group in df.groupby(["subject", "session"]):
    # Define confidence-conditioned counts
    high_conf = group[group["r2"] == 1] 
    low_conf = group[group["r2"] == 0]

    # Hit = correct high conf when stimulus == 1 (right):
    # get all true right stimulus trials in high confidence list and check if subject was correct (chose right)
    hits = ((high_conf["stimulus"] == 1) & (high_conf["correct"] == 1)).sum() 
    # get all true right stimulus trials in low confidence list and check if subject was incorrect (chose left)
    misses = ((low_conf["stimulus"] == 1) & (low_conf["correct"] == 0)).sum()

    # Correct Rejections = correct high conf when stimulus == 0 (left)
    # get all true left stimulus trials in high confidence list and check if subject was correct (chose left)
    crs = ((high_conf["stimulus"] == 0) & (high_conf["correct"] == 1)).sum()
    # get all true left stimulus trials in low confidence list and check if subject was incorrect (chose right)
    fas = ((low_conf["stimulus"] == 0) & (low_conf["correct"] == 0)).sum()

    total_HM = hits + misses 
    total_CRFA = crs + fas # correct rejections + false alarms

    # Avoid division by zero
    if total_HM == 0 or total_CRFA == 0:
        skipped.append({"subject": subj, "session": sess, "reason": "zero trials in required condition"})
        continue

    with pm.Model() as model:
        # Prior over meta-d'
        meta_d = pm.Normal("meta_d", mu=0, sigma=3)

        # Decision criterion
        c = pm.Normal("c", mu=0, sigma=2)

        # Predicted hit and CR rates
        zH = (meta_d / 2) - c
        zFA = (-meta_d / 2) - c

        pH = 1 - pm.math.phi(zH)
        pFA = 1 - pm.math.phi(zFA)

        # Binomial likelihoods
        pm.Binomial("hits", n=total_HM, p=pH, observed=hits)
        pm.Binomial("crs", n=total_CRFA, p=pFA, observed=crs)

        # Inference
        trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9, progressbar=False)

    # Posterior mean of meta-d'
    meta_d_mean = trace.posterior["meta_d"].mean().item()

    results.append({
        "subject": subj,
        "session": sess,
        "meta_d_prime": meta_d_mean
    })

# Save to CSV

if skipped:
    skipped_df = pd.DataFrame(skipped)
    skipped_df.to_csv("skipped_meta_d_prime_fits.csv", index=False)
    print("Some subject-session pairs were skipped due to insufficient data. See 'skipped_meta_d_prime_fits.csv'.")
else:
    print("No subject-session pairs were skipped.")

meta_d_df = pd.DataFrame(results)
meta_d_df.to_csv("meta_d_prime_per_ppt.csv", index=False)

print(meta_d_df.head())
