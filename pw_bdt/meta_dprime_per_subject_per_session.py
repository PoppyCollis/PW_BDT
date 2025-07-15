import pandas as pd
from pathlib import Path

from pw_bdt.helpers.utils import z_transform

def compute_meta_dprime(group):
    """
    Compute meta-d′ per participant per session from confidence-based pA–pD counts.
    
    Assumes 'stimulus', 'r1', and 'confidence' columns exist:
    - stimulus: 0 (left) or 1 (right)
    - r1: 0 or 1 (initial binary decision)
    - confidence: 1 (low) or 2 (high)
    """
    stim = group['stimulus']
    resp = group['r1']
    conf = group['r2']

    # Define key high-confidence signal detection outcomes
    pA = ((stim == 0) & (resp == 0) & (conf == 2)).sum()
    pB = ((stim == 1) & (resp == 0) & (conf == 2)).sum()
    pC = ((stim == 0) & (resp == 1) & (conf == 2)).sum()
    pD = ((stim == 1) & (resp == 1) & (conf == 2)).sum()

    # Add total for each condition to compute probabilities
    nCR = ((stim == 0) & (resp == 0)).sum()
    nM  = ((stim == 1) & (resp == 0)).sum()
    nFA = ((stim == 0) & (resp == 1)).sum()
    nH  = ((stim == 1) & (resp == 1)).sum()

    # Avoid div-by-zero and degenerate z-scores
    p_high_CR = pA / nCR if nCR > 0 else 0.5
    p_high_M  = pB / nM  if nM > 0 else 0.5
    p_high_FA = pC / nFA if nFA > 0 else 0.5
    p_high_H  = pD / nH  if nH > 0 else 0.5

    z_high_CR = z_transform(p_high_CR, nCR)
    z_high_M  = z_transform(p_high_M, nM)
    z_high_FA = z_transform(p_high_FA, nFA)
    z_high_H  = z_transform(p_high_H, nH)

    # Meta-d′ as z(H) - z(FA) using confidence-conditioned responses
    meta_d_prime = 0.5 * ((z_high_H - z_high_M) + (z_high_CR - z_high_FA))
    return meta_d_prime

def main():
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "rawChoiceData.txt"
    save_path_session = current_dir.parent / "data" / "meta_dprime_per_subject_per_session.csv"
    save_path_subject = current_dir.parent / "data" / "meta_dprime_per_subject.csv"

    df = pd.read_csv(data_path, sep=",")

    # Compute meta-d′ per (subject, session)
    metadprime_df = df.groupby(['subject', 'session']).apply(compute_meta_dprime).reset_index()
    metadprime_df.columns = ['subject', 'session', 'meta_d_prime']
    print("meta-d′ computed for all participants and sessions")
    print(metadprime_df)

    # Save session-level meta-d′
    metadprime_df.to_csv(save_path_session, index=False)

    # Compute and save subject-level average meta-d′
    metadprime_avg = metadprime_df.groupby("subject")["meta_d_prime"].mean().reset_index()
    metadprime_avg.columns = ["subject", "meta_d_prime_avg"]
    metadprime_avg.to_csv(save_path_subject, index=False)
    print("Average meta-d′ per subject saved")

if __name__ == "__main__":
    main()