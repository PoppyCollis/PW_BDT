import pandas as pd
from pathlib import Path
from pw_bdt.helpers.utils import z_transform
from pw_bdt.helpers.plots import plot_dprime_per_sub_per_session

def compute_dprime(group):
    """
        Function to compute d' per ppt per session
    """
    # Right = signal, Left = noise
    stim = group['stimulus']
    resp = group['r1']
    
    # Hits: stimulus == 1 and response == 1
    hits = ((stim == 1) & (resp == 1)).sum()
    # Rightward stimuli total
    n_signal = (stim == 1).sum()

    # False alarms: stimulus == 0 and response == 1
    false_alarms = ((stim == 0) & (resp == 1)).sum()
    # Leftward stimuli total
    n_noise = (stim == 0).sum()

    # Handle edge cases (avoid div by 0 or z(0/1))
    # assumes chance level responses if no signal or no noise
    p_hit = hits / n_signal if n_signal > 0 else 0.5 
    p_fa = false_alarms / n_noise if n_noise > 0 else 0.5

    z_hit = z_transform(p_hit, n_signal)
    z_fa = z_transform(p_fa, n_noise)

    return z_hit - z_fa

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
    pA = ((stim == 0) & (resp == 0) & (conf == 1)).sum() # High Conf | Correct rejection
    pB = ((stim == 1) & (resp == 0) & (conf == 1)).sum() # High Conf | Miss
    pC = ((stim == 0) & (resp == 1) & (conf == 1)).sum() # High Conf | False Alarm
    pD = ((stim == 1) & (resp == 1) & (conf == 1)).sum() # High Conf | Hit
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

    z_high_CR = z_transform(p_high_CR, nCR) # A
    z_high_M  = z_transform(p_high_M, nM) # B
    z_high_FA = z_transform(p_high_FA, nFA) # C
    z_high_H  = z_transform(p_high_H, nH) # D

    # Meta-d′ as z(H) - z(FA) using confidence-conditioned responses
    meta_d_prime = 0.5 * ((z_high_CR - z_high_M) + (z_high_H - z_high_FA))
    return meta_d_prime
    
def main():
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "rawChoiceData.txt"
    save_path_session = current_dir.parent / "data" / "sensitivity_per_subject_per_session.csv"
    save_path_subject = current_dir.parent / "data" / "sensitivity_per_subject.csv"

    df = pd.read_csv(data_path, sep=",")

    # Per session
    result = df.groupby(['subject', 'session']).apply(
        lambda g: pd.Series({
            'd_prime': compute_dprime(g),
            'meta_d_prime': compute_meta_dprime(g)
        })
    ).reset_index()

    # Compute M-ratio
    result['m_ratio'] = result['meta_d_prime'] / result['d_prime']

    print(result)
    result.to_csv(save_path_session, index=False)
    print("Saved session-level data to", save_path_session)

    # Per subject (average across sessions)
    result_avg = result.groupby("subject")[["d_prime", "meta_d_prime", "m_ratio"]].mean().reset_index()
    result_avg.to_csv(save_path_subject, index=False)
    print("Saved subject-level averages to", save_path_subject)
    
    df2 = pd.read_csv(save_path_session, sep=",")
    
    plot_dprime_per_sub_per_session(df2)


if __name__ == "__main__":
    main()