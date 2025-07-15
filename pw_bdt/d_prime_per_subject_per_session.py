import pandas as pd
from pathlib import Path

from pw_bdt.helpers.utils import z_transform

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

def main():
    
    # Load data
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "rawChoiceData.txt"
    save_path_session = current_dir.parent / "data" / "dprime_per_subject_per_session.csv"
    save_path_subject = current_dir.parent / "data" / "dprime_per_subject.csv"

    df = pd.read_csv(data_path, sep=",")

    # Compute d′ per (subject, session)
    dprime_df = df.groupby(['subject', 'session']).apply(compute_dprime).reset_index()
    dprime_df.columns = ['subject', 'session', 'd_prime']
    print("d′ computed for all participants and sessions")
    print(dprime_df) 

    # Save session-level d′
    dprime_df.to_csv(save_path_session, index=False)

    # Compute and save subject-level average d′
    dprime_avg = dprime_df.groupby("subject")["d_prime"].mean().reset_index()
    dprime_avg.columns = ["subject", "d_prime_avg"]
    dprime_avg.to_csv(save_path_subject, index=False)
    print("Average d′ per subject saved")
    
if __name__ == "__main__":
    main()
    
