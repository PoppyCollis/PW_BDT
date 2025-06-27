import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

# Load data
current_dir = Path(__file__).resolve().parent
data_path = current_dir.parent / "data" / "rawChoiceData.txt"

df = pd.read_csv(data_path, sep=",")

# Number of unique participants
n_participants = df['subject'].nunique()
print(f"Number of participants: {n_participants}")

# Number of unique sessions
n_sessions = df['session'].nunique()
print(f"Number of sessions per participants: {n_sessions}")

# Number of unique sessions
n_trials = df['trial'].nunique()
print(f"Number of trials per session: {n_trials}")

print(df.columns)

