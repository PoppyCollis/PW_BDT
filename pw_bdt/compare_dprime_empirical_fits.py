import pandas as pd
import numpy  as np
from scipy.stats import pearsonr, ttest_rel
import matplotlib.pyplot as plt
from pathlib import Path


# Paths
current_dir      = Path(__file__).resolve().parent
my_data_path     = current_dir.parent / "data" / "sensitivity_per_subject_per_session.csv"
locke_data_path  = current_dir.parent / "data" / "locke" / "fit_dPrime_hierarchicalBayes_rawData.txt"

# Your CSV
df_my = pd.read_csv(my_data_path)

# Locke text file (space-delimited)
df_locke = pd.read_csv(
    locke_data_path,
    sep=r'\s+',            # regex: one‐or‐more spaces/tabs
    engine='python'        # needed when using regex separators
    # header='infer'       # default; remove or set header=None if no header line
)
df_locke = df_locke.rename(columns={"sidx": "subject"})
df_locke["session"] = df_locke.groupby("subject").cumcount() + 1   # 1-7 within each subject
df_locke = df_locke[["subject", "session", "dPrime", "metadPrime"]]

# ---------- 4. inner-join on subject×session -------------------
df = pd.merge(df_my,
              df_locke,
              on=["subject", "session"],
              how="inner",
              suffixes=("_my", "_locke"))

assert len(df) == 70, "Something mismatched—expecting 70 rows (10×7)."


# ---------- 5. headline stats ----------------------------------
def quick_stats(col_my, col_locke):
    r,  rp   = pearsonr(df[col_my], df[col_locke])
    mae      = np.abs(df[col_my] - df[col_locke]).mean()
    t, tp    = ttest_rel(df[col_my], df[col_locke])
    return r, rp, mae, t, tp

r_d,  rp_d,  mae_d,  t_d,  tp_d  = quick_stats("d_prime",        "dPrime")
r_m,  rp_m,  mae_m,  t_m,  tp_m  = quick_stats("meta_d_prime",   "metadPrime")

print("\n–––  Comparison: d′  –––")
print(f"Pearson r       = {r_d:.3f}  (p = {rp_d:.3g})")
print(f"Mean abs error  = {mae_d:.3f}")
print(f"Paired t-test   t = {t_d:.2f},  p = {tp_d:.3g}")

print("\n–––  Comparison: meta-d′  –––")
print(f"Pearson r       = {r_m:.3f}  (p = {rp_m:.3g})")
print(f"Mean abs error  = {mae_m:.3f}")
print(f"Paired t-test   t = {t_m:.2f},  p = {tp_m:.3g}")

# ---------- 6. plots -------------------------------------------
# One plot per figure (guideline-friendly)
for col_my, col_locke, label in [
        ("d_prime",      "dPrime",     "d′"),
        ("meta_d_prime", "metadPrime", "meta-d′")]:
    fig, ax = plt.subplots()
    ax.scatter(df[col_locke], df[col_my], alpha=0.8)
    # identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"Locke {label}")
    ax.set_ylabel(f"My {label}")
    ax.set_title(f"{label} comparison\n$R$ = {pearsonr(df[col_my], df[col_locke])[0]:.2f},  MAE = {np.abs(df[col_my] - df[col_locke]).mean():.2f}")
    plt.tight_layout()
    plt.show()
