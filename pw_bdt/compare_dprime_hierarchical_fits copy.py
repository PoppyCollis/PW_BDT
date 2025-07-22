import pandas as pd
import numpy  as np
from scipy.stats import pearsonr, ttest_rel
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
root        = Path(__file__).resolve().parent.parent
my_path     = root / "data" / "locke" / "my_fit_dPrime_hierarchicalBayes_fitData.txt"
locke_path  = root / "data" / "locke" / "fit_dPrime_hierarchicalBayes_fitData2.txt"

# ── 1) Read your fits (already correct) ────────────────────────────────
df_my = pd.read_csv(my_path, sep="\t", skipinitialspace=True)
df_my.columns = df_my.columns.str.strip()
df_my = df_my.rename(columns={"sID":"sidx"})
df_my = df_my[["sidx","muEst","mu2Est"]]

# ── 2) Read Locke’s fits with whitespace delimiting ───────────────────
df_locke = pd.read_csv(
    locke_path,
    delim_whitespace=True,    # ← splits on any run of spaces
    skipinitialspace=True
)
df_locke.columns = df_locke.columns.str.strip()
# now you’ll see separate columns instead of one giant header
# confirm with: print(df_locke.columns.tolist())
df_locke["sidx"] = np.arange(1, len(df_locke)+1)
df_locke = df_locke[["sidx","muEst","mu2Est"]]

# ── 3) Merge & compare ────────────────────────────────────────────────
df = pd.merge(df_my, df_locke, on="sidx", suffixes=("_my","_locke"))
assert len(df)==10

def compare(a,b):
    r, rp  = pearsonr(df[a], df[b])
    mae    = np.abs(df[a]-df[b]).mean()
    t, tp  = ttest_rel(df[a], df[b])
    return r, rp, mae, t, tp

r_d, rp_d, mae_d, t_d, tp_d = compare("muEst_my","muEst_locke")
r_m, rp_m, mae_m, t_m, tp_m = compare("mu2Est_my","mu2Est_locke")

print("d′:    r",round(r_d,3), "MAE",round(mae_d,3), "t",round(t_d,2),"p",tp_d)
print("meta-d′: r",round(r_m,3), "MAE",round(mae_m,3), "t",round(t_m,2),"p",tp_m)


# ── 4. quick stats function ────────────────────────────────────────────
def compare(col1, col2):
    r, rp     = pearsonr(df[col1], df[col2])
    mae       = np.abs(df[col1] - df[col2]).mean()
    t, tp     = ttest_rel(df[col1], df[col2])
    return r, rp, mae, t, tp

print("\n––– Comparison: d′ –––")
print(f" Pearson r      = {r_d:.3f}  (p = {rp_d:.3g})")
print(f" Mean abs error = {mae_d:.3f}")
print(f" Paired t‐test  = t {t_d:.2f}, p {tp_d:.3g}")

print("\n––– Comparison: meta-d′ –––")
print(f" Pearson r      = {r_m:.3f}  (p = {rp_m:.3g})")
print(f" Mean abs error = {mae_m:.3f}")
print(f" Paired t‐test  = t {t_m:.2f}, p {tp_m:.3g}\n")

# ── 5. scatter‐plots ──────────────────────────────────────────────────
for mycol, locke_col, lab in [
    ("muEst_my",  "muEst_locke",  "d′"),
    ("mu2Est_my", "mu2Est_locke", "meta-d′")]:
    fig, ax = plt.subplots()
    ax.scatter(df[locke_col], df[mycol], alpha=0.8)
    mn = min(ax.get_xlim()[0], ax.get_ylim()[0])
    mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([mn,mx],[mn,mx], "--", linewidth=1)
    ax.set(
      xlabel=f"Locke {lab}",
      ylabel=f"My {lab}",
      title=f"{lab} comparison: R = {pearsonr(df[mycol],df[locke_col])[0]:.2f}, "
            f"MAE = {np.abs(df[mycol]-df[locke_col]).mean():.2f}"
    )
    plt.tight_layout()
    plt.show()

