import matplotlib.pyplot as plt

def plot_shrinkage(comparison_df):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 2], [0, 2], "--", color="gray", label="No shrinkage (y = x)")
    plt.scatter(comparison_df["empirical_d_prime"], comparison_df["posterior_d_prime"], color="blue")

    # Add subject labels
    for _, row in comparison_df.iterrows():
        plt.annotate(str(row["subject"]),
                    (row["empirical_d_prime"], row["posterior_d_prime"]),
                    textcoords="offset points", xytext=(5, 2), ha='left', fontsize=8)

    plt.xlabel("Empirical Mean d′ per Subject")
    plt.ylabel("Posterior Mean d′ (from model)")
    plt.title("Shrinkage Effect: Posterior vs. Empirical d′")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()