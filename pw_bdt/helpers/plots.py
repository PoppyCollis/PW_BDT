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
  

def plot_dprime_per_sub_per_session(df):  
    # Define custom colors for specific subjects
    subject_colors = {
        3: 'lightblue',
        9: 'mediumaquamarine',  # mint green
        4: 'limegreen',
        7: 'darkkhaki',         # ochre
        10: 'orange',
        5: 'lightcoral',        # light red
        1: 'mediumpurple',
        2: 'magenta', 
        6: 'pink',
        8: 'turquoise'
    }
    #default_color = 'lightgrey'

    plt.figure(figsize=(10, 6))

    for subject_id, group in df.groupby("subject"):
        color = subject_colors.get(subject_id)
        plt.plot(group["session"], group["d_prime"], marker="o", label=f"Subject {subject_id}", color=color)

    plt.xlabel("Session Number")
    plt.ylabel("d′ (Type 1 Sensitivity)")
    plt.ylim(0.5,3)
    plt.title("d′ Over Sessions by Subject")
    plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
