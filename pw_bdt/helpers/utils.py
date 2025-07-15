from scipy.stats import norm

def z_transform(p, n, correction=0.5):
    # Prevent z-transform from returning inf/-inf
    # Macmillan & Kaplan, 1985 correction:
    # Detection Theory Analysis of Group Data: Estimating Sensitivity From Average Hit and False-Alarm Rates
    if p == 1.0:
        p = (n - correction) / n
    elif p == 0.0:
        p = correction / n
    return norm.ppf(p)