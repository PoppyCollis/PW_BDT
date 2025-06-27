from scipy.stats import norm


def z_transform(p, n, correction=0.5):
    # Prevent z-transform from returning inf/-inf
    if p == 1.0:
        p = 1 - correction / n
    elif p == 0.0:
        p = correction / n
    return norm.ppf(p)