import pymc as pm
import numpy as np
from scipy.stats import norm



"""
    
    2 x 2 cont table per subject per sess
    
    count tbale is a contingency table of stim x response  with counts in each bin
    
    This is N_{res, stim}
    
    Group the data by participant and session.

Tabulate the counts for (stimulus, r1).

Model the likelihood for all those counts, given:

Shared parameters per participant

Session-specific priors/payoffs as input to the model

Optimise
    
"""


def binomial_likelihood():
    
    pass

def joint_distribution_r_s(stim, alpha, gamma, d_prime, pR_pL, vR_vL):
    
    # optimal likelihood ratio
    p_r = pR_pL / 1 + pR_pL
    p_l = 1 - p_r
    
    v_r = vR_vL
    v_l = 1
    
    ln_beta_opt = np.log(p_l/p_r) + np.log(v_l/v_r)
    
    # Convert to optimal criterion
    k_opt = 1/d_prime * ln_beta_opt
    
    # apply conservatism
    k_1 = alpha * k_opt + gamma
    
    # Compute decision probabilities
    if stim == 0:  # Left stimulus
        p_resp_L = norm.cdf(k1 + dprime/2)
        return {0: p_resp_L, 1: 1 - p_resp_L}
    else:  # Right stimulus
        p_resp_R = 1 - norm.cdf(k1 - dprime/2)
        return {1: p_resp_R, 0: 1 - p_resp_R}
    

with pm.Model() as model:
    theta = pm.Uniform('theta', lower=0, upper=1)  # flat prior
    likelihood = pm.Binomial('obs', n=n, p=theta, observed=k)
    
    # MAP estimate (corresponds to ML if prior is uniform)
    map_estimate = pm.find_MAP()