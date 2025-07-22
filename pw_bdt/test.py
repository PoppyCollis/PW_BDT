import matplotlib.pyplot as plt
import numpy as np 


def get_k1(a_pv, a_p, a_v, k_p, k_v):
    
    return a_pv*((a_p*k_p) + (a_v*k_v))



k_v = -1.1
k_p = 1.2

# # 2 alphas
# a_pv = 1 # null
# a_p = 0.8
# a_v = 0.5

# alpha2_k1 = get_k1(a_pv, a_p, a_v, k_p, k_v)

# 3 alpha
alpha_pv = 1#0.9
alpha_p = 0.8
alpha_v = 0.5

alpha3_k1 = get_k1(alpha_pv, alpha_p, alpha_v, k_p, k_v)


print(alpha3_k1)