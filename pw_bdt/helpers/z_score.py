import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

p_x = np.linspace(0.000001, 0.9999999, 1000)
y = norm.ppf(p_x)

print(norm.ppf(0))
print(norm.ppf(1))
print(norm.ppf(0.5))

plt.plot(y, p_x)
plt.title("Z-score vs Probability")
plt.xlabel("Standard Z-score")
plt.ylabel("(x)")
plt.show()