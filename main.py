import numpy as np
from utils import sigmoid
import matplotlib.pyplot as plt
from SigmoidNMD_CrossEntropyLoss import sigma_NMD_CE

# Define matrix dimensions and rank
m=10
n=10
r=2

# Generate synthetic data for testing
np.random.seed(42)
Wt = np.random.rand(m, r)
Ht = np.random.rand(r, n)
X = sigmoid(Wt @ Ht)

# Perform sigma-NMD-CE
W, H, relative_errors, error_rates, _, rmse, mae, l1_relative_errors, times = sigma_NMD_CE(X, r, init="random", method="lipschitz_step", use_extrapolation=True)

# Plot the error rate over iterations
plt.figure()
plt.semilogy(l1_relative_errors)
plt.xlabel("Iterations")
plt.ylabel("L1 Relative Error")
plt.show()