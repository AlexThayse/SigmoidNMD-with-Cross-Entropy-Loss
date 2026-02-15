# Sigmoid Nonlinear Matrix Decomposition with Cross Entropy

This code provides an algorithm to solve the following **Nonlinear Matrix Decomposition (NMD)** problem with the **cross-entropy (CE) loss**:

Given a matrix $X \in [0,1]^{m \times n}$ and an integer $r$, solve

$$\min_{W,H} \mathcal{L}_{CE}(X,\sigma(WH)),$$

where $W \in \mathbb{R}^{m \times r}$ and $H \in \mathbb{R}^{r \times n}$. The sigmoid function $\sigma(\cdot)$ is applied component-wise on $WH$ and is defined by:

$$f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

The objective function $\mathcal{L}_{CE}$ is the cross-entropy loss, defined by:

$$\mathcal{L}_{CE}(X, \tilde{X}) = \sum_{i,j} -X_{i,j}\ln(\tilde{X}_{i,j}) - (1 - X_{i,j})\ln(1 - \tilde{X}_{i,j}),$$

where $\tilde{X} = \sigma(WH)$.

You can run main.py for a simple example on synthetic data.
