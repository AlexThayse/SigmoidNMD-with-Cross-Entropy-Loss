import numpy as np
from scipy.special import expit


def sigmoid(x):
    """
    Computes the sigmoid function element-wise.
    Args:
        x (numpy array): Input array.

    Returns:
        numpy array: Sigmoid function.
    """
    return expit(x)

def fct_opti(A,b,x,epsilon=1e-12):
  """
  Computes the function value and gradient for cross-entropy optimization with sigmoid function.

  Args:
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.
      epsilon (float, optional): Small value to ensure numerical stability in log. Default: 1e-12.

  Returns:
      tuple: Function value and gradient.
  """
  z = A @ x                 
  s = np.clip(sigmoid(z), epsilon, 1 - epsilon)
  f = -np.sum(b * np.log(s) + (1 - b) * np.log(1 - s))
  g = A.T @ (s - b)
  return f, g

def w1(fct,A,b,x,d,alpha,beta1):
  """
  Checks the first Amijo-Wolfe condition.

  Args:
      fct (function): Objective function.
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.
      d (numpy.ndarray): Search direction.
      alpha (float): Step size.
      beta1 (float): Wolfe condition parameter.
  
  Returns:
      bool: True if condition is satisfied, False otherwise.
  """
  f0, g0 = fct(A,b,x)
  f1, g1 = fct(A,b,x+alpha*d)
  return f1 <= f0 + alpha*beta1*d@g0

def w2(fct,A,b,x,d,alpha,beta2):
  """
  Checks the second Wolfe condition.

  Args:
      fct (function): Objective function.
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.
      d (numpy.ndarray): Search direction.
      alpha (float): Step size.
      beta2 (float): Wolfe condition parameter.
  
  Returns:
      bool: True if condition is satisfied, False otherwise.
  """
  f0, g0 = fct(A,b,x)
  f1, g1 = fct(A,b,x+alpha*d)
  return d@g1 >= beta2*d@g0

def compute_step_length(fct,A,b,x,d,alpha,beta1=0.0001,beta2=0.9):
  """
  Performs a bisection search to find a step size that satisfies the Amijo-Wolfe conditions.

  Args:
      fct (function): Objective function.
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.
      d (numpy.ndarray): Search direction.
      alpha (float): Initial step size.
      beta1 (float, optional): Wolfe condition parameter. Default: 0.0001.
      beta2 (float, optional): Wolfe condition parameter. Default: 0.9.
  
  Returns:
      float: Step size satisfying Wolfe conditions.
  """
  aleft = 0
  aright = np.inf

  while True:
    if w1(fct,A,b,x,d,alpha,beta1) and w2(fct,A,b,x,d,alpha,beta2):
      break

    if not w1(fct,A,b,x,d,alpha,beta1):
      aright = alpha
      alpha = (aleft+aright)/2
    elif not w2(fct,A,b,x,d,alpha,beta2):
      aleft = alpha 
      if aright<np.inf:
        alpha = (aleft+aright)/2
      else : 
        alpha = 2*alpha   

  return alpha


def replace_with_nan(matrix, percentage):
    """
    Replaces a given percentage of elements in a matrix with NaN.

    Args:
        matrix (numpy.ndarray): Input matrix.
        percentage (float): Fraction of elements to replace (0 to 1).
    
    Returns:
        numpy.ndarray: Modified matrix with NaN values.
    """
    if not (0 <= percentage <= 1):
        raise ValueError("Percent between 0 et 1.")
    modified_matrix = matrix.copy()
    modified_matrix = modified_matrix.astype(float)
    total_elements = matrix.size
    num_elements_to_replace = int(total_elements * percentage)
    indices = np.random.choice(total_elements, num_elements_to_replace, replace=False)
    multi_dim_indices = np.unravel_index(indices, matrix.shape)
    modified_matrix[multi_dim_indices] = np.nan
    return modified_matrix

def cross_entropy(X, X_tilde, P, epsilon=1e-12):
    """
    Computes the binary cross-entropy loss between the true values X and predicted values X_tilde.

    Args:
        X (numpy.ndarray): Ground truth matrix.
        X_tilde (numpy.ndarray): Predicted matrix.
        P (numpy.ndarray): Binary mask (1 for observed, 0 for missing), shape (m, n).
        epsilon (float, optional): Small value to ensure numerical stability in log. Default: 1e-12.

    Returns:
        float: Total cross-entropy loss over all entries.
    """
    X_tilde = np.clip(X_tilde, epsilon, 1 - epsilon)
    CE = -(X * np.log(X_tilde) + (1 - X) * np.log(1 - X_tilde))
    return np.sum(P * CE)

def relative_cross_entropy(X, X_tilde, P):
    """
    Computes the relative cross-entropy between predicted values X_tilde and true values X,
    normalized by the cross-entropy of a naive predictor (mean of X).

    Args:
        X (numpy.ndarray): Ground truth matrix.
        X_tilde (numpy.ndarray): Predicted matrix.
        P (numpy.ndarray): Binary mask (1 for observed, 0 for missing), shape (m, n).

    Returns:
        float: Relative cross-entropy loss.
    """
    num = cross_entropy(X, X_tilde, P)
    mu = np.sum(P * X) / np.sum(P)
    naive_pred = np.full(X.shape,mu)
    den = cross_entropy(X, naive_pred, P)
    return num / den