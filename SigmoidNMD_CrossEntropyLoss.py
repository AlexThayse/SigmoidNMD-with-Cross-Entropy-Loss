import numpy as np
import time
from utils import compute_step_length, fct_opti, sigmoid, relative_cross_entropy
from WLRA import WLRA

def updateFact(X, W, H, learning_rates, method, P):
    """
    This function updates the matrix H using a block coordinate descent method.
    Given an input matrix `X` of shape (m, n), a matrix `W` of shape (m, r),
    a matrix `H` of shape (r, n), it updates each column of H using cross-entropy optimization.

    Args:
        X (numpy.ndarray): Input matrix of shape (m, n).
        W (numpy.ndarray): Matrix W of shape (m, r).
        H (numpy.ndarray): Matrix H of shape (r, n).
        learning_rates (numpy.ndarray): Learning rates for each column of H.
        method (str, optional): Line search method to use.
        P (numpy.ndarray): Binary mask (1 for observed, 0 for missing), shape (m, n).

    Returns:
        H (numpy.ndarray): Updated matrix H of shape (r, n).
        learning_rates (numpy.ndarray): Updated learning rates for each column of H.

    """

    n = X.shape[1]

    for j in range(n):
        H[:, j], learning_rates[j] = sigmoid_cross_entropy(A=W, b=X[:, j], x=H[:, j], alpha=learning_rates[j], method=method, P=P[:, j])

    return H, learning_rates

def updateFact_momentum(X, W, H, learning_rates, method, P, theta):
    """
    This function updates the matrix H using a block coordinate descent method
    enhanced with Nesterov's Accelerated Gradient (NAG).
    Given an input matrix `X` of shape (m, n), a matrix `W` of shape (m, r),
    a matrix `H` of shape (r, n), it updates each column of H using cross-entropy optimization.

    Args:
        X (numpy.ndarray): Input matrix of shape (m, n).
        W (numpy.ndarray): Matrix W of shape (m, r).
        H (numpy.ndarray): Matrix H of shape (r, n).
        learning_rates (numpy.ndarray): Learning rates for each column of H.
        method (str, optional): Line search method to use.
        P (numpy.ndarray): Binary mask (1 for observed, 0 for missing), shape (m, n).
        theta (float): Momentum parameter.

    Returns:
        H (numpy.ndarray): Updated matrix H of shape (r, n).
        learning_rates (numpy.ndarray): Updated learning rates for each column of H.
        H_old (numpy.ndarray): Copy of H before applying the update, shape (r, n).
    """
    n = X.shape[1]
    H_old = H.copy()

    for j in range(n):
        H[:, j], learning_rates[j] = sigmoid_cross_entropy(A=W, b=X[:, j], x=H[:, j], alpha=learning_rates[j], method=method, P=P[:, j])

    H += theta * (H - H_old)

    return H, learning_rates, H_old

def theta_momentum(dec, theta, theta_b, g, g_b, eta):
    """
    Updates the momentum parameters (theta and theta_b) for the Nesterov Accelerated
    Gradient (NAG) method based on the objective function's behavior.

    Args:
        dec (bool): Boolean flag indicating if the objective function decreased.
        theta (float): Current momentum parameter.
        theta_b (float): Current upper bound or secondary momentum parameter.
        g (float): Growth factor for theta when the objective decreases.
        g_b (float): Growth factor for theta_b when the objective decreases.
        eta (float): Decay factor for theta when the objective increases.

    Returns:
        theta (float): Updated momentum parameter.
        theta_b (float): Updated secondary momentum parameter.
    """
    if dec:
        theta = min(theta_b, g*theta)
        theta_b = min(1, g_b*theta_b)
    else:
        theta_b = theta
        theta /= eta

    return theta, theta_b


def sigma_NMD_CE(X,r=1,W0=None,H0=None,X_filled=None,init="random",max_iter=1e4,tol=1e-3,method="lipschitz_step",theta=0.75,theta_b=1,g=1.05,g_b=1.01,eta=1.5,use_extrapolation=False,max_times=None,stopping_criteria=True):
    """
    This function solves the Nonlinear Matrix Decomposition (NMD) problem with sigmoid function.
    Given an input matrix `X` of shape (m, n), a rank `r`, and optional initial matrices `W0` and `H0`,
    it iteratively solves the optimization problem:
    `min_{W,H} CE(X, sigmoid(WH))`,
    using a block coordinate descent method.

    Args:
        X (numpy.ndarray): Input matrix of shape (m, n).
        r (int, optional): Rank of the decomposition. Default: 1.
        W0 (numpy.ndarray, optional): Initial matrix W of shape (m, r). Default: None.
        H0 (numpy.ndarray, optional): Initial matrix H of shape (r, n). Default: None.
        X_filled (numpy.ndarray, optional): Ground truth matrix with filled missing values (for validation). Default: None.
        init (int, optional): Initialization mode (random, tsvd). Default: random.
        max_iter (int, optional): Maximum number of iterations. Default: 1e4.
        tol (float, optional): Tolerance for stopping criterion. Default: 1e-3.
        method (str, optional): Line search method to use. Options are:
            - "lipschitz_step": Coordinate descent using a local Lipschitz constant.
            - "lipschitz_step_global": Gradient descent using a global Lipschitz constant.
            - "wolfe_conditions": Gradient descent using the standard Wolfe conditions for step size selection.
            Default: "lipschitz_step".
        theta (float, optional): Momentum / extrapolation coefficient used in the extrapolation step. Default: 0.75.
        theta_b (float, optional): Auxiliary momentum parameter used to adapt theta (initial or backup value). Default: 1.
        g (float, optional): Multiplicative factor for theta. Default: 1.05.
        g_b (float, optional): Secondary multiplicative factor for theta. Default: 1.01.
        eta (float, optional): Reduction factor used when progress fails. Default: 1.5.
        use_extrapolation (bool, optional): Whether to apply momentum extrapolation updates (True) or plain alternating updates (False). Default: False.
        max_times (float, optional): Maximum times of computation. Default: None.
        stopping_criteria (bool, optional): Whether to use the stopping criterion based on relative error stagnation. Set to True to enable the criterion, or False to bypass it. Default: True.

    Returns:
        W_opt (numpy.ndarray): Optimal matrix W of shape (m, r).
        H_opt (numpy.ndarray): Optimal matrix H of shape (r, n).
        relative_errors (list): List of relative errors at each iteration. e(X, sigmoid(WH)) / e(X, mean(X))
        train_error_rates (list): List of error rates on observed values at each iteration.
        test_error_rates (list): List of error rates on missing entries at each iteration.
        rmse (list): List of RMSE values at each iteration.
        mae (list): List of MAE values at each iteration.
        l1_relative_errors (list): List of L1 relative errors at each iteration.
        times (list): List of times at each iteration.

    """
    m,n = X.shape
    if X_filled is not None:
        M = np.isnan(X) & ~np.isnan(X_filled)
    else:
        M = np.zeros_like(X)
    P = ~np.isnan(X)
    X = np.nan_to_num(X)
    X_filled = np.nan_to_num(X_filled)
    if W0 is None or H0 is None:

        if init == "random":
            W = np.random.rand(m, r) if W0 is None else W0.copy()
            H = np.random.rand(r, n) if H0 is None else H0.copy()

        elif init == "tsvd":
            if np.all(M==0):
                u, s, vh = np.linalg.svd(2*X-1, full_matrices=False)
                W = u[:, :r] @ np.diag(np.sqrt(s[:r])) if W0 is None else W0.copy()
                H = np.diag(np.sqrt(s[:r])) @ vh[:r, :] if H0 is None else H0.copy()
            else:
                if W0 is None or H0 is None:
                    W_svd, H_svd, e1,e2,e3,e4,e5,e6,e7 = WLRA(2*X-1, P, r, X_filled=X_filled, nonneg=False)
                else:
                    W_svd, H_svd, e1,e2,e3,e4,e5,e6,e7 = WLRA(2*X-1, P, r, W0, H0.T, X_filled=X_filled, nonneg=False)
                W = W_svd if W0 is None else W0.copy()
                H = H_svd if H0 is None else H0.copy()
                H = H.T

    else:
        W = W0.copy()
        H = H0.copy()

    W_opt, H_opt = W.copy(), H.copy()

    if X_filled is not None and np.sum(M) > 0:
        test_err_rate0 = np.sum(M)
        rmse0 = np.sum(M)
        mae0 = np.sum(M)
    else:
        test_err_rate0 = np.sum(P)
        rmse0 = np.sum(P)
        mae0 = np.sum(P)

    relative_errors, train_error_rates, test_error_rates, rmse, mae, l1_relative_errors = [], [], [], [], [], []
    learning_rates_W = np.ones(W.shape[0]) * 0.9
    learning_rates_H = np.ones(H.shape[1]) * 0.9
    iteration = 0

    X_tilde = sigmoid(W@H)
    relative_errors.append(relative_cross_entropy(X, X_tilde, P))
    l1_relative_errors.append(np.sum(np.abs((X - X_tilde) * P)) / np.sum(np.abs(X * P)))
    train_error_rates.append(np.sum(((X - np.round(X_tilde)) * P) ** 2) / np.sum(P))
    if X_filled is not None and np.sum(M) > 0:
        test_error_rates.append(np.sum(((X_filled - np.round(X_tilde)) * M) ** 2) / test_err_rate0)
        rmse.append(np.sqrt(np.sum(((X_filled - X_tilde) * M) ** 2) / rmse0))
        mae.append(np.sum(np.abs((X_filled - X_tilde) * M))/mae0)
    else:
        test_error_rates.append(np.sum(((X - np.round(X_tilde)) * P) ** 2) / test_err_rate0)
        rmse.append(np.sqrt(np.sum(((X - X_tilde) * P) ** 2) / rmse0))
        mae.append(np.sum(np.abs((X - X_tilde) * P)) / mae0)

    start = time.perf_counter()
    times = [0.0]
    counter = 0
    while True:

        if not use_extrapolation:
            H, learning_rates_H = updateFact(X, W, H, learning_rates_H, method=method, P=P)
            WT, learning_rates_W = updateFact(X.T, H.T, W.T, learning_rates_W, method=method, P=P.T)
            W = WT.T
            X_tilde = sigmoid(W@H)
            rel_err = relative_cross_entropy(X, X_tilde, P)

        else:
            H, learning_rates_H, H_old = updateFact_momentum(X, W, H, learning_rates_H, method=method, P=P,theta=theta)
            WT, learning_rates_W, WT_old = updateFact_momentum(X.T, H.T, W.T, learning_rates_W, method=method,P=P.T,theta=theta)
            W, W_old = WT.T, WT_old.T
            X_tilde = sigmoid(W @ H)
            rel_err = relative_cross_entropy(X, X_tilde, P)

            dec = rel_err < relative_errors[-1]
            if not dec:
                H -= theta * (H - H_old)
                W -= theta * (W - W_old)
                X_tilde = sigmoid(W @ H)
                rel_err = relative_cross_entropy(X, X_tilde, P)
            theta, theta_b = theta_momentum(dec,theta,theta_b,g,g_b,eta)

        if rel_err < min(relative_errors):
            W_opt, H_opt = W.copy(), H.copy()

        relative_errors.append(rel_err)
        l1_relative_errors.append(np.sum(np.abs((X - X_tilde) * P)) / np.sum(np.abs(X * P)))
        train_error_rates.append(np.sum(((X - np.round(X_tilde)) * P) ** 2) / np.sum(P))
        if X_filled is not None and np.sum(M) > 0:
            test_error_rates.append(np.sum(((X_filled - np.round(X_tilde)) * M) ** 2) / test_err_rate0)
            rmse.append(np.sqrt(np.sum(((X_filled - X_tilde) * M) ** 2) / rmse0))
            mae.append(np.sum(np.abs((X_filled - X_tilde) * M)) / mae0)
        else:
            test_error_rates.append(np.sum(((X - np.round(X_tilde)) * P) ** 2) / test_err_rate0)
            rmse.append(np.sqrt(np.sum(((X - X_tilde) * P) ** 2) / rmse0))
            mae.append(np.sum(np.abs((X - X_tilde) * P)) / mae0)

        timer = time.perf_counter() - start
        times.append(timer)

        iteration += 1
        if max_iter is not None and iteration >= max_iter:
            break

        if (1 - (relative_errors[-1]/relative_errors[-2]) <= tol) and (stopping_criteria==True):
            counter += 1
        else:
            counter = 0

        if counter == 10:
            break

        if max_times != None:
            if timer >= max_times:
                break

    return W_opt, H_opt, relative_errors, train_error_rates, test_error_rates, rmse, mae, l1_relative_errors, times


def sigmoid_cross_entropy(A, b, x, alpha, method, P):
    """
        Optimizes the sigmoid cross-entropy objective function for a single column
        of the factor matrix, handling missing data through a binary mask.
        This function updates the vector x using one of three optimization strategies
        based on the 'method' parameter. It restricts the computation to observed
        entries as defined by the mask P.

        Args:
            A (numpy.ndarray): The basis matrix (W) of shape (m, r).
            b (numpy.ndarray): The target observed vector (a column of X) of shape (m,).
            x (numpy.ndarray): The vector to be updated (a column of H) of shape (r,).
            alpha (float): Initial step length (used primarily for Wolfe conditions).
            method (str): The optimization method to employ. Options are:
                - "lipschitz_step": Block coordinate descent using local Lipschitz constants
                  derived from the rows of A.
                - "lipschitz_step_global": Gradient descent using a global Lipschitz
                  constant based on the spectral norm of A.
                - "wolfe_conditions": Gradient descent with a step length determined by
                  the strong Wolfe conditions.
            P (numpy.ndarray): Binary mask (m,) where values > 0 indicate observed entries.

        Returns:
            x (numpy.ndarray): The updated vector after one iteration of the chosen method.
            alpha (float): The step length used (updated if Wolfe conditions were applied).
    """
    Obs_b = P > 0
    A = A[Obs_b, :]
    b = b[Obs_b]
    if method == "lipschitz_step":
        L = 0.25 * np.sum(A**2,axis=0)
        L[L < 1e-12] = 1e-12
        z = A @ x
        r = x.shape[0]
        for j in range(r):
            sigma_Ax = sigmoid(z)
            g_j = np.dot(A[:,j], sigma_Ax - b)
            delta_x = - (1.0/L[j]) * g_j
            x[j] += delta_x
            z += A[:,j] * delta_x
        return x, alpha

    elif method == "lipschitz_step_global":
        z = A @ x
        sigma = sigmoid(z)
        g = A.T @ (sigma - b)
        norm_A = np.linalg.norm(A, 2)
        if norm_A < 1e-12:
            alpha_global = 1.0
        else:
            alpha_global = 4.0 / (norm_A ** 2)
        x_new = x - alpha_global * g
        return x_new, alpha_global

    elif method == "wolfe_conditions":
        f, g = fct_opti(A, b, x)
        d = -g
        alpha = compute_step_length(fct_opti, A, b, x, d, alpha)
        x_new = x + alpha * d
        return x_new, alpha