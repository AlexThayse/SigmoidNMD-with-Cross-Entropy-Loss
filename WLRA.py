import numpy as np
import time


def WLRA(X, P=None, r=1, W0=None, H0=None, nonneg=False, lambda_=1e-6, tol=1e-4, maxiter=1e4, alpha=0.9999, X_filled=None, verbose=False):
    if P is None:
        P = np.float64(~np.isnan(X))

    m, n = X.shape
    if X_filled is not None:
        M = np.isnan(X) & ~np.isnan(X_filled)
    else:
        M = np.zeros_like(X)

    X = np.nan_to_num(X)
    X_filled = np.nan_to_num(X_filled)

    W = np.random.rand(m, r) if W0 is None else W0.copy()
    H = np.random.rand(n, r) if H0 is None else H0.copy()

    W, H = scaling_WH(W, H)
    W_opt, H_opt = W.copy(), H.copy()

    rel_err0 = np.sum((X * P) ** 2)
    if X_filled is not None and np.sum(M) > 0:
        test_err_rate0 = np.sum(M)
        rmse0 = np.sum(M)
        mae0 = np.sum(M)
    else:
        test_err_rate0 = np.sum(P)
        rmse0 = np.sum(P)
        mae0 = np.sum(P)

    relative_errors = []
    train_error_rates = []
    test_error_rates = []
    rmse = []
    mae = []
    l1_relative_errors = []
    times = []

    def compute_metrics(W, H, start_time):
        WH = W @ H.T
        WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
        relative_errors.append(np.sqrt(np.sum(((X - WH) * P) ** 2) / rel_err0))
        l1_relative_errors.append(np.sum(np.abs((X - WH) * P)) / np.sum(np.abs(X * P)))
        train_error_rates.append(np.sum(((X - WH_bound) * P) ** 2) / np.sum(P))
        if X_filled is not None and np.sum(M) > 0:
            test_error_rates.append(np.sum(((X_filled - WH_bound) * M) ** 2) / test_err_rate0)
            rmse.append(np.sqrt(np.sum(((X_filled - WH) * M) ** 2) / rmse0))
            mae.append(np.sum(np.abs((X_filled - WH) * M)) / mae0)
        else:
            test_error_rates.append(np.sum(((X - WH_bound) * P) ** 2) / test_err_rate0)
            rmse.append(np.sqrt(np.sum(((X - WH) * P) ** 2) / rmse0))
            mae.append(np.sum(np.abs((X - WH) * P)) / mae0)

        times.append(time.perf_counter() - start_time)

    start_time = time.perf_counter()
    compute_metrics(W, H, start_time)

    iteration = 0
    while True:
        R = X - W @ H.T
        for k in range(r):
            R += np.outer(W[:, k], H[:, k])
            Rp = R * P

            W[:, k] = (Rp @ H[:, k]) / (P @ (H[:, k] ** 2) + lambda_)
            if nonneg:
                W[:, k] = np.maximum(np.finfo(float).eps, W[:, k])

            H[:, k] = (Rp.T @ W[:, k]) / (P.T @ (W[:, k] ** 2) + lambda_)
            if nonneg:
                H[:, k] = np.maximum(np.finfo(float).eps, H[:, k])

            R -= np.outer(W[:, k], H[:, k])

        compute_metrics(W, H, start_time)

        current_rel_err = relative_errors[-1]

        if current_rel_err < min(relative_errors[:-1]):
            W_opt, H_opt = W.copy(), H.copy()

        W, H = scaling_WH(W, H)

        if current_rel_err <= tol:
            break

        iteration += 1
        if maxiter is not None and iteration >= maxiter:
            break

        if iteration % 10 == 0:
            if verbose:
                print(f"iteration {iteration}: relative error={current_rel_err}")

            if min(relative_errors[-10:]) > min(relative_errors[:-10]) * alpha:
                break

    return W_opt, H_opt, relative_errors, train_error_rates, test_error_rates, rmse, mae, l1_relative_errors, times

def scaling_WH(W, H):
    m,r = W.shape
    norm_W = np.sqrt(np.sum(W ** 2, axis=0)) + 1e-16
    norm_H = np.sqrt(np.sum(H ** 2, axis=0)) + 1e-16

    for k in range(r):
        W[:, k] = W[:, k]/np.sqrt(norm_W[k])*np.sqrt(norm_H[k])
        H[:, k] = H[:, k]/np.sqrt(norm_H[k])*np.sqrt(norm_W[k])

    return W, H


