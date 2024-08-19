import numpy as np
import time
import inspect
import scipy.sparse.linalg
import Landweber
import math


def FPC(A, f, alpha, iterations, pre_whno, post_whno, stepsize, true, tol):
    u = np.zeros((iterations + 1, f.shape[0]))
    u[0] = f
    x = f
    residual = []
    t1 = time.time()
    if pre_whno is None:
        pre_whno = np.zeros(f.shape[0])
    if post_whno is None:
        post_whno = np.zeros(f.shape[0])

    for y in range(1, iterations + 1):
        if np.linalg.norm(u[y - 1] - true) / np.linalg.norm(f - true) >= tol:
            x = x - stepsize * (A.T.dot(A.dot(x + pre_whno) + post_whno - f))
            for i in range(f.shape[0]):
                u[y, i] = np.sign(x[i]) * max(abs(x[i]) - stepsize * alpha, 0)
            residual += [np.linalg.norm(u[y] - u[y - 1])]
        else:
            u = u[0:y]
            residual = residual[0:y]
            break
    t = time.time() - t1
    return u, residual, t


def l1_approx(A, f, alpha, iterations, pre_whno, post_whno, stepsize, true, tol, T_1=15, T_2=0, method="Landweber"):
    """We are using in this program that we can approximate ||x||_1 with ||L(x)x||_2^2
    where L(x) = |x|^-(1/2)"""
    u = np.zeros((iterations + 1, f.shape[0]))
    u[0] = f
    residual = []
    t1 = time.time()
    x = np.zeros(f.shape[0])
    x_zeros = np.zeros(iterations + 1)
    if pre_whno is None:
        pre_whno = np.zeros(f.shape[0])
    if post_whno is None:
        post_whno = np.zeros(f.shape[0])

    for y in range(1, iterations + 1):
        if np.linalg.norm(u[y - 1] - true) / np.linalg.norm(f - true) >= tol:
            x_z = 0
            for i in range(f.shape[0]):
                if u[y - 1, i] >= T_1:  # Threshold for numerical stability
                    x[i] = u[y - 1, i] ** -(1 / 2)
                else:
                    if T_2 == 0:
                        x[i] = 0
                    else:
                        x[i] = T_2
                    x_z += 1
            x_zeros[y] = math.log(T_1 * x_z + np.linalg.norm(x))
            u[y, :] = u[y - 1, :] - stepsize * A.T.dot(A.dot(u[y - 1, :] + pre_whno) + post_whno - f) + 2 * alpha * x
            residual += [np.linalg.norm(u[y] - u[y - 1])]

        else:
            u = u[0:y]
            residual = residual[0:y]
            break
    t = time.time() - t1
    return u, residual, x_zeros, t


def Bregman_iteration(A, f, alpha, p, iterations_mini, iterations_br, method, pre_whno, post_whno, delta, eta, true,
                      tol, stepsize=0.5):
    u_breg = np.zeros((iterations_br + 1, f.shape[0]))
    u_breg[0] = f
    x = np.zeros((iterations_br + 1, f.shape[0]))
    x[0] = f
    residual = []
    t1 = time.time()
    if pre_whno is None:
        pre_whno = np.zeros(f.shape[0])
    if post_whno is None:
        post_whno = np.zeros(f.shape[0])

    if p == 2:
        if method == "CG":
            for i in range(1, iterations_br + 1):
                if np.linalg.norm(u_breg[i - 1] - true) / np.linalg.norm(f - true) >= tol:
                    result, residual, a = solve_tykhonov_sparse(A, x[i - 1, :], alpha, iterations_mini, pre_whno,
                                                                post_whno)
                    u_breg[i, :] = result[
                        (iterations_mini - 1) + (i - 1) * iterations_mini]  # Necessary since Scipy saves result
                    # differently and result is global
                    x[i, :] = x[i - 1, :] + f - (A.dot(u_breg[i, :] + pre_whno) + post_whno)
                else:
                    u_breg = u_breg[0:i]
                    residual = residual[0:i]
                    break
            t = time.time() - t1
            return u_breg, residual, t

        if method == "Landweber":
            for i in range(1, iterations_br + 1):
                if np.linalg.norm(u_breg[i - 1] - true) / np.linalg.norm(f - true) >= tol:
                    result, residual_temp, a, cond = Landweber.Landweber(A, x[i - 1, :], alpha, iterations_mini,
                                                                         stepsize,
                                                                         pre_whno, post_whno, true, tol)
                    residual += residual_temp
                    u_breg[i, :] = result[len(result) - 1, :]
                    x[i, :] = x[i - 1, :] + f - (A.dot(u_breg[i, :] + pre_whno) + post_whno)
                else:
                    u_breg = u_breg[0:i]
                    residual = residual[0:i]
                    break
                # print(h, "<", morozov_threshhold)

            t = time.time() - t1
            return u_breg, residual, t, cond

    if p == 1:
        if method == "FPC":
            for i in range(1, iterations_br + 1):
                if np.linalg.norm(u_breg[i - 1] - true) / np.linalg.norm(f - true) >= tol:
                    result, residual_temp, a = FPC(A, x[i - 1, :], alpha, iterations_mini, pre_whno, post_whno,
                                                   stepsize, true, tol)
                    residual += residual_temp
                    u_breg[i, :] = result[len(result) - 1, :]
                    x[i, :] = x[i - 1, :] + f - (A.dot(u_breg[i, :] + pre_whno) + post_whno)
                else:
                    u_breg = u_breg[0:i]
                    residual = residual[0:i]
                    break
            t = time.time() - t1
            return u_breg, residual, t

        if method == "l1_approx":
            x_zeros = np.zeros((iterations_mini + 1, iterations_br + 1))
            for i in range(1, iterations_br + 1):
                if np.linalg.norm(u_breg[i - 1] - true) / np.linalg.norm(f - true) >= tol:
                    result, residual_temp, x_zeros[:, i], a = l1_approx(A, x[i - 1, :], alpha, iterations_mini,
                                                                        pre_whno,
                                                                        post_whno, stepsize, true, tol)
                    residual += residual_temp
                    u_breg[i, :] = result[len(result) - 1, :]
                    x[i, :] = x[i - 1, :] + f - (A.dot(u_breg[i, :] + pre_whno) + post_whno)
                else:
                    u_breg = u_breg[0:i]
                    residual = residual[0:i]
                    break
            t = time.time() - t1
            return u_breg, residual, x_zeros, t


# news.zahlt.info/en/optimization/image-deconvolution-using-tikhonov-regularization/

def solve_tykhonov_sparse(A, f, alpha, maxiter, pre_whno, post_whno):
    """
    Tykhonov regularization of an observed signal, given a linear degradation matrix
    and a Gamma regularization matrix.
    Formula is

    x* = (H'H + G'G)^{-1} H'y

    With y the observed signal, H the degradation matrix, G the regularization matrix.

    This function is better than dense_tykhonov in the sense that it does
    not attempt to invert the matrix H'H + G'G.
    To implement pre_noise and post_noise, we transform
    A^T * A * ( u + pre_noise) + post_noise - f) + G^T * G * u  = 0 to 
    (A^T * A + G^T * G)u = A^T * (f - post_noise) - A^T * A * pre_noise"""
    t1 = time.time()
    H1 = A.T.dot(A)  # may not be sparse any longer in the general case
    for n in range(0, H1.shape[0]):  # Gamma.T.dot(Gamma) is unnecessary for Gamma = Identity
        H1[n, n] = H1[n, n] + alpha ** 2  # for condition necessary because of limited space on HardWare
    if pre_whno is None:
        pre_whno = np.zeros(f.shape[0])
    if post_whno is None:
        post_whno = np.zeros(f.shape[0])
    f = A.T.dot(f - post_whno)
    pre_whno = H1.dot(pre_whno)
    f = f - pre_whno

    scipy.sparse.linalg.cg(H1, f, maxiter=maxiter, callback=report, tol=0)
    t = time.time() - t1
    return cg_result, cg_nresid, t


def report(xk):
    global cg_timer
    global cg_result
    global cg_nresid
    frame = inspect.currentframe().f_back
    cg_result[cg_timer, :] = xk
    cg_nresid = cg_nresid + [frame.f_locals['resid']]
    cg_timer += 1
