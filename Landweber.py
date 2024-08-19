import numpy as np
import time


# Landweber iteration for denoising
def Landweber_denoising(u_del_flatten, whno, alpha, iteration_cg, relaxation_term):
    # u_i+1 = u_i - \tau * ( (u + w) - f_del + gam.T * gam (u + w))
    residual = []
    u = np.zeros((iteration_cg + 1, u_del_flatten.shape[0]))
    u[0] = u_del_flatten
    t1 = time.time()
    for i in range(1, iteration_cg + 1):
        t = alpha ** 2 * u[i - 1, :]
        u[i, :] = u[i - 1, :] - relaxation_term * (u[i - 1, :] + whno - u_del_flatten + t)
        residual += [np.linalg.norm(u[i] - u[i - 1])]
    t = time.time() - t1
    return u, residual, t


# Landweber iteration for deblurring (specific CG approach for minimizing ||Ax - y||_2)

def Landweber_deblurring(u_del_flatten, A, alpha, iteration_cg, relaxation_term):
    u = np.zeros((iteration_cg + 1, u_del_flatten.shape[0]))
    residual = []
    u[0] = u_del_flatten
    t1 = time.time()

    # condition = Condition(A)
    # print(condition)
    for i in range(1, iteration_cg + 1):  # CG for minimum
        t = alpha * u[i - 1, :]
        u[i, :] = u[i - 1, :] - relaxation_term * (A.T.dot(A.dot(u[i - 1, :]) - u_del_flatten) + t)
        residual += [np.linalg.norm(u[i] - u[i - 1])]
    t = time.time() - t1
    return u, residual, t




def Landweber(A, f, alpha, iteration_cg, relaxation_term, pre_whno, post_whno, true, tol):
    u = np.zeros((iteration_cg + 1, f.shape[0]))
    u[0] = f
    residual = []
    if pre_whno is None:
        pre_whno = np.zeros(f.shape[0])
    if post_whno is None:
        post_whno = np.zeros(f.shape[0])

    condition = Condition(A.toarray())
    t1 = time.time()
    for i in range(1, iteration_cg + 1):  # Landweber for minimum
        if np.linalg.norm(u[i - 1] - true) / np.linalg.norm(f - true) > tol:
            t = alpha * u[i - 1, :]
            u[i, :] = u[i - 1, :] - relaxation_term * (A.T.dot(A.dot(u[i - 1, :] + pre_whno) + post_whno - f) + t)
            residual += [np.linalg.norm(u[i] - u[i - 1])]
        else:
            u = u[0:i]
            residual = residual[0:i]
            break
    t = time.time() - t1
    return u, residual, t, condition


def Condition(symmetric_array):
    vals = np.linalg.eigvalsh(symmetric_array, UPLO='U')
    if min(vals) == 0:
        return np.infty
    else:
        return abs(max(vals) / min(vals))
