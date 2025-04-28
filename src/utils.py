import numpy as np


def projsplx(y):
    """
    projsplx projects a vector to a simplex
    by the algorithm presented in (Chen an Ye, "Projection Onto A Simplex", 2011)
    Author: Sangwoong Yoon (sangwoong24yoon@gmail.com)
    """
    assert len(y.shape) == 1
    N = y.shape[0]
    y_flipsort = np.flipud(np.sort(y))
    cumsum = np.cumsum(y_flipsort)
    t = (cumsum - 1) / np.arange(1, N + 1).astype('float')
    t_iter = t[:-1]
    t_last = t[-1]
    y_iter = y_flipsort[1:]
    if np.all((t_iter - y_iter) < 0):
        t_hat = t_last
    else:
        # find i such that t>=y
        eq_idx = np.searchsorted(t_iter - y_iter, 0, side='left')
        t_hat = t_iter[eq_idx]
    x = y - t_hat
    # there may be a numerical error such that the constraints are not exactly met.
    x[x < 0.] = 0.
    x[x > 1.] = 1.
    assert np.abs(x.sum() - 1.) <= 1e-5
    assert np.all(x >= 0) and np.all(x <= 1.)
    return x


def proj_vh(vh):
    """
    Projection matrix V into the feasible set \mathcal{F}_V

    :param vh: matrix with dimension: (number of quasispecies) x (4*haplotype length)
    :return: matrix in the same dimension of input

    *If all the element are with the same value, the function will let the first element to 1 and others 0
    """
    (k, m) = vh.shape
    vh = np.reshape(vh, (k, -1, 4))
    # vh = np.reshape(vh, (k, int(m / 4), 4))
    # Find the indices of the maximum value along the third dimension
    max_indices = np.argmax(vh, axis=2)

    # Create a new array B with the same shape as A, initialized with zeros
    ind_mat = np.zeros_like(vh)

    # Set the value to 1 where the maximum value occurs
    ind_mat[np.arange(vh.shape[0])[:, None], np.arange(vh.shape[1]), max_indices] = 1
    vh = np.reshape(ind_mat, (k, m))
    return vh


def mat2ten(M):
    P = np.double(M != 0)  # projection matrix
    T_P = np.tile(P[:, :, np.newaxis], (1, 4)).reshape(M.shape[0], -1)
    T_M = np.dstack((np.double(M == 1), np.double(M == 2), np.double(M == 3), np.double(M == 4))).reshape(M.shape[0],                                                                                           -1)
    return P, T_P, T_M


def ten2mat(T_M):
    M = np.argmax(T_M.reshape(T_M.shape[0], -1, 4), axis=2) + 1
    return M

