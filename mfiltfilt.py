# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <14/02/2020 14:57:19 (BrianM)>

"""
Python implementation of Matlab's 'filtfilt' function
Example:
    x = np.random.randn(1001, )
    b, a = signal.butter(1, 0.5 / 32, 'high')
    y = mat_filtfilt(b, a, x)


John M. O' Toole, University College Cork
Started: 06-11-2019
last update: Time-stamp: <2019-11-06 16:20:45 (otoolej)>
"""
import numpy as np
from scipy import sparse, signal


def gain_filt(b, a):
    """Step response from filter (used as initial conditions later)

    Parameters
    ----------
    b: ndarray
        filter coefficients
    a: ndarray
        filter coefficients

    Returns
    -------
    zi : ndarray
        step response as a vector
    """
    # max. length of coefficients:
    if len(a) == 1 and a[0] == 1:
        a = np.ones([len(b)])
    N_filt = max(len(b), len(a))

    # arrange entries of sparse matrix:
    rows = [*range(N_filt - 1), *range(1, N_filt - 1), *range(N_filt - 2)]
    cols = [*np.zeros(N_filt - 1).astype(int), *
            range(1, N_filt - 1), *range(1, N_filt - 1)]
    vals = [*np.hstack(((1 + a[1]), a[2:N_filt])),
            *np.ones(N_filt - 2).astype(int),
            *-np.ones(N_filt - 2).astype(int)]
    rhs = b[1:N_filt] - b[0] * a[1:N_filt]

    AA = sparse.coo_matrix((vals, (rows, cols))).tocsr()
    zi = sparse.linalg.spsolve(AA, rhs)

    return (zi)



def mat_filtfilt(b, a, x):
    """Fowards--backwards filter to match Matlab's 'filtfilt' function


    Parameters
    ----------
    b: ndarray
        filter coefficients
    a: ndarray
        filter coefficients
    x: ndarray
        input signal

    Returns
    -------
    y : ndarray
        filtered signal
    """
    # 1. pad the signal:
    L_pad = 3 * (max(len(b), len(a)) - 1)
    x_pad = np.concatenate((2 * x[0] - x[1:(L_pad + 1)][::-1],
                            x,
                            2 * x[-1] - x[len(x)-L_pad-1:-1][::-1]))

    # 2. estimate initial filter conditions:
    zi = gain_filt(b, a)

    # 3. forwards and backwards filter:
    x_pad, _ = signal.lfilter(b, a, x_pad, zi=zi * x_pad[0])
    x_pad = x_pad[::-1]
    x_pad, _ = signal.lfilter(b, a, x_pad, zi=zi * x_pad[0])
    x_pad = x_pad[::-1]

    # 4. remove the padding:
    y = x_pad[L_pad:len(x) + L_pad]

    return (y)
