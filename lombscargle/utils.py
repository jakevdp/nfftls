import numpy as np
from pynfft.nfft import NFFT


def trig_sum(t, y, Nf, method='auto'):
    """Compute a trigonometric sum over frequencies f = range(-Nf, Nf)

    The computed sum is S_m = sum_n [y_n * exp(2 * i * f_m * t_n)]

    Parameters
    ----------
    t : array_like, length N
        Sorted array of times in the interval [-0.5, 0.5]
    y : array_like, length N
        array of weights
    Nf : integer
        number of frequencies
    method : string, default='auto'
        method to use. Must be one of ['auto', 'slow', 'nfft', 'ndft']

    Returns
    -------
    S : ndarray, length 2 * Nf
        resulting sums at each frequency
    """
    t, y = np.broadcast_arrays(t, y)
    assert t.ndim == 1
    assert t.min() > -0.5 and t.max() < 0.5
    assert method in ['nfft', 'ndft', 'slow', 'auto']

    if method == 'auto':
        method = 'nfft'
    if method == 'slow':
        freq = np.arange(-Nf, Nf)[:, np.newaxis]
        return np.dot(np.exp(2j * np.pi * freq * t), y)
    else:
        use_dft = (method == 'ndft')
        plan = NFFT(2 * Nf, len(t))
        plan.x = t
        plan.precompute()
        plan.f = y
        return plan.adjoint(use_dft=use_dft).copy()
