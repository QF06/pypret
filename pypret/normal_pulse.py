""" Provides a function to generate normal pulses with specified TBP=0.5.
"""
import numpy as np
import scipy.optimize as opt
from . import lib


def normal_pulse(pulse, edge_value=None, check=True):
    """ Creates a normal pulse with a specified time-bandwidth product.

    Parameters
    ----------
    pulse : Pulse instance
    edge_value : float, optional
        The maximal value for the pulse amplitude at the edges of the grid.
        It defaults to the double value epsilon ~2e-16.

    Returns
    -------
    bool : True on success, False if an error occured. The resulting pulse
        is stored in the Pulse instance passed to the function.

    """
    if edge_value is None:
        # this is roughly the roundoff error induced by an FFT
        edge_value = pulse.N * np.finfo(np.double).eps
    # access/calculate some fundamental grid parameters
    t, w = pulse.t, pulse.w
    t1, t2 = t[0], t[-1]
    w1, w2 = w[0], w[-1]
    t0, w0 = 0.5 * (t1 + t2), 0.5 * (w1 + w2)
    log_edge = np.log(edge_value)

    """ Calculate the width of a Gaussian function that drops exactly to
        edge_value at the edges of the grid.
    """
    spectral_width = np.sqrt(-0.125 * (w1 - w2)**2 / log_edge)
    # Now the same in the temporal domain
    max_temporal_width = np.sqrt(-0.125 * (t1 - t2)**2 / log_edge)
    # The actual temporal width is obtained by the uncertainty relation
    # from the specified TBP
    temporal_width = 2.0 * 0.5 / spectral_width

    if temporal_width > max_temporal_width:
        print("The required time-bandwidth product cannot be reached! "
              "Decrease edge_value or increase pulse.N!")
        return False

    # special case for TBP = 0.5 (transform-limited case)

    phase = np.exp(1.0j * lib.twopi)
    pulse.spectrum = lib.gaussian(w, w0, spectral_width) * phase
    return True
