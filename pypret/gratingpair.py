""" Grating pair calcuate the GVD, TOD and FOD for a grating-pair
"""
import numpy as np
import matplotlib.pyplot as plt


class GratingPair(object):
    """ A class for modelling a strecher used in CPA.
    """

    def __init__(self, pulse, theta, groove, lg):
        """ Initializes an optical pulse described by its envelope.

        Parameters
        ----------
        theta: The incident angle to the grating.

        groove: The groove spacing of the grating [lines/mm].

        lg: Distance between gratings (grating distance) [m]
        """
        self.ft = pulse.ft
        self.w = pulse.w
        self.t = pulse.t
        self.spectral_intensity = pulse.spectral_intensity
        self.wl = pulse.wl0
        self.w_min = min(pulse.wl)
        self.w_max = max(pulse.wl)
        self.theta = theta
        self.groove = groove
        self.lg = lg
        self._c = 2.99792458 * 10 ** 8
        self._d = 1 / (groove * 10 ** 3)  # line-width [m]

    def _get_GVD(self):
        """ From get the bandwidth of the input pulse

        Parameters
        ----------
        pulse: A 'pulse' instance that specifies a central wavelength and
                a spectrum width
        Returns
        -------

        """
        wl, theta, lg, d, c = self.wl, self.theta, self.lg, self._d, self._c
        self.GVD = -(2 * wl ** 3 * lg / (2 * np.pi * c ** 2 * d ** 2)) * (1 - (wl / d - np.sin(theta)) ** 2) ** -3 / 2

    def _get_TOD(self):
        """ Calculate the third-order dispersion (TOD)
        Returns
        -------

        """
        wl, theta, lg, d, c = self.wl, self.theta, self.lg, self._d, self._c
        self.TOD = -(3 / 2 * np.pi * wl / c) * self.GVD * (
                    (1 + wl / d * np.sin(theta) - np.sin(theta) ** 2) / (1 - (wl / d - np.sin(theta) ** 2)))

    def stretch(self):
        self._get_GVD()
        self._get_TOD()
        stretch_spectrum = np.exp((1.0j * 1/2 * self.GVD * self.w ** 2) + (1.0j * 1/6 * self.TOD * self.w ** 2)) * \
                            self.spectral_intensity
        self.stretch_intensity = self.ft.backward(stretch_spectrum)

    def plot_stretch(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(self.t, self.spectral_intensity)
        ax1.set_title('Original')
        ax2.plot(self.t, abs(self.stretch_intensity))
        ax2.set_title('Stretch')
        plt.show()
        plt.close()


def plot_stretch_figure(pulse, theta, groove, lg):
    pulse2 = GratingPair(pulse, theta, groove, lg)
    pulse2.stretch()
    pulse2.plot_stretch()