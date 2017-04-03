from astropy.stats import LombScargle as astropy_LombScargle
from astropy.stats.lombscargle.core import strip_units
from astropy.stats.lombscargle.implementations.main import _get_frequency_grid

from .nfftls import lombscargle_nfft


class LombScargle(astropy_LombScargle):
    __doc__ = astropy_LombScargle.__doc__

    def power(self, frequency, normalization='standard', method='auto',
              assume_regular_frequency=False, method_kwds=None):
        """Compute the Lomb-Scargle power at the given frequencies

        Parameters
        ----------
        frequency : array_like or Quantity
            frequencies (not angular frequencies) at which to evaluate the
            periodogram. Note that in order to use method='fast', frequencies
            must be regularly-spaced.
        method : string (optional)
            specify the lomb scargle implementation to use. Options are:
            - 'auto': choose the best method based on the input
            - 'nfft': use the O[N log N] nfft library.
            - 'fast': use the O[N log N] fast method. Note that this requires
              evenly-spaced frequencies: by default this will be checked unless
              ``assume_regular_frequency`` is set to True.
            - 'slow': use the O[N^2] pure-python implementation
            - 'cython': use the O[N^2] cython implementation. This is slightly
              faster than method='slow', but much more memory efficient.
            - 'chi2': use the O[N^2] chi2/linear-fitting implementation
            - 'fastchi2': use the O[N log N] chi2 implementation. Note that this
              requires evenly-spaced frequencies: by default this will be checked
              unless ``assume_regular_frequency`` is set to True.
            - 'scipy': use ``scipy.signal.lombscargle``, which is an O[N^2]
              implementation written in C. Note that this does not support
              heteroskedastic errors.
        assume_regular_frequency : bool (optional)
            if True, assume that the input frequency is of the form
            freq = f0 + df * np.arange(N). Only referenced if method is 'auto'
            or 'fast'.
        normalization : string (optional, default='standard')
            Normalization to use for the periodogram.
            Options are 'standard', 'model', 'log', or 'psd'.
        fit_mean : bool (optional, default=True)
            if True, include a constant offset as part of the model at each
            frequency. This can lead to more accurate results, especially in
            the case of incomplete phase coverage.
        center_data : bool (optional, default=True)
            if True, pre-center the data by subtracting the weighted mean of
            the input data. This is especially important if fit_mean = False
        method_kwds : dict (optional)
            additional keywords to pass to the lomb-scargle method

        Returns
        -------
        power : ndarray
            The Lomb-Scargle power at the specified frequency
        """
        if method == 'nfft':
            if self.nterms != 1:
                raise ValueError("nfft method only works for nterms=1")
            f0, df, Nf = _get_frequency_grid(strip_units(frequency),
                                             assume_regular_frequency)
            if method_kwds and 'use_fft' in method_kwds:
                use_fft = method_kwds.pop('use_fft')
                if use_fft:
                    method_kwds['exponential_sum_method'] = 'nfft'
                else:
                    method_kwds['exponential_sum_method'] = 'slow'
            power = lombscargle_nfft(*strip_units(self.t, self.y, self.dy),
                                     f0, df, Nf,
                                     center_data=self.center_data,
                                     fit_mean=self.fit_mean,
                                     normalization=normalization,
                                     **(method_kwds or {}))
            return power * self._power_unit(normalization)
        else:
            return super(LombScargle, self).power(frequency=frequency,
                                                  normalization=normalization,
                                                  method=method,
                                                  assume_regular_frequency=assume_regular_frequency,
                                                  method_kwds=method_kwds)

    def autopower(self, method='auto', method_kwds=None,
                  normalization='standard', samples_per_peak=5,
                  nyquist_factor=5, minimum_frequency=None,
                  maximum_frequency=None):

        """Compute Lomb-Scargle power at automatically-determined frequencies

        Parameters
        ----------
        method : string (optional)
            specify the lomb scargle implementation to use. Options are:
            - 'auto': choose the best method based on the input
            - 'nfft': use the O[N log N] nfft library.
            - 'fast': use the O[N log N] fast method. Note that this requires
              evenly-spaced frequencies: by default this will be checked unless
              ``assume_regular_frequency`` is set to True.
            - 'slow': use the O[N^2] pure-python implementation
            - 'cython': use the O[N^2] cython implementation. This is slightly
              faster than method='slow', but much more memory efficient.
            - 'chi2': use the O[N^2] chi2/linear-fitting implementation
            - 'fastchi2': use the O[N log N] chi2 implementation. Note that this
              requires evenly-spaced frequencies: by default this will be checked
              unless ``assume_regular_frequency`` is set to True.
            - 'scipy': use ``scipy.signal.lombscargle``, which is an O[N^2]
              implementation written in C. Note that this does not support
              heteroskedastic errors.
        method_kwds : dict (optional)
            additional keywords to pass to the lomb-scargle method
        normalization : string (optional, default='standard')
            Normalization to use for the periodogram.
            Options are 'standard', 'model', or 'psd'.
        samples_per_peak : float (optional, default=5)
            The approximate number of desired samples across the typical peak
        nyquist_factor : float (optional, default=5)
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if maximum_frequency is not provided.
        minimum_frequency : float (optional)
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline.
        maximum_frequency : float (optional)
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency.

        Returns
        -------
        frequency, power : ndarrays
            The frequency and Lomb-Scargle power
        """
        if method == 'nfft':
            frequency = self.autofrequencyy(samples_per_peak=samples_per_peak,
                                            nyquist_factor=nyquist_factor,
                                            minimum_frequency=minimum_frequency,
                                            maximum_frequency=maximum_frequency)
            power = self.power(frequency,
                               normalization=normalization,
                               method=method, method_kwds=method_kwds,
                               assume_regular_frequency=True)
            return frequency, power
        else:
            return super(LombScargle, self).autopower(method=method,
                                                      method_kwds=method_kwds,
                                                      normalization=normalization,
                                                      samples_per_peak=samples_per_peak,
                                                      nyquist_factor=nyquist_factor,
                                                      minimum_frequency=minimum_frequency,
                                                      maximum_frequency=maximum_frequency)
