import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from typing import Union, List, Tuple


class HampelFilter:
    """
    HampelFilter class for providing additional functionality such as checking the upper/lower boundaries for paramter tuning.
    """

    def __init__(self, window_size: int = 5, n_sigma: int = 3, c: float = 1.4826):
        """ Initialize HampelFilter object. Rolling median and rolling sigma are calculated here.

        :param window_size: length of the sliding window, a positive odd integer.
            (`window_size` - 1) // 2 adjacent samples on each side of the current sample are used for calculating median.
        :param n_sigma: threshold for outlier detection, a real scalar greater than or equal to 0. default is 3.
        :param c: consistency constant. default is 1.4826, supposing the given timeseries values are normally distributed.
        :return: the outlier indices
        """

        if not (type(window_size) == int and window_size % 2 == 1 and window_size > 0):
            raise ValueError("window_size must be a positive odd integer greater than 0.")

        if not (type(n_sigma) == int and n_sigma >= 0):
            raise ValueError("n_sigma must be a positive integer greater than or equal to 0.")

        self.window_size = window_size
        self.n_sigma = n_sigma
        self.c = c

        # These values will be set after executing apply()
        self._outlier_indices = None
        self._upper_bound = None
        self._lower_bound = None

    def apply(self, x: Union[List, pd.Series, np.ndarray]):
        """ Return the indices of the detected outliers by the filter.

        :param x: timeseries values of type List, numpy.ndarray, or pandas.Series

        :return: indices of the outliers
        """
        # Check given arguments
        if not (type(x) == list or type(x) == np.ndarray or type(x) == pd.Series):
            raise ValueError("x must be either of type List, numpy.ndarray, or pandas.Series.")

        # calculate rolling_median and rolling_sigma using the given parameters.
        x_window_view = sliding_window_view(np.array(x), window_shape=self.window_size)
        rolling_median = np.median(x_window_view, axis=1)
        rolling_sigma = self.c * np.median(np.abs(x_window_view - rolling_median.reshape(-1, 1)), axis=1)

        self._upper_bound = rolling_median + (self.n_sigma * rolling_sigma)
        self._lower_bound = rolling_median - (self.n_sigma * rolling_sigma)

        outlier_indices = np.nonzero(
            np.abs(np.array(x)[(self.window_size - 1) // 2:-(self.window_size - 1) // 2] - rolling_median)
            >= (self.n_sigma * rolling_sigma)
        )[0] + (self.window_size - 1) // 2

        if type(x) == list:
            # When x is of List[float | int], return the indices in List.
            self._outlier_indices = list(outlier_indices)
        elif type(x) == pd.Series:
            # When x is of pd.Series, return the indices of the Series object.
            self._outlier_indices = x.index[outlier_indices]
        else:
            self._outlier_indices = outlier_indices

        return self

    def get_indices(self) -> Union[List, pd.Series, np.ndarray]:
        """
        """
        if self._outlier_indices is None:
            raise AttributeError("Outlier indices have not been set. Execute hampel_filter_object.apply(x) first.")
        return self._outlier_indices

    def get_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the upper and lower boundaries of the filter. Note that the values are `window_size - 1` shorter than the given timeseries x.

        :return: a tuple of the lower bound values and the upper bound values. i.e. (lower_bound_values, upper_bound_values)
        """
        if self._upper_bound is None or self._lower_bound is None:
            raise AttributeError("Boundary values have not been set. Execute hampel_filter_object.apply() first.")

        return self._lower_bound, self._upper_bound


def hampel_filter(x: Union[List, pd.Series, np.ndarray], window_size: int = 5, n_sigma: int = 3, c: float = 1.4826) \
        -> Union[List, pd.Series, np.ndarray]:
    """ Outlier detection using the Hampel identifier

    :param x: timeseries values of type List, numpy.ndarray, or pandas.Series
    :param window_size: length of the sliding window, a positive odd integer.
        (`window_size` - 1) // 2 adjacent samples on each side of the current sample are used for calculating median.
    :param n_sigma: threshold for outlier detection, a real scalar greater than or equal to 0. default is 3.
    :param c: consistency constant. default is 1.4826, supposing the given timeseries values are normally distributed.
    :return: the outlier indices
    """

    return HampelFilter(window_size=window_size, n_sigma=n_sigma, c=c).apply(x).get_indices()
