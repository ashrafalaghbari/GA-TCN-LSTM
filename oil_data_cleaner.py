"""Typing module for variable type annotations."""
from typing import Union, Tuple, Iterator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class OilDataCleaner:

    """
    This class is designed to identify, visualize, and remove outliers
    in time series data from oil wells.
    We have tested the outlier detection capability of this class on a
      daily time series dataset retrieved
      from the VOLVE field provided by Equinor (formerly known as Statoil).
    `The index of the input DataFrame must be a pandas DatetimeIndex object.`

    The class can clean four types of variables:

    - `on_stream_var`: Indicates the on-stream hours for production or injection wells.
    - `rate_var`: Indicates the production rate, such as oil, water, or gas, and
      can also be used for injection rate, such as water injected volume.
    - `avg_choke_var`: Indicates the average choke size (valve) that controls the production rates.
    - `roc`: Denotes the rate of change. The `detect_outliers_with_roc` method in this class
        can clean variables that behave in a relatively stable or gradual manner
          (e.g. average reservoir pressure
        and temperature and average wellhead temperature as well).

        The function, that detects outliers based on roc, works by identifying sudden movements
          that do not conform to
        the distribution of observations within the specified window. Each window contains
        a certain number of observations selected based on the window size, which can be
          defined by the user.

        The default values in all methods have been determined based on experimental results
            and have been optimized for outlier detection and removal.However, the user can
              change the default values.

    Parameters:
    ----------
        data (pd.DataFrame): The input DataFrame that contains the oil time series
          data to be cleaned.

    Attributes:
    ----------
        data (pd.DataFrame): A copy of the input data.

    """


    def __init__(self, data: pd.DataFrame) -> None:
        # Initializes the class instance with the input data passed as
        # a pandas DataFrame. A copy of the input data is created to
        # avoid modifying the original data outside of the class
        self.data = data.copy()



    def plot_outliers(
            self,
            var: str,
            outliers: pd.Series,
            color: str = 'red',
            alpha: float = 0.3,
            **kwargs
    ) -> None:

        """
        Visualizes the outliers in a time series variable.

        Parameters:
        ----------
            `var (str)`: the name of the variable to visualize
            `outliers (pd.Series)`: the outliers to mark
            `color (str)`: color of the markers (default: 'red')
            `alpha (float)`: opacity of the markers (default: 0.3)
            `**kwargs`: additional keyword arguments to pass to matplotlib.pyplot.vlines()
              to customize the outlier lines

        Returns:
        -------
            `None`

        """
        # Retrieve the variable of original time series
        series = self.data[var]
        # Set up the figure and axis
        _, axis = plt.subplots(figsize=(18, 6))
        # Plot the original time series
        axis.plot(series.index, series.values, label=series.name, color='black')
        # Plot vertical lines at the location of the outliers
        outlier_indices = outliers.index
        axis.vlines(outlier_indices, ymin=series.min(), ymax=series.max(),
                    color=color, alpha=alpha, label='Outliers', **kwargs)
        axis.set_xlabel('Date')
        axis.set_ylabel('Value')
        axis.set_title(f'Outliers Detected: {len(outliers)}')
        axis.legend(**kwargs)
        plt.show()



    def detect_outliers_in_time(
            self,
            on_stream_var: str,
            rate_var: str,
            verbose: bool = False
    ) -> Union[pd.Series, str]:
        """
        This function detects and returns outliers in the on-stream hours
          data for production and injection wells.

        Parameters:
        ----------
            - `on_stream_var (str)`: The name of the variable in the dataset that
              represents on-stream hours.
            - `rate_var (str)`: The name of the corresponding rate variable for
              the on-stream hours.
            For example, if the on-stream hours variable represents production time,
            the rate variable could be oil rate.
            Similarly, if the on-stream hours variable represents injection time for cleaning,
              the rate variable could be injected water volume.

        Returns:
        -------
        - a pandas Series containing the outliers in the on-stream hours for
          production or inejction data.
        - a string indicating that no outliers were detected.

        """
        outliers = pd.Series()

        # Cap the maximum value of on-stream hours to 24
        outliers = self.data.loc[self.data[on_stream_var] > 24, on_stream_var]

        # If the rate variable is zero, the corresponding on-stream hours should also be zero
        mask = (self.data[rate_var] == 0) & (self.data[on_stream_var] > 0)
        outliers = pd.concat([outliers, self.data.loc[mask, on_stream_var]])

        # Check if there are outliers
        if outliers.empty:

            return "No outliers detected."
        # Print the name of the variable being analyzed
        if verbose:
            print(f'Variable: {on_stream_var}')
        # Return outliers
        return outliers.sort_index()



    def detect_outliers_in_rate(
            self,
            rate_var: str,
            on_stream_var: str,
            verbose: bool = False
    ) -> Union[pd.Series, str]:
        """
        This method is used to detect outliers in production and injection rates.

        Parameters:
        ----------
            `rate_var (str)`: The name of the variable indicating
            the production rate or inejction rates.
            `on_stream_var (str)`: The name of the variable indicating
            the corresponding on-stream hours.

        Returns:
        -------
            `pd.Series`: A series containing the identified outliers
              in the production or injection rates.
            `str`: A string indicating that no outliers were detected.

        """
        outliers = pd.Series()
        # Select data where the on-stream variable is zero and
        #  the rate variable is greater than zero.
        mask = (self.data[on_stream_var] == 0) & (self.data[rate_var] > 0)
        outliers  = pd.concat([outliers, self.data.loc[mask, rate_var]])

        # Check if there are outliers
        if outliers.empty:

            return "No outliers detected."
        if verbose:
            # Print the name of the variable being analyzed
            print(f'Variable: {rate_var}')
        # Return outliers
        return outliers.sort_index()



    def detect_outliers_in_choke(
            self,
            avg_choke_var: str,
            on_stream_var: str,
            verbose: bool = False
    ) -> pd.Series:
        """
        Detects outliers in the average choke size variable in the production data.

        Parameters:
        ----------
            `avg_choke_var (str)`: The name of the column containing the average choke size data.
            `on_stream_var (str)`: The name of the column containing the on-stream hours data.

        Returns:
        -------
            `pd.Series`: A series containing the outliers detected in
              the average choke size variable.

        """
        # The average choke size should be set to zero when the well is off
        mask = (self.data[on_stream_var] == 0) & (self.data[avg_choke_var] > 0)

        outliers = self.data.loc[mask, avg_choke_var]

        # Check if there are outliers
        if outliers.empty:

            return "No outliers detected."
        # Return outliers
        if verbose:
            # Print the name of the variable being analyzed
            print(f'Variable: {avg_choke_var}')
        return outliers.sort_index()



    def __remove_extreme_outliers(
            self,
            series: pd.Series,
            thd_z_score: Union[int, float] = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Removes extreme outliers from a pandas Series object.
        Extreme outliers are considered to be values of zero or dramatic drops in variables such as
        downhole pressure, downhole temperature, or wellhead temperature.

        For example, if Downhole_temperature = [200, 201, 200, 204, 199, 160, 0, 202],
        then 0 and 160 are deemed to be outliers for this variable.

        This private method is called by detect_outliers_with_roc to detect such anomalous behavior.

        Parameters:
        ----------
            `series (pd.Series)`: The input pandas Series object
              from which to remove outliers.
            `thd_z_score (Union[int, float])`: The threshold value for
              the Z-score above which data points are
                considered extreme outliers. Defaults to 2.

        Returns:
        -------
            `Tuple[pd.Series, pd.Series]`: A tuple of two pandas Series objects:
            - The first Series is the input Series with extreme outliers removed.
            - The second Series contains the extreme outliers that were removed.

        """
        # Create a copy of the input series to avoid modifying the original.
        extreme_filtered_series = series.copy()
        # Find any existing zeros in the series and consider them as outliers.
        outliers = extreme_filtered_series[extreme_filtered_series == 0]
        # Replace zeros with NaN values so that they are excluded
        #  from the outlier detection process.
        extreme_filtered_series[extreme_filtered_series == 0] = np.nan
        # Calculate the Z-score of each data point in the series.
        mean = extreme_filtered_series.mean()
        std = extreme_filtered_series.std()
        z_score = (extreme_filtered_series - mean) / std
        # Calculate the absolute value of the Z-score.
        abs_z_score = abs(z_score)
         # Find the data points whose Z-score exceeds the threshold value
         #  and consider them as outliers.
        outliers = pd.concat([outliers,
                               extreme_filtered_series[abs_z_score > thd_z_score]])
        # Replace the extreme outliers with NaN values so that they are
        #  excluded from the returned series.
        extreme_filtered_series[abs_z_score > thd_z_score] = np.nan
        # Return the cleaned series and the extreme outliers.
        return extreme_filtered_series, outliers



    def __get_window_mean(
            self,
            i: int,
            series: pd.Series,
            window_size: int,
            verbose: bool = False
    ) -> Tuple[Union[float, int], pd.Series, int]:
        """
        Calculates the mean value of a window of data points in a pandas Series object.

        This method calculates the mean value of a window of data points
          with a size of window_size, starting
        from the index i of the input pandas Series object. If there are
          enough data points in the Series object
        to form a complete window of size window_size, the method calculates
          the mean value of that window. If
        the number of data points is less than window_size, the method merges
          the last window with the previous
        one and calculates the mean value.

        Parameters:
        ----------
            `i (int)`: The index of the first data point in the window.
            `window_size (int)`: The number of data points to include in the window.
            `series (pd.Series)`: The pandas Series object containing the data points.

        Returns:
        -------
            `Tuple[Union[float, int], pd.Series, int]`: A tuple containing:
            - The mean value of the window (either float or int).
            - The pandas Series object representing the window.
            - The index of the next data point after the window.

        """
        if i + 2 * window_size <= len(series):
            # If there are enough data points for a complete window
            # Select the window and calculate its start and end dates
            window = series.iloc[i:i + window_size]
            start_date = self.data.iloc[i].name.strftime('%Y-%m-%d')
            end_date = self.data.iloc[i + window_size].name.strftime('%Y-%m-%d')
            i += window_size
        else:
            # If there are not enough data points for a complete window
            # Merge the remaining data points with the last window
            window = series.iloc[i:]
            start_date = self.data.iloc[i].name.strftime('%Y-%m-%d')
            end_date = self.data.iloc[len(series) - 1].name.strftime('%Y-%m-%d')
            i = len(series)
        if verbose:
            print(f"The rate of change for segment [{start_date}, {end_date}]", end='')
        # Return the mean value of the window, the window itself,
        #  and the index of the next data point
        return window.mean(), window, i



    def __get_window_outliers(
            self,
            window: pd.Series,
            mean: Union[float, int],
            rate_of_change_window: Union[float, int]
    ) -> pd.Series:
        """
        Returns a Series object containing the outliers of
          a given window of data points.

        Outliers are defined as data points that fall outside of the upper and lower bounds,
          which are calculated
        as the mean of the window plus or minus the rate of change threshold.

        Parameters:
        ----------
            `window (pd.Series)`: The pandas Series object representing the window of data points.
            `mean (Union[float, int])`: The mean value of the window.
            `rate_of_change_window (Union[float, int])`: The threshold value for the rate of change.

        Returns:
        -------
            `pd.Series`: A pandas Series object containing the outliers of the input window.

        """
        upper_bound = mean + rate_of_change_window
        lower_bound = mean - rate_of_change_window
        window_outliers = window.loc[(window < lower_bound) | (window > upper_bound)]
        return window_outliers

    def __define_roc_with_quantile(
        self,
        series: pd.Series,
        outliers: pd.Series,
        window_size: int,
        thd_quantile: Union[int, float] = .98,
        verbose: bool = False
    ) -> pd.Series:
        """
        Determines the rate of change for each window of data points
          in a pandas Series object
        using a given quantile value and identifies data points that are outliers.
        This method is useful in case the user does not know the suitable rate of change
          for each window (interval)
        For each window of data points with a size of window_size, the method calculates
        the mean value and the rate of change, which is determined using the given quantile
        value. Data points that are outliers are identified by comparing their values
        to the calculated upper and lower bounds. The method returns a pandas Series object
        containing the outliers.

        Parameters:
        ----------
            `series (pd.Series)`: The pandas Series object containing the data points.
            `outliers (pd.Series)`: A pandas Series object containing the data points that were
              previously identified as extreme outliers by __remove_extreme_outliers.
            `window_size (int)`: The number of data points to include in the window.
            `thd_quantile (Union[int, float])`: The quantile value to use
              in calculating the rate of change. Defaults to 0.98.


        Returns:
        -------
            `pd.Series`: A pandas Series object containing the data points that were
                extreme outliers and  outliers indentified by quantile rate of change.

        """
        # Extract values from kwargs with default values if not provided
        rate = []
        i = 0
        while i < len(series):
            # Calculate mean value and rate of change for the window
            mean, window, i = self.__get_window_mean(i,
                                                      window_size,
                                                      series,
                                                      verbose)
            diff_window = np.abs(np.diff(window))
            rate_of_change_window = np.nanquantile(diff_window, thd_quantile)
            rate.append(rate_of_change_window)
            if verbose:
                print(f' is {np.round(rate_of_change_window, 2)}'
                       f' with a mean of {np.round(mean, 2)}')
            # Identify outliers in the window and add them to the list of outliers
            window_outliers = self.__get_window_outliers(window,
                                                         mean,
                                                         rate_of_change_window)
            outliers = pd.concat([outliers, window_outliers])
        if verbose:
            print(f'Rate of change for each window of size {window_size}'
                  f' is: {[round(x, 2) for x in rate]}')
        return outliers




    def __validate_roc_input(
            self,
            rate_of_change: Union[list, np.ndarray],
            num_windows: int,
            all_same_rate: bool
    ) -> Iterator:
        """    Validates the input for rate of change and returns an iterator
          of the rate of change values.

        Parameters:
        ----------
            `rate_of_change (Union[list, np.ndarray])`: A list or an array of
              rate of change values.
            `all_same_rate (bool)`: A flag to indicate whether all windows should
              have the same rate of change.
            `num_windows (int)`: The number of windows.

        Returns:
        -------
            Iterator: An iterator of the rate of change values.

        Raises:
        ------
            ValueError: If the input rate_of_change is not a list or an array.
            ValueError: If all_same_rate is True and rate_of_change does
              not contain exactly one value.
            ValueError: If all_same_rate is False and the length of rate_of_change
              is not equal to num_windows.

        """
        if not isinstance(rate_of_change, (list, np.ndarray)):
            raise ValueError("rate_of_change must be a list or an array.")
        if all_same_rate:
            if len(rate_of_change) != 1:
                raise ValueError('When all_same_rate is True, rate_of_change must be a'
                                 ' list or an array with a single value.')
            return iter([rate_of_change[0]] * num_windows)
        if len(rate_of_change) != num_windows:
            raise ValueError(f"Length of rate_of_change ({len(rate_of_change)})"
                                f"does not match the number of windows ({num_windows}).")
        return iter(rate_of_change)



    def __define_roc_manually(
            self,
            series: pd.Series,
            outliers: pd.Series,
            window_size: int,
            num_windows: int,
            rate_of_change: Union[list, np.ndarray],
            all_same_rate: bool = False,
            verbose: bool = False
    ) -> pd.Series:
        """
        Detect outliers based on the defined rate of change by the user.

        The method first validates if the input rate of change conforms to the expected format.
          It then calculates the mean for each window of data points, and generates a boundary
          for each window based on the user-defined rate of change and the calculated mean.
            Data points that fall outside these boundaries are identified as outliers.

        Parameters:
        ----------
            `series (pd.Series)`: A pandas Series object containing the data points.
            `outliers (pd.Series)`: A pandas Series object containing the
              data points that were previously identified as extreme outliers
                by __remove_extreme_outliers.
            `window_size (int)`: The number of data points to include in the window.
            `num_windows (int)`: The number of windows in the series.
            `rate_of_change (Union[list[float], List[int], np.ndarray[float], np.ndarray[int]])`:
                A list or array of rate of change values defined by the user.
                  If all_same_rate is True, this value
                will be used for all windows. If all_same_rate is False,
                  the rate_of_change value should be a list or
                array with length equal to num_windows.
            `all_same_rate (bool)`: If True, one user-defined value is used for all windows.
              If False, the rate_of_change value should be a list or array
                with length equal to num_windows. Default is False.
            `verbose (bool)`: If True, prints the rate of change
              for each window. Default is False.

        Returns:
        -------
            `pd.Series`: A pandas Series object containing the extreme outliers
              and outliers that fall outside the boundary defined by
                the user-defined rate of change.

        """
        # Validate the user-defined rate of change input
        rate_of_change_iter = self.__validate_roc_input(rate_of_change,
                                                        all_same_rate,
                                                        num_windows)
        # Initialize window index
        i = 0
         # Iterate through each window of the data
        while i < len(series):
            # Calculate the mean and window of data points for the current window
            mean, window, i=  self.__get_window_mean(i,
                                                     window_size,
                                                     series,
                                                     verbose)
            # Get the rate of change value for the current window
            rate_of_change_window = next(rate_of_change_iter)
            if verbose:
                print(f' is {np.round(rate_of_change_window, 2)} with a mean of {np.round(mean,2)}')
            # Get the outliers for the current window
            window_outliers = self.__get_window_outliers(window, mean, rate_of_change_window)
            # Add the window outliers to the list of outliers
            outliers = pd.concat([outliers, window_outliers])
             # Return the list of all outliers detected
        return outliers



    def detect_outliers_with_roc(
            self,
            series: str,
            window_size: int,
            thd_z_score: int = 2,
            thd_quantile: Union[float, int] = 0.98,
            rate_of_change: Union[list, np.ndarray] = None,
            all_same_rate: bool = False,
            verbose: bool = False
    ) -> pd.Series:
        """
        Detects outliers in variables(e.g. downhole pressure, downhole temperature,
          or wellhead temperature) by creating boundaries using the rate of change,
            either defined by the user or calculated based on the rate of change
              of 98% of the data points in each window.

        The user has three options to determine the rate of change:
        - Input the expected rate of change for each window.
        - Use one value of rate of change for all windows.
        - Determine the rate of change for each window based on the quantile.

        Parameters:
        ----------
            `series (str)`: The name of the time series variable to be analyzed.
            `window_size (int)`: The number of data points to include in the window.
            `thd_z_score (int)`: The Z-score threshold value used to remove
              extreme outliers. Default is 2.
            `thd_quantile (Union[float, int])`: The quantile threshold used to identify
            outliers when rate of change is not used. Default is 0.98.
            `rate_of_change (Optional[Union[List[Union[float, int]],
              np.ndarray[Union[float, int]]]])`:
            A list or array of rate of change values defined by the user.
              If None, the quantile threshold method is used. Default is None.
            `all_same_rate (bool)`: If True,
              the same user-defined rate of change value is used for all windows.
            The user should input only one value, which will be used to calculate the outliers
            for all windows. If False, the rate_of_change value should be a list or array with
            length equal to the number of windows. The default value is False.
            `verbose (bool)`: If True, the function prints
              the rate of change value for each window. Default is False.

        Returns:
        ----------
            `pd.Series`: A pandas Series object containing the extreme outliers and
            outliers that fall outside the boundary
              defined by the rate of change or quantile threshold.

        """
        # Retrieve the variable
        series = self.data[series]
        # Determine the number of intervals (windows) in the data
        num_windows = int(len(series) / window_size)
        if verbose:
            print(f'Number of intervals: {num_windows}')
        # Check if the input series is a pandas Series object
        if not isinstance(series, pd.Series):
            raise TypeError(f"Input 'series' must be a pandas Series object, not {type(series)}")
        # Remove extreme outliers from the series using the remove_extreme_outliers function
        extreme_filtered_series, outliers = self.__remove_extreme_outliers(series, thd_z_score)
        # Detect outliers using either rate of change or quantile threshold
        if all_same_rate and rate_of_change is None:
            raise TypeError("If all rates of change are the same, "
                            "the rate_of_change argument must be provided")
        if rate_of_change is None:
            outliers = self.__define_roc_with_quantile(extreme_filtered_series,
                                                       outliers, window_size,
                                                       thd_quantile,
                                                       verbose)
        else:
            outliers = self.__define_roc_manually(extreme_filtered_series,
                                                  outliers,
                                                  window_size,
                                                  num_windows,
                                                  rate_of_change,
                                                  all_same_rate,
                                                  verbose)
        # Check if there are outliers
        if outliers.empty:

            return "No outliers detected."
        # Return outliers
        if verbose:
            # Print the name of the variable being analyzed
            print(f'Variable: {series.name}')
        return outliers.sort_index()



    def treat_outliers_in_time(
            self,
            data: pd.DataFrame,
            on_stream_var: str,
            rate_var: str,
            intp_method: str = 'linear'
    )-> pd.Series:
        """
        Treats outliers in the on-stream hours variable by
          capping the maximum value at 24 hours
        and setting the on-stream hours to zero if the rate is zero.

        Parameters:
        ----------
            `data (pd.DataFrame)`: The dataframe containing
              the on-stream hours and rate variables.
            `on_stream_var (str)`: The name of the on-stream hours variable.
            `rate_var (str)`: The name of the rate variable.
            `intp_method (str)`: The interpolation method used to fill
              in missing values. Default is 'linear'.

        Returns:
        ----------
            `pd.Series`: A pandas Series object containing the treated on-stream hours variable.

        """
        # Cap the maximum value of on-stream hours to 24
        data.loc[data[on_stream_var] > 24, on_stream_var] = 24

        # If the rate variable is zero, the corresponding on-stream hours should also be zero
        mask = (data[rate_var] == 0) & (data[on_stream_var] > 0)
        data.loc[mask, on_stream_var] = 0
        data[on_stream_var] = data[on_stream_var].interpolate(
                                  method=intp_method, limit_direction='both')
        # Return the on-stream hours variable
        return data[on_stream_var]



    def treat_outliers_in_rate(
            self,
            data: pd.DataFrame,
            rate_var: str,
            on_stream_var: str,
            intp_method: str = 'linear'
    )-> pd.Series:
        """
        Treats outliers in the rate variable by setting the rate to
          zero if the on-stream hours is zero.

        Parameters:
        ----------
            `data (pd.DataFrame)`: The dataframe containing the on-stream hours and rate variables.
            `rate_var (str)`: The name of the rate variable.
            `on_stream_var (str)`: The name of the on-stream hours variable.
            `intp_method (str)`: The interpolation method used to
              fill in missing values. Default is 'linear'.

        Returns:
        ----------
            `pd.Series`: A pandas Series object containing the treated rate variable.

        """
        # If the on-stream hours variable is zero, the corresponding rate should also be zero
        mask = (data[on_stream_var] == 0) & (data[rate_var] > 0)
        # Set the rate to zero
        data.loc[mask, rate_var] = 0
        # Interpolate the rate variable
        data[rate_var] = data[rate_var].interpolate(method=intp_method, limit_direction='both')
        # Return the rate variable
        return data[rate_var]



    def treat_outliers_with_roc(
            self,
            data: pd.DataFrame,
            var: str,
            roc_outliers: pd.Series,
            intp_method: str = 'linear'
    )-> pd.Series:
        """
        Treats outliers in a time series variable by interpolating the values of the outliers.

        Parameters:
        ----------
            `data (pd.DataFrame)`: The dataframe containing the variable to be analyzed.
            `var (str)`: The name of the variable to be treated.
            `roc_outliers (pd.Series)`: A pandas Series object containing
              the outliers detected using the rate of change method.
            `intp_method (str)`: The interpolation method to be used. Default is 'linear'.

        Returns:
        ----------
            `pd.Series`: A pandas Series object containing the variable with
              the outliers interpolated.

        """
        # Setting the outliers to NaN
        data[var][roc_outliers.index] = np.nan
        # Interpolating the values of the outliers
        data[var] = data[var].interpolate(method=intp_method, limit_direction='both')
        # Return the variable
        return data[var]

    def treat_outliers_in_choke(
            self,
            data: pd.DataFrame,
            avg_choke_var: str,
            on_stream_var: str,
            intp_method: str = 'linear'
    )-> pd.Series:
        """
        Cleans outliers in the average choke variable by setting the
          average choke to zero if the on-stream hours is zero.

        Parameters:
        ----------
            `data (pd.DataFrame)`: The dataframe containing the on-stream hours
              and average choke variables.
            `avg_choke_var (str)`: The name of the average choke variable.
            `on_stream_var (str)`: The name of the on-stream hours variable.
            `intp_method (str)`: The interpolation method used to fill
              in missing values. Default is 'linear'.

        Returns:
        ----------
            `pd.Series`: A pandas Series object containing the cleaned average
              choke variable.

        """

        # If the on-stream hours variable is zero, the corresponding average
        #  choke should also be zero
        mask = (data[on_stream_var] == 0) & (data[avg_choke_var] > 0)
        # Set the average choke to zero
        data.loc[mask, avg_choke_var] = 0
        # Interpolate the average choke variable
        data[avg_choke_var] = data[avg_choke_var].interpolate(
                                  method=intp_method, limit_direction='both')
        # Return the average choke variable
        return data[avg_choke_var]
