from typing import Union, Tuple, Optional, List, Iterator
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


class OilDataCleaner:

    """
    This class is designed to identify and visualize outliers in time series data from oil wells.
    We have tested the outlier detection capability of this class on a daily time series dataset
      from the VOLVE field provided by Equinor (formerly known as Statoil).

    The class can clean four types of variables:

    - `on_stream_var`: Indicates the on-stream hours for production or injection wells.
    - `rate_var`: Indicates the production rate, such as oil, water, or gas, and
      can also be used for injection rate, such as water injected volume.
    - `avg_choke_var`: Indicates the average choke size (valve) that controls the production rates.
    - `roc`: Denotes the rate of change. The `detect_outliers_with_roc` function
      can clean variables that behave in a relatively stable or gradual manner (e.g. average reservoir pressure
        and temperature and average wellhead temperature as well). 

    This function works by identifying sudden movements that do not conform to 
    the distribution of observations within the specified window. Each window contains
      a certain number of observations selected based on the window size, which can be defined by the user.

      The default values for all methods have been determined based on experimental results
        and have been optimized for outlier detection.

    Parameters:
        data (pd.DataFrame): The input DataFrame that contains the oil time series data to be cleaned.

    Attributes:
        data (pd.DataFrame): A copy of the input data.
    """

  
    def __init__(self, data: pd.DataFrame) -> None:
        # Initializes the class instance with the input data passed as a pandas DataFrame
        # A copy of the input data is created to avoid modifying the original data outside of the class
        self.data = data.copy()

    def plot_outliers(
            self,
            var: str,
            outliers: pd.Series, 
            legend_loc: Optional[str]=None, 
            color: str ='red', 
            alpha: Union[float, int] =0.3, 
            **kwargs
    ) -> None:

        """
        Visualizes the outliers in a time series variable.

        Parameters:
            var (str): the name of the variable to visualize
            outliers (pd.Series): the outliers to mark
            color (str): color of the markers (default: 'red')
            alpha (float): opacity of the markers (default: 0.3)
            **kwargs: additional keyword arguments to pass to matplotlib.pyplot.vlines()
              to customize the outlier lines

        Returns:
            None
        """
        # Retrieve the variable of original time series 
        series = self.data[var] 
        # Set up the figure and axis
        _, ax = plt.subplots(figsize=(18, 6))
        # Plot the original time series
        ax.plot(series.index, series.values, label=series.name, color='black')
        # Plot vertical lines at the location of the outliers
        outlier_indices = outliers.index
        ax.vlines(outlier_indices, ymin=series.min(), ymax=series.max(),
                    color=color, alpha=alpha, label='Outliers', **kwargs)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(f'Outliers Detected: {len(outliers)}')
        ax.legend(loc=legend_loc)
        plt.show()

    def detect_outliers_in_time(
            self, 
            on_stream_var: str, 
            rate_var: str
    ) -> pd.Series:
        """
        This function detects and returns outliers in the on-stream hours data for production and injection wells.
        
        Parameters:
        - on_stream_var: a string representing the name of the on-stream hours variable in the dataset.
        - rate_var: a string representing the name of the corresponding rate variable (e.g. oil, water, or gas production) in the dataset.
        
        Returns:
        - a pandas Series containing the outliers in the on-stream hours for production or inejction data.
        """
        # Print the name of the variable being cleaned
        print('Variable: {}'.format(on_stream_var))
        # Cap the maximum value of ON_STREAM_HRS to 24
        outliers = self.data.loc[self.data[on_stream_var] > 24, on_stream_var]
        # if the rate variable is zero, the correponding on-stream-hrs should be zero as well
        mask = (self.data[rate_var] == 0) & (self.data[on_stream_var] > 0)
        outliers = self.data.loc[mask, on_stream_var]
        # Return the outliers
        return outliers

    def detect_outliers_in_rate(
            self, 
            rate_var: str, 
            on_stream_var: str
    ) -> pd.Series:
        """
        This method is used to detect outliers in production and injection rates.

        Parameters:
            rate_var (str): The name of the variable indicating the production rate or inejction rates.
            on_stream_var (str): The name of the variable indicating the corresponding on-stream hours.

        Returns:
            pd.Series: A series containing the identified outliers in the production or injection rates.

        """
        # Print the name of the variable being cleaned
        print('Variable: {}'.format(rate_var))
        # Select data where the on-stream variable is zero and the rate variable is greater than zero
        mask = (self.data[on_stream_var] == 0) & (self.data[rate_var] > 0)
        outliers = self.data.loc[mask, rate_var]
        # Return the outliers
        return outliers

    def detect_outliers_in_choke(
            self, 
            avg_choke_var: str, 
            on_stream_var: str
    ) -> pd.Series:
        """
        Detects outliers in the average choke size variable in the production data.
        
        Parameters:
            avg_choke_var (str): The name of the column containing the average choke size data.
            on_stream_var (str): The name of the column containing the on-stream hours data.
        
        Returns:
            pd.Series: A series containing the outliers detected in the average choke size variable.
        """
        # Print the name of the variable being cleaned
        print('Variable: {}'.format(avg_choke_var))
        # The average choke size should be set to zero when the well is off
        mask = self.data[on_stream_var] == 0
        outliers = self.data.loc[mask, avg_choke_var]
        # Return the outliers
        return outliers

    def __remove_extreme_outliers(
            self, 
            series: pd.Series, 
            thd_z_score: Union[int, float] = 2
    ) -> Tuple[pd.Series, pd.Series]:

        extreme_filtered_series = series.copy()
        outliers = extreme_filtered_series[extreme_filtered_series == 0]
        extreme_filtered_series[extreme_filtered_series == 0] = np.nan
        z_score = (extreme_filtered_series - extreme_filtered_series.mean()) / extreme_filtered_series.std()
        abs_z_score = abs(z_score)
        outliers = pd.concat([outliers, extreme_filtered_series[abs_z_score > thd_z_score]])
        extreme_filtered_series[abs_z_score > thd_z_score] = np.nan
        return extreme_filtered_series, outliers

    def __get_window_mean(
            self, 
            i: int, 
            window_size: int, 
            series: pd.Series
    ) -> Tuple[Union[float, int], pd.Series, int]:
        """"""
        if i + 2 * window_size <= len(series):
            # If the last window is smaller than window_size, add it to the previous window
            window = series.iloc[i:i + window_size]
            start_date = self.data.iloc[i].name.strftime('%Y-%m-%d')
            end_date = self.data.iloc[i + window_size].name.strftime('%Y-%m-%d')
            i += window_size
        else:
            # If there are less than window_size elements left, add them to the previous window
            window = series.iloc[i:]
            start_date = self.data.iloc[i].name.strftime('%Y-%m-%d')
            end_date = self.data.iloc[len(series) - 1].name.strftime('%Y-%m-%d')
            i = len(series)
        print(f"The rate of change for segment [{start_date}, {end_date}]", end='')
        return window.mean(), window, i


    def __get_window_outliers(
            self, 
            window: pd.Series, 
            mean: Union[float, int], 
            rate_of_change_window: Union[float, int]
    ) -> pd.Series:
        
        upper_bound = mean + rate_of_change_window
        lower_bound = mean - rate_of_change_window
        window_outliers = window.loc[(window < lower_bound) | (window > upper_bound)]
        return window_outliers

    def __define_roc_with_quantile(
            self, 
            series: pd.Series, 
            outliers: pd.Series, 
            window_size: int,
            thd_quantile: Union[int, float] = .98
    ) -> pd.Series:
        """"""
        rate = list()
        i = 0
        while i < len(series):
            mean, window, i =  self.__get_window_mean(i, window_size, series)
            diff_window = np.abs(np.diff(window))
            rate_of_change_window = np.nanquantile(diff_window, thd_quantile)
            rate.append(rate_of_change_window)
            print(f' is {rate_of_change_window} with a mean of {mean}')
            window_outliers = self.__get_window_outliers(window, mean, rate_of_change_window)
            outliers = pd.concat([outliers, window_outliers])
        print('Rate for each window of {} is: {}'.format(window_size,[round(x, 2) for x in rate]))
        return outliers

    def __validate_roc_input(
            self,
            rate_of_change: Union[list, np.ndarray],
            all_same_rate: bool,
            num_windows: int
    ) -> Iterator:
        
        if not isinstance(rate_of_change, (list, np.ndarray)):
            raise ValueError("rate_of_change must be a list or an array.")
        if all_same_rate:
            if len(rate_of_change) != 1:
                raise ValueError("When all_same_rate is True, rate_of_change must be a list or an array with a single value.")  
            return iter([rate_of_change[0]] * num_windows)
        else:
            if len(rate_of_change) != num_windows:
                raise ValueError(f"Length of rate_of_change ({len(rate_of_change)})"
                                    f"does not match the number of windows ({num_windows}).")
            
            return iter(rate_of_change)

    def __define_roc_manually(
            self,series: pd.Series,
            outliers: pd.Series,
            window_size: int,
            num_windows: int,
            rate_of_change: Union[list[float], List[int], np.ndarray[float], np.ndarray[int]],
            all_same_rate: bool= False
    ) -> pd.Series:

        rate_of_change_iter = self.__validate_roc_input(rate_of_change, all_same_rate, num_windows)
        i = 0
        while i < len(series):
            mean, window, i=  self.__get_window_mean(i, window_size, series)
            rate_of_change_window = next(rate_of_change_iter)
            print(f' is {rate_of_change_window} with a mean of {mean}')
            window_outliers = self.__get_window_outliers(window, mean, rate_of_change_window)
            outliers = pd.concat([outliers, window_outliers])
        return outliers

    def detect_outliers_with_roc(
            self,
            series: str,
            window_size: int,
            thd_z_score: int = 2,
            thd_quantile: Union[float, int] = 0.98,
            rate_of_change: Optional[Union[List[Union[float, int]], np.ndarray[Union[float, int]]]] = None,
            all_same_rate: bool = False,
    ) -> pd.Series:
        """
        Detects outliers in a time series using either a rate of change or quantile threshold method.

        Parameters:
            series (str): the name of the variable to detect outliers in
            window_size (int): the size of the rolling window used for rate of change calculation (default=10)
            thd_z_score (int): the threshold for extreme outliers (default=2)
            thd_quantile (Union[float, int]): the threshold for quantile threshold detection (default=0.98)
            rate_of_change (Optional[Union[List[Union[float, int]], np.ndarray[Union[float, int]]]]): the threshold 
                for rate of change detection; if None, quantile threshold detection is used (default=None)
            all_same_rate (bool): if True, all rates of change are the same, and the rate_of_change argument 
                must be provided (default=False)

        Returns:
            pd.Series: a series containing the detected outliers
        """
        print('Variable: {}'.format(series))
        series = self.data[series]
        num_windows = int(len(series) / window_size)
        print(f'Number of intervals: {num_windows}')

        # Check if the input series is a pandas Series object
        if not isinstance(series, pd.Series):
            raise TypeError(f"Input 'series' must be a pandas Series object, not {type(series)}")  

        # Remove extreme outliers from the series using the remove_extreme_outliers function
        extreme_filtered_series, outliers = self.__remove_extreme_outliers(series, thd_z_score) 

                
        # Detect outliers using either rate of change or quantile threshold
        if all_same_rate and rate_of_change is None:
            raise TypeError("If all rates of change are the same, the rate_of_change argument must be provided")
        elif rate_of_change is None:
            outliers = self.__define_roc_with_quantile(extreme_filtered_series, outliers, window_size, thd_quantile)
        else:
            outliers = self.__define_roc_manually(extreme_filtered_series, outliers, window_size,
                                                  num_windows, rate_of_change, all_same_rate)  
        # Sort and return the outliers
        return outliers.sort_index()

