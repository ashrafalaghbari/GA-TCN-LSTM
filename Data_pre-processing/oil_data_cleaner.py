class OilDataCleaner:
    def __init__(self, data):

        self.data = data.copy()
    
    def __visualize_outliers(self, series, outliers):

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(18, 6))

        # Plot the original time series
        ax.plot(series.index, series.values, color='black', label=series.name)

        # Plot vertical lines at the location of the outliers
        outlier_indices = outliers.index
        ax.vlines(outlier_indices, ymin=series.min(), ymax=series.max(), color='red', label='Outliers',  alpha=0.3)

        # Add labels and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(f'Outliers Detected: {len(outliers)}')
        ax.legend(loc='lower right')

        # Show the plot
        plt.show()
        
    def find_on_stream_hrs_outliers(self, on_stream_var, rate_var):
        # Cap the maximum value of ON_STREAM_HRS to 24
        outliers = self.data.loc[self.data[on_stream_var] > 24, on_stream_var]
        mask = (self.data[rate_var] == 0) & (self.data[on_stream_var] > 0)
        outliers = pd.concat([outliers, self.data.loc[mask, on_stream_var]])
        self.__visualize_outliers(self.data[on_stream_var], outliers)
        return outliers

    def find_rates_outliers(self, rate_var, on_stream_var):
        
        mask = (self.data[on_stream_var] == 0) & (self.data[rate_var] > 0) 
        outliers = self.data.loc[mask , rate_var]
        self.__visualize_outliers(self.data[rate_var], outliers)
        return outliers 

 

    def find_choke_outliers(self, avg_choke_var, on_stream_var):
        # The average choke size should be set to zero when the well is off
        mask = self.data[on_stream_var] == 0
        outliers = self.data.loc[mask, avg_choke_var] 
        self.__visualize_outliers(self.data[avg_choke_var], outliers)
        return outliers


    def __remove_extreme_outliers(self, series, thd_z_score=2):
        extreme_filtered_series = series.copy()
        outliers = extreme_filtered_series[extreme_filtered_series == 0]
        extreme_filtered_series[extreme_filtered_series == 0] = np.nan
        z_score = (extreme_filtered_series - extreme_filtered_series.mean()) / extreme_filtered_series.std()
        abs_z_score = abs(z_score)
        outliers = pd.concat([outliers, extreme_filtered_series[abs_z_score > thd_z_score]])
        extreme_filtered_series[abs_z_score > thd_z_score] = np.nan
        return extreme_filtered_series, outliers

    def __get_window_mean(self, i, window_size, series):
        if i + 2 * window_size <= len(series):
            # If the last window is smaller than window_size, add it to the previous window
            window = series.iloc[i:i + window_size]
            start_date = self.data.iloc[i].name.strftime('%Y-%m-%d')
            end_date = self.data.iloc[i + window_size].name.strftime('%Y-%m-%d')
        else:
            window = series.iloc[i:]
            start_date = self.data.iloc[i].name.strftime('%Y-%m-%d')
            end_date = self.data.iloc[len(series) - 1].name.strftime('%Y-%m-%d')
            i = len(series)

        print(f"The rate of change for segment [{start_date}, {end_date}]", end='')

        return window.mean(), window, i


    def __get_window_outliers(self, window, mean, rate_of_change_window):
        upper_bound = mean + rate_of_change_window
        lower_bound = mean - rate_of_change_window
        window_outliers = window.loc[(window < lower_bound) | (window > upper_bound)]
        return window_outliers

    def __detect_outliers_with_thd_quantile(self, series, outliers, window_size, thd_quantile=.98, segments_start_date=None,
                                    segments_end_date=None, segments_rate=None):
        for i in range(0, len(series), window_size):
            
            mean, window, i =  self.__get_window_mean(i, window_size, series)
            diff_window = np.abs(np.diff(window))
            rate_of_change_window = np.nanquantile(diff_window, thd_quantile)

            print(f' is {rate_of_change_window}', end='')
            print(f" with a mean of {mean}")
            window_outliers = self.__get_window_outliers(window, mean, rate_of_change_window)
            outliers = pd.concat([outliers, window_outliers])
            # If there are less than window_size elements left, add them to the previous window
            if i == len(series) :
                break
        return outliers

    def __validate_rate_of_change(self, rate_of_change, all_same_rate, num_windows):
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

    def __detect_outliers_with_rate_of_change(self, series, outliers, window_size, rate_of_change, all_same_rate=False):
        num_windows = int(np.ceil(len(series) / window_size))
        rate_of_change_iter = self.__validate_rate_of_change(rate_of_change, all_same_rate, num_windows)
        for i in range(0, len(series), window_size):
            mean, window, i =  self.__get_window_mean(i, window_size, series)
            rate_of_change_window = next(rate_of_change_iter)
            print(f' is {rate_of_change_window}', end='')
            print(f" with a mean of {mean}")
            window_outliers = self.__get_window_outliers(window, mean, rate_of_change_window)
            outliers = pd.concat([outliers, window_outliers])
            if i == len(series):
                break
        
        return outliers

    def find_rate_of_change_outlier(self, series, window_size, thd_z_score=2, thd_quantile=0.98, rate_of_change=None, all_same_rate=False):

        series = self.data[series]
        # Check if the input series is a pandas Series object
        if not isinstance(series, pd.Series):
            raise TypeError(f"Input 'series' must be a pandas Series object, not {type(series)}")  

        # Remove extreme outliers from the series using the remove_extreme_outliers function
        extreme_filtered_series, outliers = self.__remove_extreme_outliers(series) 

        
        # Detect outliers using either rate of change or quantile threshold
        if rate_of_change is None:
            outliers = self.__detect_outliers_with_thd_quantile(extreme_filtered_series, outliers, window_size, thd_quantile)
            self.__visualize_outliers(series, outliers) 
        else:
            outliers = self.__detect_outliers_with_rate_of_change(extreme_filtered_series, outliers, window_size, rate_of_change, all_same_rate)  
            self.__visualize_outliers(series, outliers)
        # Sort and return the outliers
        return outliers.sort_index() 

