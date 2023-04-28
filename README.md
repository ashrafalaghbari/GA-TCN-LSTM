
# MULTIVARIATE TIME SERIES FORECASTING OF OIL PRODUCTION BASED ON ENSEMBLE DEEP LEARNING AND GENETIC ALGORITHM

This project focuses on developing a forecasting model for oil production using advanced machine learning techniques and optimization algorithms. The project includes the development of a Genetic Algorithm- Temporal Convolutional Neural Network- Long Short-Term Memory (GA-TCN-LSTM) ensemble model, as well as benchmarking against conventional models such as Recurrent Neural Network (RNN), Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), and Temporal Convolutional Network (TCN).

Additionally, the project includes exploratory data analysis and data cleaning using our custom-built oil_data_cleaner module. The  module is designed to detect, visualize, and treat outliers in oil well datasets.



## Motivation
Oil production forecasting is a critical task for many oil and gas companies, governments, and policy-makers. Accurate forecasts are essential for planning and decision-making, such as determining production rates, managing inventory, and estimating future revenue.

Conventional oil production forecasting methods have limitations due to the complexity and non-linear relationships in the data. Therefore, the use of advanced machine learning techniques and optimization algorithms can improve the accuracy of forecasts by accounting for these complexities and identifying the optimal combination of hyperparameters for each model. This project aims to provide decision-makers with better information to make informed decisions and improve the overall forecasting process.
## Methodology
In this project, we followed a systematic approach to obtain and process the required oil well data. Initially, we extracted the necessary data from the `raw_data.xlsx` file. Next, we conducted exploratory data analysis (EDA) to better understand the characteristics of the data. The resulting dataset was then saved with the name `F_14.csv,` which includes production and injection data that significantly affect the well's production. We performed data cleaning on `F_14.csv` using our custom module, `oil_data_cleaner.py,` and saved the cleaned data to a file named `cleaned_F_14.csv.`

After cleaning the data, we used it as input to our proposed and reference models for forecasting. All datasets used in this project can be found in the `/datasets` folder, and the EDA and data cleaning process can be found in the `/data_preprocessing` folder. Finally, the modeling process, including the proposed and reference models, is documented in the `/modeling` folder.

## Directory Tree
```
├── .gitignore
├── requirements.txt
├── README.md
├── data_preprocessing
│   ├── eda
│   └── data_cleaning
├── datasets
│   ├── raw_dat.xlsx
│   ├── F_14.csv
│   └── cleaned_F_14.csv
└── modeling
    ├── proposed_model
    │   ├── GA-TCN-LSTM_model.ipynb
    └── reference_models
        ├── GRU_model.ipynb
        ├── LSTM_model.ipynb
        ├── RNN_model.ipynb
        ├── TCN_model.ipynb




```
## Evaluation

| Model          | RMSE, bbl | wMAPE, % | MAE, bbl | R<sup>2</sup> score |
|----------------|-----------|----------|----------|-----------|
| **GA-TCN-LSTM**     | **199.39**    | **5.13**     | **117.11**   | **0.93**      |
| TCN            | 213.22    | 5.36     | 122.72   | 0.92      |
| LSTM           | 216.00    | 5.84     | 133.52   | 0.91      |
| GRU            | 209.33    | 5.48     | 125.06   | 0.92      |
| RNN            | 214.71    | 5.66     | 129.36   | 0.92      |

<img src="https://user-images.githubusercontent.com/98224412/235143302-00ebbd39-c977-4ce5-8b3c-abcd3a528c70.jpg" alt="GA-TCN-LSTM actual and predicted values on training and testing sets" width="700" height="400">






## Dataset

The `raw_data.xlsx` file used in this project was provided by Equinor (formerly known as Statoil) and is available on their website as part of their Volve Field Data sharing initiative.

To access the dataset, please visit the following link: https://www.equinor.com/energy/volve-data-sharing. Once on the page, select the "Go to the Volve dataset: data.equinor.com" option and follow the instructions to obtain the `raw_data.xlsx` file.

Please note that the raw data is subject to the terms and conditions outlined on the Equinor website.
## Installation

To use the GA-TCN-LSTM forecasting model and other components of this project, please follow the steps below:

1. Clone the GitHub repository to your local machine using the following command in your terminal or command prompt:

```bash
git clone https://github.com/ashrafalaghbari/GA-TCN-LSTM.git
```

You may need to install `Git` on your system if it's not already installed to clone the GitHub repository. If you don't want to use `Git` to clone the GitHub repository, you can download the project's source code as a ZIP archive from the project's GitHub page.

- Click on the green `Code` button, and then click on `Download ZIP` to download the project as a ZIP archive.
- Extract the ZIP archive to a folder on your local machine.
- Open your terminal or command prompt, and navigate to the folder - where you extracted the ZIP archive.
- Continue with step 2.

2. Set up the required environment by installing the necessary packages in `requirements.txt`. Users can install the dependencies by running:

```bash
pip install -r requirements.txt
```
3. Run the Jupyter files and the `oil_data_cleaner.py` module.

If you only want to use the `oil_data_cleaner` module, click on the `oil_data_cleaner.py` file and copy and paste the code into a text editor. Save the file with a `.py` extension and install the necessary dependencies, which are `numpy, pandas, and matplotlib.pyplot` by running:
```bash
pip install numpy pandas matplotlib.pyplot
```
4. To use the developed module for outliers treatment, import the `OilDataCleaner class` in your Python code or Jupyter Notebook by including the following statement at the beginning of your script:

```bash
from oil_data_cleaner import OilDataCleaner
```
Examples:

```bash
# Create an instance of the OilDataCleaner class
clean = OilDataCleaner(df)

# Detect outliers in BORE_GAS_VOL variable
BORE_GAS_VOL_outliers = clean.detect_outliers_in_rate('BORE_GAS_VOL', 'ON_STREAM_HRS')

# Plot the detected outliers
clean.plot_outliers('BORE_GAS_VOL', BORE_GAS_VOL_outliers)

# Treat the outliers
df['BORE_GAS_VOL'] = clean.treat_outliers_in_rate(data, 'BORE_GAS_VOL', 'ON_STREAM_HRS')
```

To use the `OilDataCleaner module`, users can refer to the file itself as everything is documented there. To see examples of using this module, users can refer to the data cleaning file in `/data_preprocessing/data_cleaning.ipynb`, as it was used to clean the dataset and exemplify the usage.

That's it! You should now be able to use the `GA-TCN-LSTM` forecasting model, run the Jupyter Notebooks, and use the `OilDataCleaner` module.
## Tech Stack



GA-TCN-LSTM
=======




[![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.10.7-blue.svg)](https://www.python.org/)


[<img target="_blank" src="https://www.fullstackpython.com/img/logos/scipy.png" width=100>](https://scipy.org/) [<img target="_blank" src="https://keras.io/img/logo.png" width=100>](https://keras.io/) [<img target="_blank" src="https://matplotlib.org/stable/_static/logo2_compressed.svg" width=100>](https://matplotlib.org/stable/index.html) [<img target="_blank" src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" width=100>](https://seaborn.pydata.org/)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" width=100>](https://numpy.org/)

 [<img target="_blank" src="https://pandas.pydata.org/static/img/pandas.svg" width=100>](https://pandas.pydata.org/) [<img target="_blank" src="https://scikit-learn.org/stable/_images/scikit-learn-logo-notext.png" width=100>](https://scikit-learn.org/stable/)

  [<img target="_blank" src="https://www.statsmodels.org/stable/_images/statsmodels-logo-v2.svg" width=100>](https://www.statsmodels.org/stable/index.html) [<img target="_blank" src="https://www.tensorflow.org/images/tf_logo_social.png" width=100>](https://www.tensorflow.org/)




## Credits
- [Simple Genetic Algorithm From Scratch in Python](https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/)
- [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
## License

[MIT](https://choosealicense.com/licenses/mit/)


## Contributing

Contributions are always welcome!

See [contributing.md](https://github.com/ashrafalaghbari/GA-TCN-LSTM/blob/main/contributing.md) for ways to get started.



