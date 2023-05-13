
# Multivariate Time Series Forecasting of Oil Production Based on Ensemble Deep Learning and Genetic Algorithm


This project focuses on developing a forecasting model for oil production using advanced machine learning techniques and optimization algorithms. The project includes the development of a Genetic Algorithm- Temporal Convolutional Neural Network- Long Short-Term Memory (GA-TCN-LSTM) ensemble model, as well as benchmarking against conventional models such as Recurrent Neural Network (RNN), Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), and Temporal Convolutional Network (TCN).

Additionally, the project includes exploratory data analysis and data cleaning using our custom-built `odc module`. The  module is designed to detect, visualize, and treat outliers in oil well datasets and is readily available in the odc repository.



## Motivation
Oil production forecasting is a critical task for many oil and gas companies, governments, and policy-makers. Accurate forecasts are essential for planning and decision-making, such as determining production rates, managing inventory, and estimating future revenue.

Conventional oil production forecasting methods have limitations due to complex data, high uncertainty, and failure to reflect the actual system and dynamic changes.. Therefore, the use of advanced machine learning techniques and optimization algorithms can improve the accuracy of forecasts by accounting for these complexities and identifying the optimal combination of hyperparameters for each model. This project aims to provide decision-makers with better information to make informed decisions and improve the overall forecasting process.
## Workflow
In this project, we followed a systematic approach to obtain and process the required oil well data. Initially, we extracted the necessary data from the `raw_data.xlsx` file. Next, we conducted exploratory data analysis (EDA) to better understand the characteristics of the data. The resulting dataset was then saved with the name `F_14.csv,` which includes production and injection data that significantly affect the well's production. We performed data cleaning on `F_14.csv` using our custom module, `odc.py,` and saved the cleaned data to a file named `cleaned_F_14.csv.`

After cleaning the data, we used it as input to our proposed and reference models for forecasting. All datasets used in this project can be found in the `/datasets` folder, and the EDA and data cleaning process can be found in the `/data_preprocessing` folder. Finally, the modeling process, including the proposed and reference models, is documented in the `/modeling` folder.

## Directory Tree
```
├── .gitignore
├── requirements.txt
├── README.md
├── license
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

## View Notebooks in Colab

| Notebook | Colab Link |
| -------- | ---------- |
| Exploratory Data Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashrafalaghbari/GA-TCN-LSTM/blob/main/data_preprocessing/eda.ipynb) |
| Data Cleaning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashrafalaghbari/GA-TCN-LSTM/blob/main/data_preprocessing/data_cleaning.ipynb) |
| Proposed Model (GA-TCN-LSTM) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashrafalaghbari/GA-TCN-LSTM/blob/main/modeling/proposed_model/GA_TCN_LSTM_model.ipynb) |
| Reference Model (TCN) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashrafalaghbari/GA-TCN-LSTM/blob/main/modeling/reference_models/TCN_model.ipynb) |
| Reference Model (LSTM) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashrafalaghbari/GA-TCN-LSTM/blob/main/modeling/reference_models/LSTM_model.ipynb) |
| Reference Model (GRU)| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashrafalaghbari/GA-TCN-LSTM/blob/main/modeling/reference_models/GRU_model.ipynb) |
| Reference Model (RNN) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashrafalaghbari/GA-TCN-LSTM/blob/main/modeling/reference_models/RNN_model.ipynb) |





## Dataset

The `raw_data.xlsx` file used in this project was provided by Equinor (formerly known as Statoil) and is available on their website as part of their Volve Field Data sharing initiative.

To access the dataset, please visit the following link: https://www.equinor.com/energy/volve-data-sharing. Once on the page, select the "Go to the Volve dataset: data.equinor.com" option and follow the instructions to obtain the `raw_data.xlsx` file.

Please note that the raw data is subject to the terms and conditions outlined on the Equinor website.

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

[MIT](https://github.com/ashrafalaghbari/GA-TCN-LSTM/blob/main/license)


## Contact

If you have any questions or encounter any issues running this project, please feel free to [open an issue](https://github.com/ashrafalaghbari/Data-Viz/issues) or contact me directly at [ashrafalaghbari@hotmail.com](mailto:ashrafalaghbari@hotmail.com). I'll be happy to help!



