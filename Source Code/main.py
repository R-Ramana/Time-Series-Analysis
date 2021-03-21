import eval_model
from sklearn.model_selection import train_test_split
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

import csv

import statsmodels.tsa.api as smt
import statsmodels as sm

# to ignore warnings thrown by deprecated models of statmodels to use updated ones
import warnings
warnings.filterwarnings("ignore")

import metrics
import data_models as dm

from eval_model import model_metrics

from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression

from pylab import rcParams

def main_fn(file_name, col_name, num_hours, split_date):
    #Extra settings
    seed = 42
    np.random.seed(seed)
    plt.style.use('bmh')
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['text.color'] = 'k'

    #Data Input
    df = pd.read_csv(file_name, parse_dates=['timestamp'])
    #print(df.columns)
    df.set_index('timestamp',inplace=True)
    #df.head()
    #df.describe()

    # plot column
    #df.plot(y=col_name,  figsize=(20, 10), grid=True)

    #Decompose data
    #result = dm.decompose(df, col_name, 8760, 0)

    #autoregression 
    #moving average
    #Fit a linear regression model to identify trend
    plt.figure(figsize=(15, 7))
    layout = (3,2)
    pm_ax = plt.subplot2grid(layout, (0,0), colspan=2)
    mv_ax = plt.subplot2grid(layout, (1,0), colspan=2)
    fit_ax = plt.subplot2grid(layout, (2,0), colspan=2)

    #pm_ax.plot(result.trend)
    pm_ax.set_title("Automatic decomposed trend")

    mm = df[col_name].rolling(24).mean()
    #mv_ax.plot(mm)
    mv_ax.set_title("Moving average 24 steps")
    #plt.show()

    X = [i for i in range(0, len(df[col_name]))]
    X = np.reshape(X, (len(X), 1))
    y = df[col_name].values
    model = LinearRegression()
    model.fit(np.isnan(X), np.isnan(y))

    # calculate trend
    trend = model.predict(X)
    #fit_ax.plot(trend)
    fit_ax.set_title("Trend fitted by linear regression")
    #plt.tight_layout()
    #plt.show()

    #check first month of data
    #result = dm.decompose(df, col_name, 744, 0)

    #check last month of data
    #result = dm.decompose(df, col_name, 8760, (8760-744))

    #Looking for weekly seasonality
    resample = df.resample('W')
    weekly_mean = resample.mean()
    plt.figure()
    weekly_mean[col_name].plot(label='Weekly mean')
    plt.title("Resampled series to weekly mean values")
    plt.legend()
    #plt.show()

    #Preparing data for forecasting
    #Split training and test data for verification
    df_training = df.loc[df.index <= split_date]
    df_testcase = df.loc[df.index > split_date]

    #To download datasets for viewing
    #dm.download(df_training, 'training')
    #dm.download(df_testcase, 'test')

    resultsDict={}
    predictionsDict={}

    #ACF/PACF
    #dm.calc_pacf(df.dropna(), col_name)
    #dm.calc_acf(df.dropna(), col_name)

    #dm.training_forecast("SES", col_name, resultsDict, predictionsDict, df, df_training, df_testcase)
    #dm.training_forecast("HWES", col_name, resultsDict, predictionsDict, df, df_training, df_testcase)
    dm.training_forecast("AR", col_name, resultsDict, predictionsDict, df, df_training, df_testcase)
    #dm.training_forecast("ARIMA", col_name, resultsDict, predictionsDict, df, df_training, df_testcase)
    #dm.training_forecast("diff_ARIMA", col_name, resultsDict, predictionsDict, df, df_training, df_testcase)
    #dm.training_forecast("SARIMAX", col_name, resultsDict, predictionsDict, df, df_training, df_testcase)
    #dm.training_forecast("Auto", col_name, resultsDict, predictionsDict, df, df_training, df_testcase)

   


    #print(predictionsDict)
    #print(resultsDict)
    #dm.diff_arima(df, col_name)

    dm.pickles(resultsDict, predictionsDict)

    #best_model = model_metrics(resultsDict)
    #print(best_model)
    dm.forecast('AR', col_name, resultsDict, predictionsDict, df, num_hours)
