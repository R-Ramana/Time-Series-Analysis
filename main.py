import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import io

import statsmodels.tsa.api as smt
import statsmodels as sm
import pmdarima as pm

import warnings
warnings.filterwarnings("ignore") #We will use deprecated models of statmodels which throw a lot of warnings to use more modern ones

import metrics
import plots
import data_models as dm

from metrics import evaluate
from plots import bar_metrics

from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.linear_model import LinearRegression

from pylab import rcParams

#Extra settings
seed = 42
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'

#Data Input
df = pd.read_csv('data/cleaned_data.csv',parse_dates=['Time'])
df.set_index('Time',inplace=True)
#d.head()
#d.describe()
values = df.values
groups = [0, 1, 2, 3]
i = 1

# plot each column
df.plot(y=['Fluct','Ambient_Temp', 'Daily_Avg'], figsize=(20,10), grid=True)


#Decompose data
result = dm.decompose(df, "Ambient_Temp", 8760, 0)

#autoregression 
#moving average
#Fit a linear regression model to identify trend
fig = plt.figure(figsize=(15, 7))
layout = (3,2)
pm_ax = plt.subplot2grid(layout, (0,0), colspan=2)
mv_ax = plt.subplot2grid(layout, (1,0), colspan=2)
fit_ax = plt.subplot2grid(layout, (2,0), colspan=2)

pm_ax.plot(result.trend)
pm_ax.set_title("Automatic decomposed trend")

mm = df.Ambient_Temp.rolling(24).mean()
mv_ax.plot(mm)
mv_ax.set_title("Moving average 24 steps")


X = [i for i in range(0, len(df.Ambient_Temp))]
X = np.reshape(X, (len(X), 1))
y = df.Ambient_Temp.values
model = LinearRegression()
model.fit(X.astype(np.float32), y.astype(np.float32))

# calculate trend
trend = model.predict(X)
fit_ax.plot(trend)
fit_ax.set_title("Trend fitted by linear regression")

plt.tight_layout()
plt.show()

#check first month of data
result = dm.decompose(df, "Ambient_Temp", 744, 0)

#check last month of data
result = dm.decompose(df, "Ambient_Temp", 8760, (8760-744))

#Looking for weekly seasonality
resample = df.resample('W')
weekly_mean = resample.mean()
plt.figure()
weekly_mean.Ambient_Temp.plot(label='Weekly mean')
plt.title("Resampled series to weekly mean values")
plt.legend()
plt.show()

#Preparing data for forecasting
#Split training and test data for verification
split_date ='31/10/2004 23:59'
df_training = df.loc[df.index <= split_date]
df_testcase = df.loc[df.index > split_date]

#To download datasets for viewing
dm.download(df_training, 'training')
dm.download(df_testcase, 'test')

resultsDict={}
predictionsDict={}

dm.forecasting("SES", resultsDict, predictionsDict, df, df_training, df_testcase)
dm.forecasting("HWES", resultsDict, predictionsDict, df, df_training, df_testcase)
dm.forecasting("AR", resultsDict, predictionsDict, df, df_training, df_testcase)
dm.forecasting("ARIMA", resultsDict, predictionsDict, df, df_training, df_testcase)
dm.forecasting("SARIMAX", resultsDict, predictionsDict, df, df_training, df_testcase)

dm.pickles(resultsDict, predictionsDict)

bar_metrics(resultsDict)