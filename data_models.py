import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
import metrics

import pickle

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# progress bar
from tqdm import tqdm, tqdm_notebook

def decompose(df, column_name, end_value, start_value):
    rcParams['figure.figsize'] = 18, 8
    plt.figure(num=None, figsize=(50, 20), dpi=80, facecolor='w', edgecolor='k')
    series = df[column_name][start_value:end_value]
    result = seasonal_decompose(series.values, model='additive', period = 24)
    result.plot()
    plt.show()
    return result

def download(df, name):
    filepath = 'data\\results\\' + name + '.csv'
    df.to_csv(filepath)

def forecasting(model_name, resultsDict, predictionsDict, df, df_training, df_testcase):
    #index = len(df_training)
    yhat = list()

    for t in tqdm(range(len(df_testcase.Ambient_Temp))):
        temp_train = df[:len(df_training)+t]
        
        if model_name == "SES":
            model = SimpleExpSmoothing(temp_train.Ambient_Temp)
        elif model_name == "HWES":
            model = ExponentialSmoothing(temp_train.Ambient_Temp)
        elif model_name == "AR":
            model = AR(temp_train.Ambient_Temp)
        # elif model_name == "MA":
        #     model = ARMA(temp_train.Ambient_Temp, order=(0, 1))
        # elif model_name == "ARMA":
        #     model = ARMA(temp_train.Ambient_Temp, order=(1, 1))
        elif model_name == "ARIMA":
            model = ARIMA(temp_train.Ambient_Temp, order=(1,0, 0))
        elif model_name == "SARIMAX":
            model = SARIMAX(temp_train.Ambient_Temp, order=(1, 0, 0), seasonal_order=(0, 0, 0, 3))

        model_fit = model.fit()

        if model_name == "SES" or "HWES":
            predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        elif model_name == "AR" or "ARIMA" or "SARIMAX":
            predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict[model_name] = metrics.evaluate(df_testcase.Ambient_Temp, yhat.values)
    predictionsDict[model_name] = yhat.values
    plt.plot(df_testcase.Ambient_Temp.values , label='Original')
    plt.plot(yhat.values,color='red',label= model_name + ' predicted')
    plt.legend()
    plt.show()

def pickles(resultsDict, predictionsDict):
    with open('scores.pickle', 'wb') as handle:
        pickle.dump(resultsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('predictions.pickle', 'wb') as handle:
        pickle.dump(predictionsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('scores.pickle', 'rb') as handle:
        resultsDict = pickle.load(handle)

    ## Load our results from the model notebook
    with open('predictions.pickle', 'rb') as handle:
        predictionsDict = pickle.load(handle)