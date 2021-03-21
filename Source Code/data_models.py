import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
import metrics

import pickle

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pmdarima.arima import auto_arima


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

def calc_pacf(df, column_name):
    plot_pacf(df[column_name], lags = 40)
    plt.show()
    #print(pacf(df[column_name], nlags = 24))

def calc_acf(df, column_name):
    plot_acf(df[column_name], lags = 40)
    plt.show()
    #print(acf(df[column_name], nlags = 24))


def download(df, name):
    filepath = 'data\\training\\' + name + '.csv'
    df.to_csv(filepath)

def training_forecast(model_name, column_name, resultsDict, predictionsDict, df, df_training, df_testcase):
    yhat = list()

    for t in tqdm(range(len(df_testcase[column_name]))):
        temp_train = df[:len(df_training)+t]
        
        if model_name == "SES":
            model = SimpleExpSmoothing(temp_train[column_name])
        elif model_name == "HWES":
            model = ExponentialSmoothing(temp_train[column_name])
        elif model_name == "AR":
            model = AR(temp_train[column_name])
        # elif model_name == "MA":
        #     model = ARMA(temp_train[column_name], order=(0, 1))
        # elif model_name == "ARMA":
        #     model = ARMA(temp_train[column_name], order=(1, 1))
        elif model_name == "ARIMA":
            model = ARIMA(temp_train[column_name], order=(2,0, 1))
        # elif model_name == "diff_ARIMA":
            
        #     #update order
        #     model = ARIMA(differenced, order=(1, 0, 0))
        elif model_name == "SARIMAX":
            model = SARIMAX(temp_train[column_name], order=(
                1, 0, 0), seasonal_order=(0, 0, 0, 3))
        
        # elif model_name == "Auto":
        #     model = auto_arima(temp_train[column_name], start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0,
        #                        max_P=5, max_D=5, max_Q=5, m=12, seasonal=True, error_action='warn', trace=True, surpress_warnings=True, stepwise=True, random_state=20, n_fits=50)

        model_fit = model.fit()

        if model_name == "SES" or "HWES":
            predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        elif model_name == "AR" or "ARIMA" or "SARIMAX" or "Auto":
            predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict[model_name] = metrics.evaluate(
        df_testcase[column_name], yhat.values)
    predictionsDict[model_name] = yhat.values
    plt.plot(df_testcase[column_name].values , label='Original')
    plt.plot(yhat.values,color='red',label= model_name + ' predicted')
    plt.legend()
    #plt.show()
    
    pd.DataFrame(yhat, columns=[model_name]).to_csv('data/models/' + model_name + '.csv')


# def diff_arima(df, column_name):
#     hours = 24
#     differenced = df[column_name].diff()
#     plot_acf(differenced, lags=40)
#     plt.show()
#     plot_pacf(differenced, lags=40)
#     plt.show()
#     # differenced = difference(df, column_name, 1)
#     print(differenced)
    #calc_acf(differenced, column_name)
    #calc_pacf(differenced, column_name)
    # invert differenced value
    # def inverse_difference(history, yhat, interval=1):
    #     return yhat + history[-interval]

    
    
    # multi-step out-of-sample forecast
    # forecast = model_fit.forecast(steps=24)
    # invert the differenced forecast to something usable
    # history = [x for x in X]
    # hour = 1
    # for yhat in forecast:
    #     inverted = inverse_difference(history, yhat, hours_in_a_day)
    #     print('Hour %d: %f' % (hour, inverted))
    #     history.append(inverted)
    #     hour += 1

# create a differenced series
# interval = difference = 1
# def difference(df, col_name, interval):
#     for i in range(interval, len(df[col_name])):
#         difference = df[i] - df[i - interval]
#     return difference

# Walk throught the test data, training and predicting 1 day ahead for all the test data
# index = len(df_training)
# yhat = list()
# for t in tqdm(range(len(df_test.pollution_today))):
#     temp_train = air_pollution[:len(df_training)+t]
#     model = SARIMAX(temp_train.pollution_today, order=(
#         1, 0, 0), seasonal_order=(0, 0, 0, 3))
#     model_fit = model.fit(disp=False)
#     predictions = model_fit.predict(
#         start=len(temp_train), end=len(temp_train), dynamic=False)
#     yhat = yhat + [predictions]

# yhat = pd.concat(yhat)
# resultsDict['SARIMAX'] = evaluate(df_test.pollution_today, yhat.values)
# predictionsDict['SARIMAX'] = yhat.values
# plt.plot(df_test.pollution_today.values, label='Original')
# plt.plot(yhat.values, color='red', label='SARIMAX')
# plt.legend()


def forecast(model_name, column_name, resultsDict, predictionsDict, df, num_hours):
    yhat = list()
    df.fillna(value=0, inplace=True)

    for t in tqdm(range(len(df[column_name]))):
        temp_train = df[:len(df)+t]
        
        if model_name == "SES":
            model = SimpleExpSmoothing(temp_train[column_name])
        elif model_name == "HWES":
            model = ExponentialSmoothing(temp_train[column_name])
        elif model_name == "AR":
            model = AR(temp_train[column_name])
        # elif model_name == "MA":
        #     model = ARMA(temp_train[column_name], order=(0, 1))
        # elif model_name == "ARMA":
        #     model = ARMA(temp_train[column_name], order=(1, 1))
        elif model_name == "ARIMA":
            model = ARIMA(temp_train[column_name], order=(1,1, 0))
        elif model_name == "SARIMAX":
            model = SARIMAX(temp_train[column_name], order=(
                1, 0, 0), seasonal_order=(0, 0, 0, 3))
        # elif model_name == "Auto":
        #     model = auto_arima(temp_train[column_name], start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0,
        #                        max_P=5, max_D=5, max_Q=5, m=12, seasonal=True, error_action='warn', trace=True, surpress_warnings=True, stepwise=True, random_state=20, n_fits=50)
        
        model_fit = model.fit()
        start_index = len(temp_train)
        end_index = start_index + num_hours

        if model_name == "SES" or "HWES":
            predictions = model_fit.predict(start=start_index, end=end_index)
        elif model_name == "AR" or "ARIMA" or "SARIMAX" or "Auto":
            predictions = model_fit.predict(start=start_index, end=end_index, dynamic=False)
        #yhat = yhat + [predictions]
    
    yhat = yhat + [predictions]
    yhat = pd.concat(yhat)
    pd.DataFrame(yhat, columns=[model_name]).to_csv('data/' + model_name + '_' + str(num_hours) + '_' + column_name + '_forecast.csv')
    predicted_values = pd.DataFrame(yhat, columns=[column_name])
    
    #append data to predictions list
    # new_data = df
    # new_data.dropna(inplace=True)
    # new_data.append(predicted_values, ignore_index=False).to_csv('data/' + model_name + '_append_test.csv')

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
