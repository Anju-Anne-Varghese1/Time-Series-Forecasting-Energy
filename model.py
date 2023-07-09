import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle


df = pd.read_csv('dataset/IPG2211A2N.csv', parse_dates=True)
print(df)


df.rename(columns={'IPG2211A2N': 'Energy_Production'}, inplace=True)


df['DATE'] = pd.to_datetime(df['DATE'])
print(df)


df.set_index('DATE', inplace=True)
print(df)


df1 = df.copy()
df1['date'] = df1.index
df1['year'] = df1['date'].dt.year
df1['month'] = df1['date'].dt.month


df1.drop(['date'], axis=1, inplace=True)
print(df1)


df_features = df.copy()
df_features['date'] = df_features.index
df_features['year'] = df_features['date'].dt.year
df_features['month'] = df_features['date'].dt.month
df_features['day'] = df_features['date'].dt.day
df_features['day_of_year'] = df_features['date'].dt.dayofyear
df_features['day_of_week'] = df_features['date'].dt.dayofweek
df_features['day_of_week_name'] = df_features['date'].dt.day_name()
df_features['quarter'] = df_features['date'].dt.quarter
df_features['week_of_year'] = df_features['date'].dt.isocalendar().week
df_features['date_offset'] = (df_features['date'].dt.month*100 + df_features['date'].dt.day - 320) % 1300
df_features['season'] = pd.cut(df_features['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Summer', 'Fall', 'Winter'])


df_features.to_csv(f"df_features_target.csv", index=False)


df2 = df1[df1.index > '1960-05-01']


df2['Energy_Production 12 Difference'] = df2['Energy_Production'] - df2['Energy_Production'].shift(12)
print(df2)


def adf_test(value):
    result = adfuller(value)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


adf_test(df2['Energy_Production 12 Difference'].dropna())


acf12 = plot_acf(df2["Energy_Production 12 Difference"].dropna())
plt.show()

pacf12 = plot_pacf(df2["Energy_Production 12 Difference"].dropna())
plt.show()


train_dataset_end = datetime(2010, 12, 1)  # end date for training data 2010-12-01
test_dataset_end = datetime(2023, 5, 1)  # end date for testing data 2023-05-01
train_data = df2[:train_dataset_end]
test_data = df2[train_dataset_end+timedelta(days=1):test_dataset_end]
print(train_data)
print(test_data)


pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]


model_SARIMA = SARIMAX(train_data['Energy_Production'], order=(12, 0, 4), seasonal_order=(0, 1, 0, 12))
model_SARIMA_fit = model_SARIMA.fit()
print(model_SARIMA_fit.summary())

pred_Sarima = model_SARIMA_fit.predict(start=datetime(2011, 1, 1), end=datetime(2023, 5, 1))
print(pred_Sarima)


model_SARIMA_fit.resid.plot()
model_SARIMA_fit.resid.plot(kind='kde')


test_data['Predicted_SARIMA'] = pred_Sarima


print(test_data.isnull().sum())


test_data[['Energy_Production', 'Predicted_SARIMA']].plot()
plt.show()


rmse = np.sqrt(mean_squared_error(y_true=test_data['Energy_Production'], y_pred=test_data['Predicted_SARIMA']))
print(rmse)

print(df2)


file = open('energy.pkl', 'wb')


pickle.dump(model_SARIMA_fit, file)


pickle.dump(df2, open('df.pkl', 'wb'))
