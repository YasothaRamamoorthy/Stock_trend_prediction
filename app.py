import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas_datareader as data

from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
from keras.models import load_model
import streamlit as st


# Define a start date and End Date
start = dt.datetime(2021,1,1)
end =  dt.datetime(2023,12,31)

st.title('STOCK TREND PREDICTION')
user_input = st.text_input('Enter stock ticker','SBIN.NS')
# Read Stock Price Data 
data = yf.download(user_input, start , end)

#describe

st.subheader('Data from 2021 to 2023')
st.write(data.describe())

# Plotting the Closing Prices of the Stocks over the given period
st.subheader('Closing price Vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing price Vs Time chart for the last 100 days')
fig=plt.figure(figsize=(12,6))
moving_100=data.Close.rolling(100).mean()
plt.plot(data.Close)
plt.plot(moving_100)
st.pyplot(fig)

st.subheader('Closing price Vs Time chart for the last 100 days and 200days')
fig=plt.figure(figsize=(12,6))
moving_100=data.Close.rolling(100).mean()
moving_200=data.Close.rolling(200).mean()
plt.plot(data.Close)
plt.plot(moving_100,'r',label='mean_of_last_100days')
plt.plot(moving_200,'g',label='mean_of_last_200days')
st.pyplot(fig)


data_training= pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler(feature_range=(0,1))

data_training_array= scalar.fit_transform(data_training)


#load mode
model=load_model('/workspaces/Stock_trend_prediction/keras_model.h5')

#testingst

past_100_days=data_training.tail(100)

final_data = pd.concat([past_100_days, data_testing], ignore_index=True)


input_data = scalar.fit_transform(final_data)
x_test=[]
y_test=[]

for i in range(100,data_training_array.shape[0]):
    x_test.append(data_training_array[i-100:i])
    y_test.append(data_training_array[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_pred = model.predict(x_test)

scaler_=scalar.scale_

scale_factor= 1/scaler_[0]
y_pred= scale_factor*y_pred
y_test=y_test*scale_factor

#final
st.subheader('Prediction Vs original')
fig2=plt.figure(figsize=(12,6))

plt.plot(y_test,'b',label='Original price')
plt.plot(y_pred,'r',label='predicted price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Prediction vs Actual Price')
plt.legend()

st.pyplot(fig2)
