import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas_datareader as data

from pandas_datareader import data as pdr
import yfinance as yf
# yf.pdr_override()
# y_symbols = ['SCHAND.NS', 'TATAPOWER.NS', 'ITC.NS']
# from datetime import datetime
# startdate = datetime(2022,12,1)
# enddate = datetime(2022,12,15)
# data = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)

# start ='2010-01-01'
# end='2023-12-31'
# import yfinance 
# yfinance.download('AMZN', start, end)


import datetime as dt
import yfinance as yf

company = 'AAPL'

# Define a start date and End Date
start = dt.datetime(2021,1,1)
end =  dt.datetime(2023,12,31)

# Read Stock Price Data 
data = yf.download(company, start , end)

#print(data.tail(10))

data=data.reset_index()
#print(data.head())

data = data.drop(['Date','Adj Close'],axis=1)
#print(data.head())

plt.plot(data.Close)
plt.title("Stock Closing Prices")

moving_100=data.Close.rolling(100).mean()
#print(moving_100)

plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(moving_100,'r')
plt.show

moving_200 =data['Close'].rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(moving_100,'r')
plt.plot(moving_200,'g')
plt.show

#print(data.shape)
#splitting data into training and traning

data_training= pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

# print("training:",data_training.shape)
# print("testing:",data_testing.shape)

# size : training: (527, 1) testing: (226, 1)

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler(feature_range=(0,1))

data_training_array= scalar.fit_transform(data_training)

#print(data_training_array)

x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

# print(x_train)
x_train,y_train=np.array(x_train),np.array(y_train)

#Creating ml model

from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
model=Sequential()
#Adding the first LSTM layer and some Dropout regularisation .
#The unites are not normalised so we will use the MinMaxScaler to scale
model.add(LSTM(
    units=50,
    return_sequences=True,
    input_shape=(x_train.shape[1],1),
    activation='relu'
))
model.add(Dropout(0.2))

#second layer
model.add(LSTM(
    units=60,
    return_sequences=True,
    activation='relu'
))
model.add(Dropout(0.3))

#third layer
model.add(LSTM(
    units=80,
    return_sequences=True,
    activation='relu'
))
model.add(Dropout(0.4))

#fourth layer
model.add(LSTM(
    units=120,
    activation='relu'
))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# print(model.summary())
# Model: "sequential"
# #_________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 100, 50)           10400
                                                

#  dropout (Dropout)           (None, 100, 50)           0
                                                

#  lstm_1 (LSTM)               (None, 100, 60)           26640
#        45120

#  dropout_2 (Dropout)         (None, 100, 80)           0

#  lstm_3 (LSTM)               (None, 120)               96480

#  dropout_3 (Dropout)         (None, 120)               0

#  dense (Dense)               (None, 1)                 121

# =================================================================
# Total params: 178761 (698.29 KB)
# Trainable params: 178761 (698.29 KB)
# Non-trainable params: 0 (0.00 Byte)

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)

model.save('keras_model.h5')

# print(data_testing.head())#index start = 527
# print(data_training.tail())#index end = 526

past_100_days=data_training.tail(100)

final_data = pd.concat([past_100_days, data_testing], ignore_index=True)


input_data = scalar.fit_transform(final_data)

x_test=[]
y_test=[]

for i in range(100,data_training_array.shape[0]):
    x_test.append(data_training_array[i-100:i])
    y_test.append(data_training_array[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#making prediction

y_pred = model.predict(x_test)

# print(scalar.scale_) 0.01368176

scale_factor= 1/0.01368176
y_pred= scale_factor*y_pred
y_test=y_test*scale_factor

plt.figure(figsize=(12,6))

plt.plot(y_test,'b',label='Original price')
plt.plot(y_pred,'r',label='predicted price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Prediction vs Actual Price')
plt.legend()
plt.show()