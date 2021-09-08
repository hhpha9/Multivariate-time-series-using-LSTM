# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 20:35:36 2021

@author: TechFast Australia
"""
# Original example from
# https://youtu.be/tepxdcepTbY
"""
NOTE I HAVE MODIFIED THIS CODE SO PLEASE USE THE ATTACHED CSV FILE IN THE EMAIL
Original data from:
@author: Sreenivas Bhattiprolu
Code tested on Tensorflow: 2.2.0
    Keras: 2.4.3
dataset: https://finance.yahoo.com/quote/GE/history/
Also try S&P: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
#from datetime import datetime

#Read the csv file
df = pd.read_csv('CSV Creep Spray.csv')
print(df.head()) #7 columns, including the Date. 

#Separate time for future plotting
train_time = df['Time(hours)']
print(train_time.tail(15)) #Check last few hours. 

#Variables for training
cols = list(df)[1:5]
#Date column is not used in training. 
print(cols) #['Strain', 'Stress', 'Temperature', 'Humidity']

#New dataframe with only training data - 4 columns
df_for_training = df[cols].astype(float)


#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 4. We will make timesteps = 14 (past hours data used for training). 

#Empty lists to be populated using formatted training data
trainX = [] #training series
trainY = [] #prediction

n_future = 1   # Number of hours we want to look into the future based on the past days.
n_past= 14  # Number of hours we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (13247, 5)
#13247 refers to the number of data points and 4 refers to the columns (multi-variables).
#len() function returns the number of items in an object. When the object is a string, 
# returns the number of characters in the string

for i in range(n_past, len(df_for_training_scaled) - n_future +1): 
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

#In my case, trainX has a shape (13233, 14, 5). 
#13233 because we are looking back 14 steps of hours (13247 - 14 = 13233). 
#Remember that we cannot look back 14 hours until we get to the 15th hour. 
#Also, trainY has a shape (13233, 1). Our model only predicts a single value, but 
#it needs multiple variables (4 in my example) to make this prediction. 
#This is why we can only predict a single day after our training, the day after where our data ends.
#To predict more hours in future, we need all the 4 variables which we do not have. 
#We need to predict all variables if we want to do that. 

# define the Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


#Remember that we can only predict one hour in future as our model needs 5 variables
#as inputs for prediction. We only have all 5 variables until the last hour in our dataset.
n_past = 16
n_hours_for_prediction=15  #let us predict past 15 hours

predict_period_hours = range(list(train_time)[-n_past],periods=n_hours_for_prediction)
print(predict_period_hours)

# Next 2 commented lines are from original code, please ignore
# predict_period_hours = pd.date_range(list(train_time)[-n_past], periods=n_days_for_prediction).tolist()
# print(predict_period_hours)

#Make prediction
prediction = model.predict(trainX[-n_hours_for_prediction:]) #shape = (n, 1) where n is the n_hours_for_prediction

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]


# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_hours:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Time(hours)':np.array(forecast_dates), 'strain (ustrain)':y_pred_future})
df_forecast['Time(hours)']=pd.to_datetime(df_forecast['Time(hours)'])


original = df[['Time(hours)', 'strain (ustrain)']]
original['Time(hours)']=pd.to_datetime(original['Time(hours)'])
original = original.loc[original['Time(hours)'] >= '2020-5-1']

sns.lineplot(original['Time(hours)'], original['strain (ustrain)'])
sns.lineplot(df_forecast['Time(hours)'], df_forecast['strain (ustrain)'])