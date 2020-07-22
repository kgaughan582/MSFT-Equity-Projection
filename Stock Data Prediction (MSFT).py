#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas_datareader as pdr
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[126]:


#Extract Data From API
key = 'f9b723b76349dcbdfd4ecb614b82f4904b7f9c1e'
df = pdr.get_data_tiingo('MSFT', api_key = key)
df.to_csv('MSFT Historical Prices')
df = pd.read_csv('MSFT Historical Prices')
df = pd.DataFrame(df)
df


# In[127]:


#Data Clean-up/ Preprocessing 
df.columns = map(lambda x: str(x).capitalize(), df.columns)
df.rename(columns = {'Close price': 'Close'}, inplace = True)
df.shape # 1257 trading days in dataset 
df['Date'] = pd.to_datetime(df['Date']).dt.date
df.dropna() #drop all n/a values 
null_check = df.isnull()
#Checking for Missing Values
for column in null_check.columns.values.tolist():
    print (null_check[column].value_counts())   


# In[128]:


#Visualizing Closing Price History 
df_plot_close = df[['Close', 'Date']]
df_plot_close.plot(kind = 'line', x = 'Date', y = 'Close')
plt.gcf().set_size_inches(12, 8)
plt.title('Historical MSFT Close Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')


# In[129]:


#Splitting Train vs Test Data
df_close = df[['Close', 'Date']]
df_close.set_index('Date', inplace = True)
train_ratio = 0.80
rolling = 60
train_data = np.array(df_close[:int(len(df_close)*train_ratio)])
test_data = df_close[int(len(train_data))-60:]


# In[130]:


train_data_df = df_close[:int(len(df_close)*train_ratio)]


# In[131]:


#Normalization - MinMaxScaler
mms = MinMaxScaler(feature_range = (0,1))
scaled_close = mms.fit_transform(train_data)
df_close.shape, scaled_close.shape, test_data.shape


# In[146]:


#Seperating x/y Training Sets
x_train = []
y_train = []
for i in range(rolling, len(scaled_close)):
    x_train.append(scaled_close[i-rolling:i,0])
    y_train.append(scaled_close[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train_shape = x_train.shape[0]
x_train = np.reshape(x_train, (x_train_shape, rolling, 1))
y_train = np.reshape(y_train, (-1,1))


# In[148]:


#Seperating x/y Testing Sets
x_train = []
y_train = df_close[int(len(train_data)):]
for i in range(rolling, len(test_data)):
    x_train.append(test_data[i-rolling:i,0])
x_test = np.array(x_test)
x_test_shape = x_test.shape[0]
x_test = np.reshape(x_test, (x_test_shape, rolling, 1))


# In[24]:


#LSTM Model
lstm = Sequential()
lstm.add(LSTM(rolling, return_sequences = True, input_shape = (rolling, 1)))
lstm.add(LSTM(rolling, return_sequences = False))
lstm.add(Dense(rolling/2))
lstm.add(Dense(1))
lstm.compile(optimizer = 'adam', loss = 'mse', metrics = ['mean_squared_error'])


# In[27]:


#LSTM Model Fitting
lstm.fit(x_train, y_train, verbose = 1, batch_size = 1, epochs = 1)


# In[66]:


#Train/ Test Data Predictions
x_train_p = lstm.predict(x_train)
x_train_p = mms.inverse_transform(x_train_p)
x_test_p = lstm.predict(x_test)
x_test_p = mms.inverse_transform(x_test_p)


# In[79]:


training_train = df_close[rolling:int(len(train_data))]
training_train['Predicted Close'] = x_train_p
plt.figure(figsize = (16,8))
plt.title('LSTM Model (MSFT Prices)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.plot(training['Close'])
plt.plot(training_train[['Predicted Close']])
plt.legend(['Training Data', 'Validation'])
plt.show()


# In[80]:


training_train = df_close[:int(len(train_data))]
validation = df_close[int(len(train_data)):]
x = pd.DataFrame(x_test_p, columns= df_close['Date'][int(len(train_data))-rolling:])
plt.figure(figsize = (16,8))
plt.title('LSTM Model (MSFT Prices)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.plot(training['Close'])
plt.plot(validation[['Close']])
plt.legend(['Training Data', 'Validation'])
plt.show()


# In[58]:


training = df_close[:int(len(train_data))]
validation = df_close[int(len(train_data)):]
validation['Predicted Close'] = np.concatenate((,x_test)
plt.figure(figsize = (16,8))
plt.title('LSTM Model (MSFT Prices)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.plot(training['Close'])
plt.plot(validation[['Close']])
plt.legend(['Training Data', 'Validation'])
plt.show()


# In[48]:





# In[ ]:


#LSTM Model Valuation (RMSE, MSE, R^2)
MSE = np.mean(x_test_p - y_test)**2
RMSE = np.sqrt(MSE)
print ('Mean Squared Error:', MSE, '\nRoot Mean Squared Error:', RMSE)

