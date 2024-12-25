import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
def scale_data(train,test):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test, scaler


# load dataset
# dataset = read_csv('pollution.csv', header=0, index_col=0)
dataset = pd.read_csv("/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv").sort_values(by="date")
dataset.fillna(method='bfill', inplace=True)
dataset.drop(columns=['date', 'cpi',  'treasury', 'mortgage', 'score', 'rating', 'unemployment', 'gdp', 'stock splits'], inplace=True)
print(dataset.tail(10))

# integer encode direction
# encoder = LabelEncoder()
# dataset["cbwd"] = encoder.fit_transform(dataset["cbwd"])
# ensure all data is float
dataset = dataset.astype('float32')
print(dataset.head(2))
# 
dataset['label']= dataset.close.shift(-1)




# handle this???
dataset.dropna(inplace=True)
print(dataset.head(2))

values = dataset.values


 
# split into train and test sets
# values = reframed.values
n_train_hours = len(values)-10#365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X,test_X,scaler= scale_data(train_X,test_X)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

 
# design network
model = Sequential()
model.add(LSTM(1500, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=2000, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
print(yhat)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print(inv_y, inv_yhat)
