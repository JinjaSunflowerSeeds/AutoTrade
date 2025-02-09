import sys

import numpy as np
np.random.seed(1)
from keras.metrics import AUC,Precision
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error  ,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from math import sqrt
import datetime as dt
import time
plt.style.use('ggplot')

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import random, sys, random, pickle
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Input, LSTM, Dense, Dropout
from keras.regularizers import l2, l1
from keras.layers import Reshape, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Lambda
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler, TensorBoard
from keras.optimizers import SGD,  RMSprop
from keras.optimizers.legacy import Adam
from keras.layers import concatenate
import keras.backend as K
from tensorflow.python.client import device_lib

sys.path.append('/Users/ragheb/myprojects/stock/src/config')
sys.path.append('/Users/ragheb/myprojects/stock/src/model')
from train_conf import (
    FeatureList,
    TrainConf
)

# import numpy as np
# np.random.seed(1989)  
# import tensorflow as tf
# tf.random.set_seed(1989)  


from keras import backend as K
from keras.callbacks import Callback

from labeling.labeler import Labeler

class GradientClipper(Callback):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        grads = K.get_value(K.gradients(self.model.total_loss, self.model.trainable_weights))
        clipped_grads = [np.clip(g, -self.clip_value, self.clip_value) for g in grads]
        K.set_value(K.gradients(self.model.total_loss, self.model.trainable_weights), clipped_grads)


class LSTM_(FeatureList, Labeler):
    def __init__(self):
        super().__init__()
        FeatureList.__init__(self)
        Labeler.__init__(self)
        self.series = None

        
        self.scaling_featues=self.features#['open','close','high','volume','log-ret']
        print(self.scaling_featues)
        print("*"*120)
        # Setting up an early stop
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,
                                  verbose=1, mode='min')
        self.callbacks_list = [earlystop]
        self.timesteps = 10#100#50
        self.batch_size = 256
        self.num_epochs = 150
        self.drop_rate=0.5
        self.initial_rate=0.005
        self.step_epoch=50
        self.lrList = {}


        self.train= None
        self.val= None
        self.test= None
        self.X_train = []
        self.Y_train = []
        self.X_val = []
        self.Y_val = []
        self.X_test = []
        self.Y_test = []

    def step_decay(self, epoch):
        lrate = self.initial_rate
        if epoch >= self.step_epoch:
            lrate = lrate*self.drop_rate
        elif epoch >=self.step_epoch*2:
            lrate = lrate*self.drop_rate**2
        elif epoch >=self.step_epoch*3:
            lrate = lrate*self.drop_rate**3
        elif epoch >=self.step_epoch*4:
            lrate = lrate*self.drop_rate**4
        if lrate not in self.lrList:
            self.lrList[lrate]=0
        self.lrList[lrate]+=1
        return lrate

    def recall_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self,y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    # Plotting the predictions
    def plot_data(self, Y_test,Y_hat):
        plt.plot(Y_test,c = 'r')
        plt.plot(Y_hat,c = 'y')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.title('Stock Prediction Graph using Multivariate-LSTM model')
        plt.legend(['Actual','Predicted'],loc = 'lower right')
        plt.show()

    # Plotting the training errors
    def plot_error(self, train_loss,val_loss):
        plt.plot(train_loss,c = 'r')
        plt.plot(val_loss,c = 'b')
        plt.ylabel('Loss')
        plt.legend(['train','val'],loc = 'upper right')
        plt.show()

    def set_data(self):
        url =  "/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv"
        df = pd.read_csv(url,parse_dates = True,index_col=0)  
        
        
        df['next_open'] = df.open.shift(-1) 
        
              
        df['date'] = pd.to_datetime(df.date)
        df.sort_values(by='date',inplace=True)
        # Extracting the series
         # Picking the self.series with high correlation
         
         
        df['label'] = 100*(df.high- df.open)/df.open>=2
        # self.df=df
        # self.long_label_profit_stop_loss()
        # df=self.df
        
        
        df['label'] = df['label'].astype(int)
        df[self.features] = df[self.features].astype(float)
        self.series = df[self.features + ['label']]
        # self.series= self.series.astype(float)
        print("label count=", self.series['label'].value_counts())
        # exit(1)
        print( self.series.shape)
        print(self.series.tail())
        
        
        
        
        print("NaN cnt={}".format(self.series[self.series.isnull().any(axis=1)].shape))
        self.series.fillna(method='bfill', inplace=True)
        print(self.series.shape)


        # Train Val Test Split 
        train_end = int(len(self.series)*0.9)  
        self.train = self.series.loc[0:train_end-1]#loc is inclusive 
        self.train.reset_index(inplace=True, drop=True)
        #  
        val_end = len(self.series) - 1
        self.val = self.series.loc[train_end:val_end-1]
        self.val.reset_index(inplace=True, drop=True) 
        # 
        self.test = self.series.loc[val_end:]
        self.test.reset_index(inplace=True, drop=True)
        assert self.test.shape[0] + self.val.shape[0] + self.train.shape[0] == self.series.shape[0]

        print(self.train.shape,self.val.shape,self.test.shape)
        if self.timesteps>self.val.shape[0]:
            self.timesteps = self.val.shape[0]//2
            print("Error: timesteps is too big, setting it to {}".format(self.timesteps))
        assert self.train.columns.tolist()[-1]=='label' and self.val.columns.tolist()[-1]=='label' and self.test.columns.tolist()[-1]=='label' and "last column must be label otherwise target wont be right"

        print("#Normalisation")
        from sklearn.preprocessing import Normalizer

        sc = MinMaxScaler()#Normalizer(norm='l2')#
        self.train.loc[:,self.scaling_featues] = sc.fit_transform(self.train[self.scaling_featues])
        self.val.loc[:,self.scaling_featues] = sc.transform(self.val[self.scaling_featues])
        self.test.loc[:,self.scaling_featues] = sc.transform(self.test[self.scaling_featues])
        print(self.train.shape, self.val.shape, self.test.shape)

        print("sequencing the data...")
        """
        we take a sequence of timestamp data point i-t:i and then las column of i+1 as label
        """ 
        # Loop for training data
        for i in range(self.timesteps,self.train.shape[0]):
            self.X_train.append(self.train[i-self.timesteps:i]) #[i-t:t-1] for training   
            self.Y_train.append(self.train.iloc[i:i+1].values[0][-1] )# t for prediction (last column)
            # self.Y_train.append(self.train.iloc[i][-1])
        self.X_train,self.Y_train = np.array(self.X_train),np.array(self.Y_train) 
        print("train", self.X_train.shape, self.Y_train.shape)
        print(self.Y_train)
        # Loop for val data
        for i in range(self.timesteps, self.val.shape[0]):
            self.X_val.append(self.val[i-self.timesteps:i])
            self.Y_val.append(self.val[i:i+1].values[0][-1] )
        self.X_val,self.Y_val = np.array(self.X_val), np.array(self.Y_val)
        print("val", self.X_val.shape,  self.Y_val.shape)
        print(self.Y_val)
        # Loop for testing data
        assert self.test.shape[0] ==1 and "test is useless for now, as it predicts tomottow that we don have yet"
        for i in range(0, self.test.shape[0]):
            self.X_test.append(self.test[i-self.timesteps:i])
            self.Y_test.append(self.test[i:i+1].values[0][-1] )
        self.X_test,self.Y_test = np.array(self.X_test),np.array(self.Y_test)

        print("test", self.X_test.shape, self.Y_test.shape)
        

    # Evaluating the model
    def evaluate_model(self, model):
        t=0
        for df_x, df_y  in [(self.X_train, self.Y_train),
                            (self.X_val, self.Y_val),
                            (self.X_test, self.Y_test)]:
            try:
                Y_prob = model.predict(df_x, verbose=0)
                # Y_prob =[ Y_prob[i][0] for i in range(len(Y_prob))]
                Y_prob = Y_prob.flatten()
                if t>0:
                    print([(df_y[i], Y_prob[i]) for i in range(len(Y_prob)) if df_y[i]==1 or Y_prob[i]>=0.5])
                t+=1
                Y_hat= [int(i>=0.5) for i in Y_prob]
                
                print(np.sum(Y_hat),len(df_y))
                # Y_prob = model.predict_proba(df_x, verbose=1)[:,1]
                # print(Y_prob[:10])

                acc= accuracy_score(df_y, Y_hat)
                f1= f1_score(df_y, Y_hat)
                precision=precision_score(df_y,Y_hat)
                auc=roc_auc_score(df_y,Y_prob),

                mse = mean_squared_error(df_y,Y_hat)
                rmse = sqrt(mse)
                r2 = r2_score(df_y,Y_hat)

                print('MSE = {}'.format(mse))
                print('RMSE = {}'.format(rmse))
                print('R-Squared Score = {}'.format(r2))
                print("precision={:.2f}, auc={:.2f} accuracy={:.2f}, f1={:.2f}, ".format(
                    precision, auc[0], acc, f1))
                print("="*100)
            except Exception as e:
                print(e)
        # self.plot_data(true,predicted)
        # print(predicted>0.5)


    def model_arch(self):
        K.clear_session()
        main_input = Input(shape=(self.X_train.shape[1],self.X_train.shape[2]),name='main_input')
        # Adding Layers to the model
        
        m=LSTM(units=50,return_sequences=True)(main_input)
        m=Dropout(0.2)(m) 
        m=LSTM(units=50,return_sequences=True,kernel_regularizer=l2(0.001))(m)#
        m= Dropout(0.2)(m)
        m=LSTM(units=50 )(m)#
        m= Dropout(0.2)(m)
 
 
        # m=LSTM(15,return_sequences=False,kernel_regularizer=l2(0.01))(main_input)
        m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
        m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
        m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
        m=Dropout(0.3)(m)

        m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
        m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
        m=Dropout(0.3)(m)
    
    # a listing that everyone would nuf
    
        # m=LSTM(10,return_sequences=True, kernel_regularizer=l2(0.01))(main_input)
        # m=LSTM(50,return_sequences=True, kernel_regularizer=l2(0.01))(m)#
        # m= Dropout(0.2)(m)
        # m=LSTM(50,return_sequences=True, kernel_regularizer=l2(0.01))(m)#
        # m= Dropout(0.2)(m) 
        # m= LSTM(5, return_sequences=False)(m)
        # m= Dropout(0.2)(m) 
        main_cross=Dense(1,activation='sigmoid',name='main_cross')(m)

        model=Model(inputs=main_input,outputs=[main_cross])
        return model

    #Build and train the model
    def fit_model(self):     
        
        optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)  # Norm of gradients clipped to 1.0
        lr = LearningRateScheduler(self.step_decay)
        self.callbacks_list = [lr]
        
        model = self.model_arch()
        model.compile(optimizer='adam',loss= 'binary_crossentropy' ,
                      metrics=[
                        #   'accuracy',
                      AUC(name='auc'),
                      Precision(name='precision')
                      ]
                      )
        print("training...")
        history=model.fit(self.X_train,self.Y_train,
                          shuffle=False,
                          callbacks=self.callbacks_list,
                          validation_data=(self.X_val,[self.Y_val]),
                          epochs=self.num_epochs,
                          batch_size=self.batch_size,verbose=2
                          )
        print("self.lrList={}".format(self.lrList))
        # model.reset_states()
        return model, history.history['loss'], history.history['val_loss']

    def driver(self):
        model,train_error,val_error = self.fit_model()
        print(set(self.lrList))
        self.evaluate_model(model)
        self.plot_error(train_error,val_error)


if  __name__ == "__main__":
    lstm = LSTM_()
    lstm.set_data()
    lstm.driver()
