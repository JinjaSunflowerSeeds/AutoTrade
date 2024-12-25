from sklearn.preprocessing import MinMaxScaler
import os, sys
import time
import torch
import torch.nn as nn
import pandas as pd
import scipy as sp
import numpy as np

from tqdm import tqdm 

import nltk
from nltk.corpus import movie_reviews

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(1989)
"""
this is single variable prediction
It takes the df and uses n-1 for training and 1 for testing/prediction
"""

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out  
    
class Driver():
    def __init__(self):
        self.lookback =15 # choose sequence length
        self.hidden_dim = 128
        self.num_layers = 2
        self.output_dim =  1
        self.num_epochs = 250
        
        self.predicting_col= 'log-ret'
        self.features = [
            'open','close','high',
            'volume',
            'log-ret',
            "macd_crossover",
                "rsi_singal",
                "rsi_singal_v2",
                "mfi_signal",
                "BullishHorn",
]
        
        self.input_dim = len(self.features)
        
        self.df= None
        self.model=None
        self.criterion =None
        self.optimiser=None
        self.scaler=None
        
    def get_data(self):
        self.df = pd.read_csv("/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv").sort_values(by="date")
        self.df = self.df[self.features]
        self.df['label'] = self.df['log-ret']>0.01
        print(self.df.label.value_counts())
        # self.df = self.df[[self.predicting_col]]
        print(self.df.shape)
        
    def scale_data(self):
        self.scaler = MinMaxScaler( )
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])


    def split_data(self,   test_set_size=0.2):
        data_raw = self.df#.to_numpy() # convert to numpy array
        data = []
        
        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - self.lookback): 
            data.append(data_raw[index: index + self.lookback])
        
        data = np.array(data)
        
        # it takes all the data except for last one for training 
        # train size samples, then n-1 sequence and all cols for training 
        # how to make take all the features and predict only one?
        x_train = data[:-1,:-1,:] 
        y_train = data[:-1,-1,:] 
        
        x_test = data[-1:,:-1]
        y_test = data[-1:,-1,:]
        print(data.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        
        self.x_train = x_train#torch.from_numpy(x_train).type(torch.Tensor)
        self.x_test = x_test#torch.from_numpy(x_test).type(torch.Tensor)
        self.y_train = y_train#torch.from_numpy(y_train).type(torch.Tensor)
        self.y_test = y_test#torch.from_numpy(y_test).type(torch.Tensor)     
        print("Done split and torching the data.")
    
    def lstm_model(self):    
        self.model=  LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, num_layers=self.num_layers)
        self.criterion = torch.nn.BCELoss() #torch.nn.CrossEntropyLoss
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01)

     
        
    def grad(self, epoch_display=25):
        hist = np.zeros(self.num_epochs)
        start_time = time.time()
        lstm = []
        for t in range(self.num_epochs):
            y_train_pred = self.model(self.x_train)
            loss = self.criterion(y_train_pred, self.y_train) 
            if t%epoch_display==0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
    
        training_time = time.time()-start_time
        print("Training time: {}".format(training_time))
        return hist
     
    def pred(self):
        y_test_pred = self.model(self.x_test)
        loss = self.criterion(y_test_pred, self.y_test)
        print("loss={}; pred={}; actual={}".format(loss.item(), self.scaler.inverse_transform(np.array([y_test_pred.item()]).reshape(-1, 1)), self.scaler.inverse_transform(np.array([self.y_test.item()]).reshape(-1, 1))))

    def driver(self):
        self.get_data()
        # self.scale_data()
        self.split_data()
        # 
        self.lstm_model()
        self.grad()
        self.pred()
        # 
        self.gru_model()
        self.grad()
        self.pred()






if __name__ == '__main__':
    Driver().driver()











# GRU


# import time
# hist = np.zeros(num_epochs)
# start_time = time.time()
# lstm = []
# for t in range(num_epochs):
#     y_train_pred = model(x_train)
#     loss = criterion(y_train_pred, y_train)
#     if t%10==0:
#         print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()
#     optimiser.zero_grad()
#     loss.backward()
#     optimiser.step()
    
# training_time = time.time()-start_time
# print("Training time: {}".format(training_time))
# y_test_pred = model(x_test)
# loss = criterion(y_test_pred, y_test)
# print(loss.item())
# print("pred={}; actual={}".format(scaler.inverse_transform(np.array([y_test_pred.item()]).reshape(-1, 1)), scaler.inverse_transform(np.array([y_test.item()]).reshape(-1, 1))))
