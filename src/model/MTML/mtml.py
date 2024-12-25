# https://colab.research.google.com/drive/1ZMZ16BG9_RK4_hdNiIqtV8eI9TNqDPYK#scrollTo=B_h-23BEr1He
import os, sys

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

def load_movie_data():
    """
      Loads labeled movie reviews from NLTK. 
      Returns a pandas dataframe with `text` column and `is_positive` for labels.
    """
    nltk.download('movie_reviews', quiet = True)
    df= pd.read_csv("/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv")
    df['is_positive']= 100.0*(df.close - df.open)/df.open
    df["is_positive"] = df["is_positive"]#.shift(-1)
    df["is_positive"] = df.is_positive > 0.001
    print(df.is_positive.value_counts())
    return df.head(len(df)-1)
    
    negative_review_ids = movie_reviews.fileids(categories=["neg"])
    positive_review_ids = movie_reviews.fileids(categories=["pos"])
        
    neg_reviews = [movie_reviews.raw(movie_id) for movie_id in movie_reviews.fileids(categories = ["neg"])]
    pos_reviews = [movie_reviews.raw(movie_id) for movie_id in movie_reviews.fileids(categories = ["pos"])]
    labels = [0 for _ in range(len(neg_reviews))] + [1 for _ in range(len(pos_reviews))]

    movie_df = pd.DataFrame(data = {'text' : neg_reviews + pos_reviews, 'is_positive' : labels})
    return movie_df

def load_yelp_data():
    """
      Loads in a sample of the larger Yelp dataset of annotated restaurant reviews. Reviews are from 1 to 5 stars.
      Removes 2 and 4 star reviews, leaving negative (1 star), neutral (3 stars) and positive (5 stars) reviews
      Returns a pandas dataframe with `text` column and `is_negative`, `is_neutral`, `is_positive` columns
    """
    full_df = pd.read_csv('yelp.csv')
    df = full_df[full_df['stars'].isin([1, 3, 5])].copy().reset_index(drop = True)
    df['is_positive'] = (df['stars'] == 5).astype(int)
    df['is_neutral'] = (df['stars'] == 3).astype(int)
    df['is_negative'] = (df['stars'] == 1).astype(int)
    return df

def prep_data(downsample_yelp = True):
    """
      Loads both datasets and creates a bag-of-words representation for both. Because Yelp is larger than
      the movie dataset, downsample to be equivalent size.
      Returns a dictionary which contain the input and output variables for either set.
    """
    movie_df = load_movie_data()
    # print(movie_df)
    # print("*"*100)
    yelp_df = load_yelp_data()
    
     
    data = {'movie' : 
               {'X' : movie_df.loc[:, ~movie_df.columns.isin(['is_positive', 'date', 'Unnamed: 0', 'treasury','mortgage', 'score', 'rating', 'unemployment'])], #cv.transform(movie_df['text']).astype(np.int8),
               'y' : movie_df['is_positive'].astype(np.int8).values.reshape(-1, 1)},
        #    'yelp' : 
        #        {'X' : cv.transform(yelp_df['text']).astype(np.int8),
        #        'y' : yelp_df[['is_positive', 'is_neutral', 'is_negative']].astype(np.int8).values}
           }
    return data
class Task_Dataset(Dataset):
    def __init__(self, X : sp.sparse.csr.csr_matrix, 
                       y : np.ndarray):
        self.X = X
        self.y = torch.from_numpy(y).float()    # convert to torch.Tensor here
        assert self.X.shape[0] == self.y.shape[0]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):    
        # convert X to torch.Tensor at run-time, keep it sparse to converse memory otherwise
        X = torch.tensor(self.X.iloc[idx].values).float() 
        # X = torch.from_numpy(self.X[idx].astype(np.int8).todense()).float().squeeze()
        y = self.y[idx]
        return X, y
    
class SingleTask_Network(nn.Module):
    """
      torch NN with single-hidden layer and sigmoid non-linearity
      not too interesting but useful to see minimal changes needed to make
      it multi-task
    """
    def __init__(self, input_dim : int, 
                 output_dim : int = 1, 
                 hidden_dim : int = 100):
        super(SingleTask_Network, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # define the hidden layer and final layers
        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.final = nn.Linear(self.hidden_dim, self.output_dim)
        

    def forward(self, x : torch.Tensor):
        x = self.hidden(x)  # apply hidden layer
        x = torch.sigmoid(x)  # apply sigmoid non-linearity
        x = self.final(x) # apply final layer
        return x    # note: no sigmoid/softmax here; that is taken care of by loss functions
    
data = prep_data()

movie_X_train, movie_X_test, movie_y_train, movie_y_test =  train_test_split(data['movie']['X'], data['movie']['y'], 
                                                                             random_state = 1225)
 
# yelp_X_train, yelp_X_test, yelp_y_train, yelp_y_test =  train_test_split(data['yelp']['X'], data['yelp']['y'], 
#                                                                              random_state = 1225)


# create Dataset and Dataloaders for each. Batch size 64

movie_ds = Task_Dataset(movie_X_train, movie_y_train)
movie_dl = DataLoader(movie_ds, batch_size = 64, shuffle = True)

# yelp_ds = Task_Dataset(yelp_X_train, yelp_y_train)
# yelp_dl = DataLoader(yelp_ds, batch_size = 64, shuffle = True)

def make_preds_single(model, X):
    # helper function to make predictions for a model
    # with torch.no_grad():
    #     # y_hat = model(torch.from_numpy(X.astype(np.int8).todense()).float())
    #     y_hat=model(torch.tensor(X.values).float() )
    y_hat=model(torch.tensor(X.values).float() )
    print(y_hat)
    exit(1)
    return y_hat
    
model = SingleTask_Network(movie_ds.X.shape[1], movie_ds.y.shape[1])
# define optimizer. Specify the parameters of the model to be trainable. Learning rate of .001
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = nn.BCEWithLogitsLoss()

# some extra variables to track performance during training 
f1_history = []
trainstep = 0
for i in tqdm(range(6)):
    for j, (batch_X, batch_y) in enumerate(movie_dl):
        preds = model(batch_X)
        loss = loss_fn(preds, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_hat = torch.sigmoid(make_preds_single(model, movie_X_test)) >= .5
        f1_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                          'task' : 'movie', 'f1' : f1_score(movie_y_test, y_hat)})
        trainstep += 1
        
# overall F1 of fit model. ~.82 F1 score. Not too bad
y_hat = torch.sigmoid(make_preds_single(model, movie_X_test)) 
print(movie_y_test)
print( y_hat  )
print(classification_report(movie_y_test, y_hat, 
                            target_names=['negative', 'positive']))
