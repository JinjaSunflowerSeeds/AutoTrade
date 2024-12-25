# https://github.com/martinwg/stockprediction/blob/master/wikipedia_download.R


import re
import requests
import pandas as pd
import config as cfg
from eodhd import APIClient
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI


API_KEY= '65c6b1408d13f8.27041860'
api = APIClient(API_KEY)
resp_0 = api.financial_news(s = "AAPL.US", from_date = '2013-01-01', to_date = '2021-01-30', limit = 1000)
resp_1 = api.financial_news(s = "AAPL.US", from_date = '2021-01-30', to_date = '2022-01-30', limit = 1000)


df = pd.DataFrame(resp) # converting the json output into datframe
print(df.tail())
