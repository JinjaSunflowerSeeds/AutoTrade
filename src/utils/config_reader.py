import json


def get_ticker_data_conf(conf_file="./config/data_conf.json"):
    with open(conf_file) as f:
        x=json.load(f)
        data = x['ticker']
        interval=x['interval']
    stock= data['stock']
    indices= data['indices']
    correlated_stocks= data['correlated_stocks']
    data_output_base_dir=data['data_output_base_dir']#.format(stock)
    _lookback= int(interval["lookback"])
    interval=interval["interval"]
    return stock, interval, _lookback, indices, correlated_stocks, data_output_base_dir

def get_economy_data_conf(conf_file="./config/data_conf.json"):
    with open(conf_file) as f:
        x=json.load(f)
        data = x['economy']
        interval=x['interval']
    return data['fred_api_key'], data['data_output_base_dir'], int(interval["lookback"])

def get_fundamentals_data_conf(conf_file="./config/data_conf.json"):
    with open(conf_file) as f:
        x=json.load(f)
        data = x['fundamentals']
        interval=x["interval"]
        tickers = x['ticker']
    
    stock= tickers['stock']
    company= tickers['company_name']
    # indices= tickers['indices']
    # correlated_stocks= tickers['correlated_stocks']
    data_output_base_dir=data['data_output_base_dir']#.format(stock)
    _lookback= int(interval["lookback"])
    # interval=interval["interval"]
    return stock, company, data_output_base_dir, _lookback

def get_merger_output_file(conf_file="./config/data_conf.json"):
    with open(conf_file) as f:
        data = json.load(f)
    return data['merger']['data_output_base_dir'].format(data["ticker"]["stock"], data["interval"]["interval"])
