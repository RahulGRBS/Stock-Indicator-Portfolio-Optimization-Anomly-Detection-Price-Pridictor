import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import math
import scipy.optimize as spo
from sklearn.preprocessing import MinMaxScaler # Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler #Import StandardScaler to normalize the price data in the training data



def symbol_to_path(symbol, base_dir="../data"):
    """Return CSV file path for the given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates,colname = 'Adj Close'):
    # Read Data
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')
    for symbol in symbols:
        df_temp= pd.read_csv(symbol_to_path(symbol), index_col= "Date", parse_dates= True, usecols=['Date', colname], na_values=['nan'])
        #rename to prevent clash
        df_temp = df_temp.rename(columns={colname: symbol})
        df=df.join(df_temp) #use default left
        if symbol == 'SPY': # drop dates SPY did not trade 
            df = df.dropna(subset=["SPY"])
            
    # remove SPY from symbols otherwise it will drag itself to the get_data function call
    symbols.pop(0)
                
    return df

def plot_data(df, title="Stock prices", xlabel="Date" , ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
def fill_missing_values(df):
    """Fill missing values (first forward then backward) in data frame, in place."""
    df.fillna(method ='ffill', inplace= True)
    df.fillna(method ='bfill', inplace= True)
    return df

def normalized_data(df):
    """Normalize stock prices using the first row of the dataframe"""
    return (df/df.iloc[0])

def normalised_MinMaxScaler(df,featureRange_0To1=False):
    """
    Normalises the data values using MinMaxScaler from sklearn
    :param data: a DataFrame 
    :return: a DataFrame with normalised value for all the columns except index
    """
    # Initialize a scaler, then apply it to the features
    if (not featureRange_0To1):
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = MinMaxScaler()
    columns = list(df.columns)  
    df[columns] = scaler.fit_transform(df[columns])
    return df

def normalised_StandardScaler(df):
    """
    Normalises the data values using StandardScaler from sklearn
    :param data: a DataFrame 
    :return: a DataFrame with normalised value for all the columns except index
    """
    # Initialize a scaler, then apply it to the features
    scaler = StandardScaler()
    columns = list(df.columns)
    df[columns] = scaler.fit_transform(df[columns])
    return df

def get_rolling_mean(prices, window=14):
    """Return rolling mean of given prices, using specified window size."""
    #return pd.rolling_mean(values, window=window)
    return prices.rolling(window, min_periods=1, center=False).mean()

def get_rolling_std(prices, window=14):
    """Return rolling standard deviation of given values, using specified window size."""
    return prices.rolling(window,min_periods=window,center=False).std() #Volatility

def get_bollinger_bands(prices, rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    #bollinger_band_percentage = (prices- lower_band)/(upper_band-lower_band)
    bollinger_band_percentage = (prices- rm)/(2*rstd)
    return upper_band,lower_band, bollinger_band_percentage

def get_indicators(normed_syms_price, sym):
    
    prices=normed_syms_price[sym]
    sym_indicators = pd.DataFrame(index=prices.index)
    sym_indicators['price']= prices
    #Simple Moving Average
    sym_indicators['rolling_mean'] = get_rolling_mean(normed_syms_price)
    #Volatility
    sym_indicators['vol_std']= get_rolling_std(normed_syms_price)
    #Bollinger bands
    upper_band, lower_band, bbp = get_bollinger_bands(sym_indicators['price'],sym_indicators['rolling_mean'],sym_indicators['vol_std'])
    sym_indicators['upper_band']= upper_band
    sym_indicators['lower_band']= lower_band
    sym_indicators['bollinger_band_percent']=bbp
    
    return sym_indicators.dropna()

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()           #copy the given Dataframe to match size and column names
    daily_returns[1:] = (df[1:]/df[:-1].values) -1
    #Alternate way of doing of above two steps
    #daily_returns = (df/df.shift(1)) -1 #compute daily returns  for row 1 onwards  
    daily_returns.ix[0] =0         #set daily returns for row 0 to zero otherwise pandas will assign nan values
    return daily_returns

def compute_cumulative_returns(df):
    """Compute and return the cumulative return values."""
    #cumulative_returns = (df[-1]/df[0]) -1
    cumulative_returns = (df/df.iloc[0]) -1 #compute cumulative returns  for row 1 onwards 
    #df.dropna()
    #cumulative_returns = (df+1).cumprod()
    return cumulative_returns

def sharpe_ratio_funtion(allocs, normed_syms_price):
    """Compute and return the sharpe ratio."""
    # Get daily portfolio value
    allocated       = normed_syms_price * allocs
    position_value  = allocated 
    portfolio_value = position_value.sum(axis=1)
    # Daily return
    daily_returns_portfolio = compute_daily_returns(portfolio_value)
    # Daily standard deviation 
    std_daily_ret = daily_returns_portfolio.std()
    # Sharpe ratio
    sf= 252                           #trading_days 252 when computed daily
    rfr =0.0
    k= np.sqrt(sf) 
    rp= daily_returns_portfolio - rfr #risk_free_rate
    sharpe_ratio = k * (rp.mean()/std_daily_ret)
    # -1 is multiplied beacuse notion is to mazimize the sharpe ration 
    # but spo_optimizer minimize the function and give smallest value of sharpe ratio
    negative_sharpe_ratio = sharpe_ratio *-1
    return (negative_sharpe_ratio)

def normalized_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM']):
        
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all = fill_missing_values(prices_all)
    price_syms = prices_all[syms]  # only portfolio symbols
    price_SPY  = prices_all['SPY'] # only SPY, for comparison later
    normed_syms_price = normalized_data(price_syms)   # only portfolio symbols
    normed_SPY_price  = normalized_data(price_SPY)  # only SPY, for comparison later
    return normed_syms_price, normed_SPY_price

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    normed_syms_price,normed_SPY_price  = normalized_portfolio(sd,ed,syms)
    # Get daily portfolio value
    allocated       = normed_syms_price * allocs
    position_value  = allocated*sv 
    portfolio_value = position_value.sum(axis=1)  # daily portfolio values
    # Get portfolio statistics (note: std_daily_ret = volatility)
    # Average daily return
    daily_returns_portfolio = compute_daily_returns(portfolio_value)
    avg_daily_return = daily_returns_portfolio.mean()
    # Daily standard deviation 
    std_daily_ret = daily_returns_portfolio.std()
    # Cumulative return
    portfolio_cumulative_returns = compute_cumulative_returns(portfolio_value)
    # Sharpe ratio
    k= np.sqrt(sf) #trading_days = sf
    rp= daily_returns_portfolio - rfr #risk_free_rate
    sharpe_ratio = k * (rp.mean()/std_daily_ret)
    # Compute end value
    end_value= portfolio_value[-1]
    
    # Compute CAPM equation
    daily_returns_syms = compute_daily_returns(normed_syms_price*sv)
    daily_returns_SPY = compute_daily_returns(normed_SPY_price*sv)
    # Appending daily portfolio returns
    daily_returns_all = pd.concat([daily_returns_SPY,daily_returns_syms, daily_returns_portfolio], axis=1)  
    daily_returns_all = daily_returns_all[1:] # Remove first row "0" for portfolio calculations
    daily_returns_all.rename(columns={0:"Portfolio"}, inplace=True)
    # ri(t) = beta*rm(t) + alpha(t)
    beta_portfolio, alpha_portfolio =np.polyfit(daily_returns_all['SPY'], daily_returns_all['Portfolio'],1)
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([portfolio_value, normed_SPY_price*sv], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily portfolio value against SPY stock")
    # Scatter Plot - CAPM b/w SPY and Portfolio
        daily_returns_all.plot(kind='scatter', x='SPY', y='Portfolio')
        plt.plot(daily_returns_all['SPY'], beta_portfolio * daily_returns_all['SPY'] + alpha_portfolio, '-', color='red')
        plt.show()
        
        pass
    
    return portfolio_cumulative_returns, avg_daily_return, std_daily_ret, sharpe_ratio, end_value,beta_portfolio,alpha_portfolio 

def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    
    normed_syms_price,normed_SPY_price  = normalized_portfolio(sd,ed,syms)
    # Guessing that the allocation equally divided among the portfolio stocks
    allocation_guesses = np.asarray([1.0/len(syms)] * len(syms)) 
    # Find the allocations for the optimal portfolio
    bounds = [(0.0, 1.0)] * len(syms)
    #Lambda function applies sum_constraint such that return value must come back as 0 to be accepted
    #if return value is anything other than 0 it's rejected as not a valid answer.
    constraint = {'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}  
    optimized_allocations = spo.minimize(sharpe_ratio_funtion, allocation_guesses,args=(normed_syms_price, ),\
                             method='SLSQP', options={'disp':True}, bounds=bounds, constraints=constraint).x
    #allocated = normed_syms_price * optimized_allocations
    return optimized_allocations

def get_train_test(df, size=0.8):
    '''
    Not sure to use model_selection.train_test_split because it doesn't do roll-forward cross validation
    prams : dataframe , default train size as 80% percent
    return: test and train dataframe
    '''
    train_size = int(len(df) * size)  # default 80% of the dataframe will be used for training
    test_size  = len(df) - train_size # default 20% of the dataframe will be used for testing
    train, test = df[0:train_size], df[train_size:len(df)] # Remove iloc for array as dataframe iloc method specify the slicing by index
    
    return train, test

def add_timesteps(X, y, time_steps=30):
    '''
    LSTM requires converting input data into 3-D array combining TIME_STEPS
    params: dataframe X , dataframe y , set default TIME_STEPS=30 so that network have memory of 30 days
    return  arrays
    '''
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    
    return np.array(Xs), np.array(ys)


def make_data_matrix(prc_arry, look_back=10):
    '''
    Convert an array of values into a dataset matrix
    params: dataframe X , dataframe y , set default look_back=10 so that network have memory of 10 days
    return  arrays
    '''
    dataX, dataY = [], []
    for i in range(len(prc_arry)-look_back-1):
        dataX.append(prc_arry[i:(i+look_back),0])
        dataY.append(prc_arry[i + look_back,0])
    
    return np.array(dataX), np.array(dataY)
