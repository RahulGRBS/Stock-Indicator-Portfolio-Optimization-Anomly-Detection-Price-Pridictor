import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

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
    cumulative_returns = (df[-1]/df.iloc[0]) -1 #compute cumulative returns  for row 1 onwards 
       
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
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([portfolio_value, normed_SPY_price*sv], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily portfolio value against SPY stock")
        pass

    return portfolio_cumulative_returns, avg_daily_return, std_daily_ret, sharpe_ratio, end_value


