import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
