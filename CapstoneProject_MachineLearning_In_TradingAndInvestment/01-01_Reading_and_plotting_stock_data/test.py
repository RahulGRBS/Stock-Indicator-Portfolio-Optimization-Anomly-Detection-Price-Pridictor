import pandas as pd
import matplotlib.pyplot as plt

def get_max_close(symbol):
    """Retuen the maximum closing value for stock indicated by symbol.
    
    Note: Dat for stock is stored in file :data/<symbol>.csv
    """
    
    df = pd.read_csv("data/{}.csv".format(symbol)) #read in the data
    return df['Close'].max() # compute and return maximum

def get_mean_volume(symbol):
    """Return the mean volume for stock indicated by symbol.
    
    Note: Data for a stock is stored in file: data/<symbol>.csv
    """
    df = pd.read_csv("data/{}.csv".format(symbol))  # read in data
    return df['Volume'].mean()

def test_run():
    """"Function called by test_run. """
    #for symbol in ['AAPL','IBM']:
    #    print ("Max Close and Mean Volumne")
    #    print (symbol,get_max_close(symbol),get_mean_volume(symbol))
        
        
    df= pd.read_csv("data/AAPL.csv")
    print(df['Adj Close'])
    df['Adj Close'].plot() 
    plt.show() # must be called to shot the plot
  