import requests
import yfinance as yf
from yahoo_fin import stock_info as si
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter



'''
A function that takes the market(eg: US, India etc.) as input and returns a list of symbols which were 
delisted 1-year 1-month after start-date and which are listed prior to 1-year 1-month from today along with current symbols
'''

def collect_tickers(market, start_date, end_date):
    if market.upper() == 'US':
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Fetch the S&P 500 ticker symbols
    symbols = list(pd.read_html(url)[0].Symbol)

    # Remove the symbols which  were listed < 1 years back from current date
    x = pd.read_html(url)[1]

    # Updating the date column to Datetime for calculations
    x.Date = pd.to_datetime(x.Date.Date)
    cutoff_date = end_date - timedelta(days=365)
    stocks_remove = x[x.Date.Date > cutoff_date]
    stocks_remove = stocks_remove['Added']['Ticker']

    # We will add the stocks which were delisted after 1 year 1 month from starting date.
    add_date = pd.to_datetime(start_date) + timedelta(days=396)
    stocks_add = x[x.Date.Date > add_date]
    stocks_add = stocks_add['Removed']['Ticker']

    # Add the stocks_add list to symbols and remove the stocks_remove list from symbols
    symbols.extend(stocks_add)
    symbols = [tickers for tickers in symbols if tickers not in stocks_remove]

    n = len(symbols)
    print(f'Sucesfully downloaded {n} symbols.')
    
    return symbols


'''
A function that takes tickers for which data is to be loaded from start_date to end_date
Input: tickers, start_date, end_date
Output: a dictionary with symbols as keys and the dataframe as values
'''

def download_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        try:
            # Download historical data for each ticker
            data = yf.download(ticker, start=start_date, end=end_date)
            # Store the dataframe in the dictionary
            stock_data[ticker] = data
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")

    print(f'Sucesfully downloaded data for {len(stock_data.keys())} stocks!')
    
    return stock_data

'''
A function that sets all the missing values to 0
'''
def clean_data(tickers, stocks_data, date_range):
    count = 0
    for symbol in tickers:

        try:
            df = stocks_data[symbol]
            # Generate a complete date range from the start to the end of the original DataFrame
            df_reindexed = df.reindex(date_range)

            df_filled = df_reindexed.ffill()
            df_filled = df_filled.ffill().fillna(0)
            stocks_data[symbol] = df_filled
        except:
            count += 1
            tickers.remove(symbol)

    if count > 0:
        print(f'Could not update data for {count} stocks.')
    return stocks_data


'''
A function that takes a list of tickers and returns a dataframe where the (i,j)th call is the return of stock i for (j+p)-th month
Input: symbols(a list of symbols), end_date(date upto which return is to be calculated, by default- today), period
Output: a dataframe
'''
def calculate_return(symbols, stocks_data, end_date=datetime.today().strftime('%Y-%m-%d'), period=1):
    # Initialize a dictionary to store the returns data
    i = 0
    n = len(symbols)
    returns_data = {}
    while i < n:
        try:
            stock_data = stocks_data[symbols[i]][:end_date]['Adj Close']
            
        except:
            i = i+1
            stock_data = stocks_data[symbols[i]][:end_date]['Adj Close']
        resampled_data = stock_data.resample('ME').last()   
        returns = ((resampled_data.shift(-int(period)) - resampled_data) / resampled_data)* 100
        # Store the returns data in the dictionary
        returns_data[symbols[i]] = returns
        i += 1
    # Create a DataFrame from the dictionary
    returns_df = pd.DataFrame(returns_data)

    # Transpose the DataFrame so that timestamps are columns and stock names are rows
    returns_df = returns_df.transpose()

    # Sort the columns (timestamps)
    returns_df = returns_df.sort_index(axis=1)

    return returns_df


'''
A function to calculate the past 12-months momentum
'''
def calculate_momentum(stock_data):
    # Resample to get the last value of each month
    monthly_data = stock_data['Adj Close'].resample('ME').last()
    
    # Calculate the 12-1 momentum returns for each timestamp
    momentum_returns = {}
    for i in range(12, len(monthly_data)):
        period_end = monthly_data.index[i]
        if monthly_data[i-12] != 0:        
            momentum_return = ((monthly_data[i] - monthly_data[i-12]) / monthly_data[i-12]) * 100
        else:
            momentum_return = 0
        momentum_returns[period_end] = momentum_return
    
    return momentum_returns

'''
A function for the 12-1 momentum calculation
'''
def calculate_12_1_momentum(stock_data):
    # Resample to get the last value of each month
    monthly_data = stock_data['Adj Close'].resample('ME').last()
    
    # Calculate the 12-1 momentum returns for each timestamp
    momentum_returns = {}
    for i in range(12, len(monthly_data)):
        period_end = monthly_data.index[i]
        if monthly_data[i-12] != 0:        
            momentum_return = (monthly_data[i-1] - monthly_data[i-12]) / monthly_data[i-12] * 100
        else:
            momentum_return = 0
        momentum_returns[period_end] = momentum_return
    
    return momentum_returns


'''
A function to get the stocks with highest momentum(top p%)
'''
# Function to find the top p-percentile stock symbols(row names) for each column (timestamp)
def top_p_percentile_stocks(df, p):
    top_p_percentile_stocks_dict = {}
    for timestamp in df.columns:
        column_data = df[timestamp].sort_values(ascending=False)
        num_stocks_to_select = int(len(column_data) * p / 100)
        top_p_stocks = column_data.index[:num_stocks_to_select].tolist()
        top_p_percentile_stocks_dict[timestamp] = top_p_stocks
    return top_p_percentile_stocks_dict


'''
A function to get the stocks with lowest momentum(bottom p%)
'''
# Function to find the bottom p-percentile stock symbols(row names) for each column (timestamp)
def bottom_p_percentile_stocks(df, p):
    bottom_p_percentile_stocks_dict = {}
    for timestamp in df.columns:
        column_data = df[timestamp].sort_values(ascending=True)
        num_stocks_to_select = int(len(column_data) * p / 100)
        bottom_p_stocks = column_data.index[:num_stocks_to_select].tolist()
        bottom_p_percentile_stocks_dict[timestamp] = bottom_p_stocks
    return bottom_p_percentile_stocks_dict


'''
A function to calculate the total return of stocks in a list and exclude stock if abs(return) > threshold.
'''
def total_return(stocks_data, momentum_dict, returns_df, threshold=50):
    total_returns_dict = {}
    for timestamp, stocks in momentum_dict.items():
        total_return = 0
        total_invested = 0
        for stock in stocks:
            stock_change = returns_df.loc[stock, timestamp]
            if abs(stock_change) < threshold:
                total_return += (stock_change*stocks_data[stock]['Adj Close'][timestamp - pd.DateOffset(months=1)])/100
                total_invested += stocks_data[stock]['Adj Close'][timestamp - pd.DateOffset(months=1)]
            else:
                total_return = 0

        if total_return != 0:
            total_returns_dict[timestamp] = (total_return/total_invested)*100
        else:
            total_returns_dict[timestamp] = 0

    return total_returns_dict


'''
A function to calculate the total return of stocks in a list and exclude stock if abs(return) > threshold.
'''
def long_sort_return(stocks_data, momentum_dict, sort_dict, returns_df, threshold=50):
    total_returns_dict = {}
    for timestamp, stocks in momentum_dict.items():
        sort = sort_dict[timestamp]
        
        total_return = 0
        total_invested = 0
        for stock in stocks:
            stock_change = returns_df.loc[stock, timestamp]
            if abs(stock_change) < threshold:
                total_return += (stock_change*stocks_data[stock]['Adj Close'][timestamp - pd.DateOffset(months=1)])/100
                total_invested += stocks_data[stock]['Adj Close'][timestamp - pd.DateOffset(months=1)]
            else:
                total_return = 0
        
        for stock in sort:
            stock_change = returns_df.loc[stock, timestamp]
            if abs(stock_change) < threshold:
                total_return -= (stock_change*stocks_data[stock]['Adj Close'][timestamp - pd.DateOffset(months=1)])/100
                total_invested += stocks_data[stock]['Adj Close'][timestamp - pd.DateOffset(months=1)]
            else:
                total_return = 0

        if total_return != 0:
            total_returns_dict[timestamp] = (total_return/total_invested)*100
        else:
            total_returns_dict[timestamp] = 0

    return total_returns_dict


'''
A function to get benchmark data
'''
def benchmark_data(market, start_date, end_date=datetime.today().strftime('%Y-%m-%d')):
    if market.upper() == 'US':
        ticker = '^GSPC'
    data = yf.download(ticker, start_date, end_date)
    data = data['Adj Close']

    return data


'''
A function to get the benchmark returns
'''
def benchmark_returns(data,  period='ME'):    
    monthly_returns = data.resample(period).last().loc['2001-02-28':'2024-04-30']
    gains = monthly_returns.pct_change().dropna()*100

    return gains



"""
    Calculate the m-month moving average of a stock.
    
    Parameters:
    - stock_data (pd.DataFrame): DataFrame containing the stock data with 'Adj Close' prices.
    - m (int): The number of months over which to calculate the moving average.
    
    Returns:
    - pd.Series: A series with the m-month moving average of the adjusted close prices.
"""
def calculate_m_month_moving_average(stock_data, m):
    # Resample to get the last value of each month
    monthly_data = stock_data['Adj Close'].resample('ME').last()
    
    # Calculate the m-month moving average
    moving_average = monthly_data.rolling(window=m).mean()
    
    return moving_average


# Define the function to calculate the total return with m-month moving average filtering
def total_return_with_moving_average(momentum_dict, returns_df, m, stock_data_dict):
    total_returns_dict = {}
    
    for timestamp, stocks in momentum_dict.items():
        valid_stocks = []
        total_return = 0

        for stock in stocks:
            stock_data = stock_data_dict[stock]
            # Calculate the m-month moving average for the stock
            moving_average = calculate_m_month_moving_average(stock_data, m)
            
            # Get the current price and the moving average for the given timestamp
            if timestamp in stock_data.index and timestamp in moving_average.index:
                current_price = stock_data.loc[timestamp, 'Adj Close']
                moving_avg = moving_average.loc[timestamp]
                
                # Check if the current price is greater than the m-month moving average
                if current_price > moving_avg:
                    # Add the stock's return to the total return
                    stock_return = returns_df.loc[stock, timestamp]
                    total_return += stock_return
                    valid_stocks.append(stock)
        
        # Update the momentum_dict to only include valid stocks
        momentum_dict[timestamp] = valid_stocks
        # Calculate the average return if there are valid stocks
        if valid_stocks:
            total_return /= len(valid_stocks)
        
        total_returns_dict[timestamp] = total_return
    
    return total_returns_dict


def seasonality_strategy(returns_df, top_p_momentum_dict, max_abs_return=50):
    trade_returns = {}
    
    for timestamp in returns_df.columns:
        month = timestamp.month
        if 1 <= month <= 5 or month == 12:  # Buy the top momentum stocks from December to May
            if timestamp in top_p_momentum_dict:
                stocks = top_p_momentum_dict[timestamp]
                stock_returns = returns_df.loc[stocks, timestamp]
                filtered_returns = stock_returns[stock_returns.abs() <= max_abs_return]
                total_return = filtered_returns.mean()
                trade_returns[timestamp] = total_return

    trade_returns = pd.DataFrame.from_dict(trade_returns, orient='index', columns=['Trade Returns'])
    trade_returns.sort_index(inplace=True)

    return trade_returns


# def seasonality_strategy(returns_df, top_p_momentum_dict, snp_returns, max_abs_return=50):
#     trade_returns = {}
#     returns_df = returns_df.loc[:, '2001-02-28 00:00:00':'2024-05-31 00:00:00']
#     snp_returns = snp_returns.loc['2001-02-28 00:00:00':]
#     for timestamp in returns_df.columns:
#         month = timestamp.month
#         if 1 <= month <= 5 or month == 12:  # Buy the top momentum stocks from December to May
#             if timestamp in top_p_momentum_dict:
#                 stocks = top_p_momentum_dict[timestamp]
#                 stock_returns = returns_df.loc[stocks, timestamp]
#                 filtered_returns = stock_returns[stock_returns.abs() <= max_abs_return]
#                 total_return = filtered_returns.mean()
#                 trade_returns[timestamp] = total_return
#         else:
#             trade_returns[timestamp] = snp_returns.loc[timestamp]

#     trade_returns = pd.DataFrame.from_dict(trade_returns, orient='index', columns=['Trade Returns'])
#     trade_returns.sort_index(inplace=True)

#     return trade_returns


def calculate_yearly_stats(strategy_dict, benchmark_returns, risk_free_rate=0):
    # Convert the strategy dictionary to a DataFrame
    strategy_returns_df = pd.Series(strategy_dict)
    strategy_returns_df.index = pd.to_datetime(strategy_returns_df.index)
    strategy_returns_df = strategy_returns_df.resample('YE').apply(lambda x: ((1 + x / 100).prod() - 1) * 100)
    print(strategy_returns_df)
    # Calculate average yearly returns for the strategy
    average_yearly_returns = strategy_returns_df.mean()

    # Calculate the standard deviation of yearly returns
    std_dev_yearly_returns = strategy_returns_df.std()

    # Calculate the Sharpe ratio
    sharpe_ratio = (average_yearly_returns - risk_free_rate) / std_dev_yearly_returns

    # Convert the benchmark returns to a DataFrame
    benchmark_returns = benchmark_returns.resample('YE').apply(lambda x: ((1 + x / 100).prod() - 1) * 100)
    
   
    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Year': strategy_returns_df.index.year,
        'Strategy Returns': strategy_returns_df.values,
        'Benchmark Returns': benchmark_returns.values,
        'Strategy Sharpe Ratio': [sharpe_ratio] * len(strategy_returns_df),
    })

    results_df.set_index('Year', inplace=True)
    
    return results_df
