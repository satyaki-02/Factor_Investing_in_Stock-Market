import yfinance as yf
from yahoo_fin import stock_info as si
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

'''-----------------------------------DATA COLLECTION-------------------------------'''
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
    cutoff_date =  pd.to_datetime(end_date) - timedelta(days=365)
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
    data = yf.download(tickers, start_date, end_date)
    data = data['Adj Close']
    data = data.dropna(axis=1, how='all')
    return data

'''
A function that sets all the missing values to 0
'''

def clean_data(stocks_data, full_date_range):
    
    # Reindex the DataFrame to include weekends
    stocks_data_full = stocks_data.reindex(full_date_range)
    
    # Forward fill the missing values to propagate Friday's values to Saturday and Sunday
    stocks_data_full_ffill = stocks_data_full.ffill()
    
    return stocks_data_full_ffill

'''
A function that takes a list of tickers and returns a dataframe where the (i,j)th call is the return of stock i for (j+p)-th month
Input: symbols(a list of symbols), end_date(date upto which return is to be calculated, by default- today), period
Output: a dataframe
'''

def calculate_returns(stocks_data, rebalance='W', periods=1):
    # Ensure the index is a datetime index
    
    # Resample to weekly data, using the end of the week as the label
    weekly_data = stocks_data.resample(rebalance).last()
    
    # Calculate the percentage change
    weekly_pct_change = weekly_data.pct_change(periods=1)*100
    
    return weekly_pct_change.shift(-1).T



'''
A function to calculate the total return of stocks in a list and exclude stock if abs(return) > threshold.
'''
def total_return(stocks_data, momentum_dict, returns_df, threshold=50):
    total_returns_dict = {}
    for timestamp, stocks in momentum_dict.items() :
        total_return = 0
        total_invested = 0
        for stock in stocks:
            stock_change = returns_df.loc[stock, timestamp]
            if abs(stock_change) < threshold:
                total_return += (stock_change*stocks_data[stock][timestamp - pd.DateOffset(weeks=1)])/100
                total_invested += stocks_data[stock][timestamp - pd.DateOffset(weeks=1)]

        if total_return != 0:
            total_returns_dict[timestamp] = ((total_return)/total_invested)*100
        else:
            total_returns_dict[timestamp] = 0

    return total_returns_dict


'''
A function to get the stocks with highest momentum(top p percentage)
'''
def top_p_percentage_stocks(df, p):
    top_p_percentage_stocks_dict = {}
    for timestamp in df.columns:
        column_data = df[timestamp].sort_values(ascending=False)
        num_stocks_to_select = int(len(column_data) * p / 100)
        top_p_stocks = column_data.index[:num_stocks_to_select].tolist()
        top_p_percentage_stocks_dict[timestamp] = top_p_stocks
    return top_p_percentage_stocks_dict


'''
A function to get the stocks with lowest momentum(bottom p%)
'''
def bottom_p_percentage_stocks(df, p):
    bottom_p_percentage_stocks_dict = {}
    for timestamp in df.columns:
        column_data = df[timestamp].sort_values(ascending=True)
        num_stocks_to_select = int(len(column_data) * p / 100)
        bottom_p_stocks = column_data.index[:num_stocks_to_select].tolist()
        bottom_p_percentage_stocks_dict[timestamp] = bottom_p_stocks
    return bottom_p_percentage_stocks_dict


'''
A function to get the stocks with highest momentum(top p percentile)
'''
# Function to find the top p-percentile stock symbols(row names) for each timestamp (column)
def top_p_percentile_stocks(df, p):
    top_p_percentile_stocks_dict = {}
    for timestamp in df.columns:
        column_data = df[timestamp].sort_values(ascending=False)
        
        # Calculate the p-th percentile value
        threshold_value = column_data.quantile((100 - p) / 100.0)
        
        # Select stocks below the p-th percentile value
        top_p_stocks = column_data[column_data <= threshold_value].index.tolist()
        
        top_p_percentile_stocks_dict[timestamp] = top_p_stocks
    return top_p_percentile_stocks_dict


'''
A function to get the stocks with lowest momentum(bottom p-percentile)
'''
# Function to find the bottom p-percentile stock symbols(row names) for each column (timestamp)
def bottom_p_percentile_stocks(df, p):
    bottom_p_percentile_stocks_dict = {}
    for timestamp in df.columns:
        column_data = df[timestamp].sort_values(ascending=True)
        
        # Calculate the p-th percentile value
        threshold_value = column_data.quantile(p / 100.0)
        
        # Select stocks below the p-th percentile value
        bottom_p_stocks = column_data[column_data <= threshold_value].index.tolist()
        
        bottom_p_percentile_stocks_dict[timestamp] = bottom_p_stocks
    return bottom_p_percentile_stocks_dict


'''
A function to calculate the total return of stocks in a list and exclude stock if abs(return) > threshold.
'''
def long_sort_return(stocks_data, long_dict, short_dict, returns_df, threshold=50):
    total_returns_dict = {}
    
    # Get all unique timestamps from both dictionaries
    all_timestamps = set(short_dict.keys()).union(set(long_dict.keys()))
    
    for timestamp in all_timestamps:
        longs = long_dict.get(timestamp, [])
        shorts = short_dict.get(timestamp, [])
        
        total_return = 0
        total_invested = 0
        
        # Process long stocks
        for stock in longs:
            stock_change = returns_df.loc[stock, timestamp]
            if abs(stock_change) < threshold:
                total_return += (stock_change * stocks_data[stock][timestamp - pd.DateOffset(weeks=1)]) / 100
                total_invested += stocks_data[stock][timestamp - pd.DateOffset(weeks=1)]
            else:
                total_return = 0
        
        # Process short stocks
        for stock in shorts:
            stock_change = returns_df.loc[stock, timestamp]
            if abs(stock_change) < threshold:
                total_return -= (stock_change * stocks_data[stock][timestamp - pd.DateOffset(weeks=1)]) / 100
                total_invested += stocks_data[stock][timestamp - pd.DateOffset(weeks=1)]
            else:
                total_return = 0
        
        if total_return != 0:
            total_returns_dict[timestamp] = (total_return / total_invested) * 100
        else:
            total_returns_dict[timestamp] = 0

    return total_returns_dict


def portfolio_return(returns_df, buy_dict, sell_dict, threshold=50):
    portfolio_returns = {}

    all_dates = set(buy_dict.keys()).union(sell_dict.keys())

    for date in all_dates:
        total_buy_return = 0
        total_sell_return = 0
        total_stocks = 0

        if date in buy_dict:
            for stock in buy_dict[date]:
                if stock in returns_df.index and date in returns_df.columns:
                    stock_return = returns_df.at[stock, date]
                    if abs(stock_return) < threshold:
                        total_buy_return += stock_return
                        total_stocks += 1

        if date in sell_dict:
            for stock in sell_dict[date]:
                if stock in returns_df.index and date in returns_df.columns:
                    stock_return = returns_df.at[stock, date]
                    if abs(stock_return) < threshold:
                        total_sell_return += stock_return
                        total_stocks += 1

        if total_stocks > 0:
            total_return = (total_buy_return - total_sell_return) / total_stocks
            portfolio_returns[date] = total_return
        else:
            portfolio_returns[date] = 0

    return portfolio_returns



'''----------------------------------------STRATEGIES---------------------------------------------'''
'''
A function that takes a particular stock and returns the m-month momentum
of the stock
'''
def calculate_m_momentum(stock_data, m, resample = 'W'):
    if resample == 'W':
        if m == 12:
            periods=52
        else:
            periods=m*4
        # Resample to get the last value of each week (Sunday)
        data = stock_data.resample('W').last()
    elif resample == 'ME':
        data = stock_data.resample('ME').last()
        periods = m
    
    # Calculate the percentage change over m*4 weeks
    momentum_returns = data.pct_change(periods=periods) * 100
    
    # Replace any infinite values with 0
    momentum_returns.replace([np.inf, -np.inf], 0, inplace=True)
    
    return momentum_returns.T.dropna(axis=1, how='all')


'''
A function for the 12-1 momentum calculation
'''
def calculate_12_1_momentum(stock_data, resample = 'W'):
    # Resample to get the last value of each week (Sunday)
    if resample == 'W':
        data = stock_data.resample('W').last()
        
        # Shift the weekly data by 4 weeks (1 month)
        shifted_data = data.shift(periods=4)
        
        # Calculate the 11-month momentum returns for each timestamp
        momentum_returns = shifted_data.pct_change(periods=48) * 100  # 11-month momentum (44 weeks)

    elif resample == 'ME':
        data = stock_data.resample('ME').last()
        
        # Shift the weekly data by 4 weeks (1 month)
        shifted_data = data.shift(periods=1)
        
        # Calculate the 11-month momentum returns for each timestamp
        momentum_returns = shifted_data.pct_change(periods=11) * 100  # 11-month momentum (44 weeks)
    
    # Replace any infinite values with 0
    momentum_returns.replace([np.inf, -np.inf], 0, inplace=True)

    return momentum_returns.T.dropna(axis=1, how='all')




'''
Price above m-month MA
'''
"""
    Calculate the m-month moving average of a stock.
    
    Parameters:
    - stock_data (pd.DataFrame): DataFrame containing the stock data with 'Adj Close' prices.
    - m (int): The number of months over which to calculate the moving average.
    
    Returns:
    - pd.Series: A series with the m-month moving average of the adjusted close prices.
"""
def calculate_m_week_moving_average(stock_data, m):
    # Resample to get the last value of each week (Sunday)
    weekly_data = stock_data['Adj Close'].resample('W-SUN').last()
    
    # Calculate the m-week moving average (equivalent to 6-month moving average with weekly data)
    moving_average = weekly_data.rolling(window=m).mean()
    
    return moving_average

# Define the function to calculate the total return with m-week moving average filtering
def stocks_moving_average(momentum_dict, m, stock_data_dict):
    stock_dict = {}
    
    for timestamp, stocks in momentum_dict.items():
        valid_stocks = []

        for stock in stocks:
            stock_data = stock_data_dict[stock]
            # Calculate the m-week moving average for the stock
            moving_average = calculate_m_week_moving_average(stock_data, m)
            
            # Get the current price and the moving average for the given timestamp
            if timestamp in stock_data.index and timestamp in moving_average.index:
                current_price = stock_data.loc[timestamp, 'Adj Close']
                moving_avg = moving_average.loc[timestamp]
                
                # Check if the current price is greater than the m-week moving average
                if current_price > moving_avg:
                    valid_stocks.append(stock)
        
        # Update the momentum_dict to only include valid stocks
        stock_dict[timestamp] = valid_stocks
        
    return stock_dict

'''
VALUE STRATEGY
'''
def calculate_p_percentile(price_series, p):
    # Calculate the p-percentile of the price series
    return np.percentile(price_series, p)


def value_strategy(stocks_data, percentile=25):
    
    # Create a dictionary with lists of stocks in the specified percentile as values and weekend dates as keys
    result_dict = {}

     # Get the weekly resampled data
    weekly_data = stocks_data.resample('W').last()   
    # Iterate over each weekend date
    for date in weekly_data.index[52:]:
        if date in stocks_data.index:
            # Get the last 52 weeks of data
            last_52_weeks = stocks_data.loc[date - pd.DateOffset(weeks=52):date]
            
            # Calculate the percentile value for each stock
            percentile_values = last_52_weeks.apply(lambda x: calculate_p_percentile(x.dropna(), percentile)if x.dropna().shape[0] > 0 else np.nan)
            
            # Check if the current value is below the percentile
            current_prices = stocks_data.loc[date]
            valid_stocks = current_prices[current_prices < percentile_values].index.tolist()
            
            # Update the result dictionary
            result_dict[date] = valid_stocks
    
    return result_dict


def value_strategy_ls(stocks_data, percentile_buy, percentile_sell):
   
    # Create dictionaries to hold buy and sell lists with weekend dates as keys
    result_dict_buy = {}
    result_dict_sell = {}

    # Get the weekly resampled data
    weekly_data = stocks_data.resample('W').last()

    # Iterate over each weekend date starting from the 53rd week to ensure 52 weeks of past data
    for date in weekly_data.index[52:]:
        if date in stocks_data.index:
            # Initialize lists to store valid stocks for buy and sell
            valid_stocks_buy = []
            valid_stocks_sell = []
            
            # Iterate over each stock
            for stock in stocks_data.columns:
                # Get the last 52 weeks of data for the stock
                last_52_weeks = stocks_data.loc[date - pd.DateOffset(weeks=52):date, stock].dropna()
                
                if last_52_weeks.shape[0] > 0:
                    # Calculate the percentile values for buy and sell
                    percentile_value_buy = calculate_p_percentile(last_52_weeks, percentile_buy)
                    percentile_value_sell = calculate_p_percentile(last_52_weeks, percentile_sell)
                    
                    # Get the current price of the stock
                    current_price = stocks_data.loc[date, stock]
                    
                    # Check if the current price is below the buy percentile value or above the sell percentile value
                    if current_price < percentile_value_buy:
                        valid_stocks_buy.append(stock)
                    if current_price > percentile_value_sell:
                        valid_stocks_sell.append(stock)
            
            # Update the result dictionaries
            result_dict_buy[date] = valid_stocks_buy
            result_dict_sell[date] = valid_stocks_sell
    
    return result_dict_buy, result_dict_sell


'''
Volatility
'''
def calculate_vol(price_data, y=1, resample='W'):
    if resample == 'W':
        time = 52
    elif resample == 'ME':
        time = 12
    volatility = price_data.T.rolling(window=time*y).std()
    volatility = volatility * np.sqrt(time)
    
    return volatility.T.dropna(axis=1, how='all')

'''
SEASONALITY
'''
def calculate_return_signals(return_df, y, rebalance='ME'):
    results_dict = {stock: {} for stock in return_df.index}
    
    for current_date in return_df.columns:
        for stock in return_df.index:
            signals = [0 for _ in range(y)]
            for i in range(1, y + 1):
                if rebalance == 'W':
                    past_date = current_date - pd.DateOffset(weeks=52*i)
                elif rebalance == 'ME':
                    past_date = current_date - pd.DateOffset(months=12*i)
                
                if past_date in return_df.columns:
                    past_return = return_df.at[stock, past_date]
                    if np.isnan(past_return):
                        signal = 0
                    elif past_return > 0:
                        signal = 1
                    elif past_return < 0:
                        signal = -1
                    signals[i-1] = signal

            # Store the signals if we have y years of data
            results_dict[stock][current_date] = signals

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    
    return results_df


def generate_buy_sell_signals(signals_df, m=1, n=1):
    buy_dict = {}
    sell_dict = {}

    # Iterate through each date (timestamp)
    for date in signals_df.columns:
        buy_stocks = []
        sell_stocks = []

        # Iterate through each stock in the DataFrame
        for stock in signals_df.index:
            signals = signals_df.at[stock, date]

            # Ensure that signals is a list of integers
            if isinstance(signals, str):
                signals = eval(signals)  # Convert string representation of list to actual list

            if isinstance(signals, list):
                # Convert elements to integers if they are not
                signals = [int(x) for x in signals]
                if signals.count(1) >= m:
                    buy_stocks.append(stock)
                if signals.count(-1) >= n:
                    sell_stocks.append(stock)

                if signals.count(1) >= m:
                    buy_stocks.append(stock)
                if signals.count(-1) >= n:
                    sell_stocks.append(stock)

        # Store the lists in the dictionaries if there are any stocks to buy or sell
        if buy_stocks:
            buy_dict[date] = buy_stocks
        if sell_stocks:
            sell_dict[date] = sell_stocks

    return buy_dict, sell_dict


'''---------------------------------------METRICES-------------------------------------------------------------'''


'''
I/P: A dataframe with return% and number of years for which SR is to be calculated
O/P: A df with strategy names as column names and Sr as values
'''
def y_years_sharpe(combined_df, y, risk_free_rate=0):
    # Ensure the index is a datetime index
    combined_df.index = pd.to_datetime(combined_df.index)
    
    # Define the start year based on y
    start_year = combined_df.index.year.max() - y + 1
    
    # Filter the data to start from the calculated start year
    filtered_df = combined_df[combined_df.index.year >= start_year]
    
    # Initialize a dictionary to store the Sharpe ratios
    sharpe_ratios = {}

    # Iterate over each strategy
    for strategy in combined_df.columns:
        returns = filtered_df[strategy].values  

        mean_return = np.mean(returns)
        std_dev = np.std(returns, ddof=1)

        # Calculate the Sharpe ratio
        sharpe_ratio = (mean_return  - risk_free_rate) / std_dev  # Annualized mean return
        
        sharpe_ratios[strategy] = sharpe_ratio*np.sqrt(52*y)

    # Convert to DataFrame
    sharpe_ratio_df = pd.DataFrame.from_dict(sharpe_ratios, orient='index', columns=['Sharpe Ratio'])

    return sharpe_ratio_df.T.round(2)


'''
A function that takes a df as input with datetime as index and monthly-return% as data
'''
def y_years_cagr(combined_df, y):
    # Ensure the index is a datetime index
    combined_df.index = pd.to_datetime(combined_df.index)
    
    # Define the start year based on y
    start_year = combined_df.index.year.max() - y + 1
    
    # Filter the data to start from the calculated start year
    filtered_df = combined_df[combined_df.index.year >= start_year]
    
    
    # Initialize a dictionary to store the CAGRs
    cagr_dict = {}

    # Iterate over each strategy
    for strategy in combined_df.columns:
        returns = filtered_df[strategy].values / 100  # Convert to decimal
        # Calculate the cumulative product of the returns over the period
        cum_prod = ((1+returns).cumprod()[-1])  # Cumulative product of returns

        # Calculate CAGR
        cagr = (cum_prod ** (1 / y)) - 1

        cagr_dict[strategy] = cagr * 100  # Convert to percentage

    # Convert to DataFrame
    cagr_df = pd.DataFrame.from_dict(cagr_dict, orient='index', columns=['CAGR'])

    return cagr_df


'''--------------------------OVERALL SR AND CAGR-----------------------------------------------------------'''
# Converting the results of dict to df
def combine_strategy_returns(strategy_returns_dict):
    # Create an empty list to store DataFrames
    dfs = []
    
    # Iterate over the dictionary
    for strategy, df in strategy_returns_dict.items():
        # Rename the column to the strategy name
        df = df.rename(columns={df.columns[0]: strategy})
        # Append the DataFrame to the list
        dfs.append(df)
    
    # Concatenate all DataFrames in the list along the columns
    combined_df = pd.concat(dfs, axis=1)
    
    return combined_df


def calculate_cagr(strategy_returns, rebalance):
    if rebalance == 'W':
        period = 52
    elif rebalance == 'ME':
        period = 12
    # Ensure the returns are in decimal form
    strategy_returns = strategy_returns / 100
    
    # Calculate the cumulative return
    cumulative_return = (1 + strategy_returns).cumprod()[-1]
    
    # Calculate the number of years
    n_years = len(strategy_returns) / period
    
    # Calculate the CAGR
    cagr = (cumulative_return ** (1 / n_years))-1
    
    return (cagr * 100).round(2)  # Convert to percentage

def calculate_overall_cagrs(strategies_df, rebalance):
    cagr_dict = {}

    for strategy in strategies_df.columns:
        strategy_returns = strategies_df[strategy].dropna()
        cagr = calculate_cagr(strategy_returns, rebalance)
        cagr_dict[strategy] = cagr

    cagr_series = pd.Series(cagr_dict, name='CAGR')

    return cagr_series.round(2)


def calculate_sharpe_ratio(strategy_returns, rebalance, risk_free_rate=0):
    if rebalance == 'W':
        period = 52
    elif rebalance == 'ME':
        period = 12
    risk_free_rate = risk_free_rate/100
    # Ensure the returns are in decimal form
    strategy_returns = strategy_returns / 100
    
    # Calculate the mean and standard deviation of returns
    mean_return = strategy_returns.mean()
    std_dev = strategy_returns.std()
    # Calculate the Sharpe ratio
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    
    # Annualize the Sharpe ratio assuming weekly returns
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(period)
    
    return sharpe_ratio_annualized.round(2)

def calculate_overall_sharpe_ratios(strategies_df, rebalance, risk_free_rate=0):
    
    sharpe_ratios = {}

    for strategy in strategies_df.columns:
        strategy_returns = strategies_df[strategy].dropna()
        sharpe_ratio = calculate_sharpe_ratio(strategy_returns, rebalance, risk_free_rate)
        sharpe_ratios[strategy] = sharpe_ratio

    sharpe_ratios_series = pd.Series(sharpe_ratios, name='Sharpe Ratio')

    return sharpe_ratios_series.round(2)

'''-----------------------------------------PLOT----------------------------------------------------------------'''



'''-----------------------------------------ML----------------------------------------------------------------'''
# Function to melt DataFrame
def melt_df(df, value_name):
    df = df.iloc[:, :-1]
    melted_df = df.reset_index().melt(id_vars=['Ticker'], var_name='Date', value_name=value_name)
    melted_df.rename(columns={'Ticker': 'Stock'}, inplace=True)
    return melted_df


def convert_df_values(df, m, n):

    def convert_value(x):
        if x > m:
            return 1
        elif x < n:
            return -1
        else:
            return 0
    df.fillna(0)
    return df.applymap(convert_value)


def convert_signals(seasonality_df, m):
    def convert_list(signal_list):
        # Ensure the signal_list is a list
        if not isinstance(signal_list, list):
            return 0  # Handle unexpected types by returning 0
        
        # Check if the list contains all 0 values
        if all(x == 0 for x in signal_list):
            return 0
        
        # Count the number of +1 values
        plus_one_count = signal_list.count(1)
        
        # Convert to +1 if the count of +1 values is greater than m, otherwise to -1
        return 1 if plus_one_count >= m else -1

    # Apply the conversion function to each element in the DataFrame
    converted_df = seasonality_df.applymap(convert_list)
    
    return converted_df


def normalize_columns(df, columns):
    for col in range(len(columns)):
        mean = df.iloc[:, col].mean()
        std = df.iloc[:, col].std()
        df.iloc[:, col] = (df.iloc[:, col] - mean) / std
    return df

# Plot the vol-adjusted graph against benchmark individually
def vol_adj_plot(df, benchmark):
    # Calculate the volatility (standard deviation) of each strategy and the benchmark
    volatilities = df.std()

    # Extract the volatility of the benchmark
    benchmark_volatility = volatilities['Adj Close']

    # Calculate the scaling factor for each strategy to match the benchmark volatility
    scaling_factors = benchmark_volatility / volatilities

    # Apply the scaling factor to each strategy
    adjusted_df = df.mul(scaling_factors, axis=1)

    # Calculate the cumulative returns
    cumulative_returns = (1 + adjusted_df / 100).cumprod()

    # Generate individual plots for each strategy
    for column in cumulative_returns.columns:
        if column != 'Adj Close':
            # Create Plotly traces
            fig = go.Figure()

            # Add the benchmark trace
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['Adj Close'], mode='lines', name=benchmark))

            # Add the adjusted strategy trace
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[column], mode='lines', name=f'{column} '))

            # Update layout
            fig.update_layout(
                title=f'{benchmark} vs {column} ',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                template='plotly_dark'
            )
            
            # Show the figure
            fig.show()

# Plot cumulative graph, all in one
def calculate_cumulative_returns(df):
    cumulative_returns_df = (1 + df / 100).cumprod()
    return cumulative_returns_df

def plot_all_strategy_performance(strategy_df):
    # Ensure the index is a datetime index
    strategy_df.index = pd.to_datetime(strategy_df.index)
    
    # Calculate cumulative returns
    cumulative_returns_df = calculate_cumulative_returns(strategy_df)
    
    # Create the plot
    fig = go.Figure()

    # Add traces for each strategy
    for strategy in cumulative_returns_df.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_returns_df.index,
            y=cumulative_returns_df[strategy],
            mode='lines',
            name=strategy
        ))
    
    # Add titles and labels
    fig.update_layout(
        title='Cumulative Returns of Strategies',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend_title='Strategies'
    )

    # Show the plot
    fig.show()
