from helper import *


'''-----------------------------------MOMENTUM----------------------------------------------------------------------'''
def momentum_long(snp_stocks_data, momentum_df, returns_df,p,strategy, strategy_returns):
    # Stocks in the top p-percentile 
    top_p_momentum_stocks_dict = top_p_percentage_stocks(momentum_df, p)
    # Returns from longing the top p-percentile
    long_dict = total_return(snp_stocks_data, top_p_momentum_stocks_dict, returns_df, p)
    long = pd.DataFrame.from_dict(long_dict, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long

# Go long on top p-percentile and sort on bottom p-percentile
def momentum_long_sort(snp_stocks_data, momentum_df, returns_df, p_long, p_sort, strategy, strategy_returns):
    # Stocks in the top p-percentile 
    top_p_momentum_stocks_dict = top_p_percentage_stocks(momentum_df, p_long)

    # Stocks in the bottom p-percentile
    bottom_p_momentum_stocks_dict = bottom_p_percentage_stocks(momentum_df, p_sort)
    
    # Retuens from longing as well as sorting at the same time
    sort_long_dict = portfolio_return(returns_df, top_p_momentum_stocks_dict, bottom_p_momentum_stocks_dict)
    long = pd.DataFrame.from_dict(sort_long_dict, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long

'''----------------------12-1 Momentum------------------------'''
def momentum_12_1(snp_stocks_data, momentum_12_1_df, returns_df, p, strategy_returns):
    # Stocks from the top p-percentile
    top_p_momentum_stocks_dict_12_1 = top_p_percentage_stocks(momentum_12_1_df, p)

    # Returns from the top p-percentile
    long_dict_12_1 = total_return(snp_stocks_data, top_p_momentum_stocks_dict_12_1, returns_df)
    long = pd.DataFrame.from_dict(long_dict_12_1, orient='index')
    long.rename({0:f'12_1_momentum_{p}'}, axis='columns', inplace=True)
    strategy_returns[f'12_1_momentum_{p}'] = long


def momentum_12_1_ls(snp_stocks_data, momentum_12_1_df, returns_df, p, strategy_returns):
    # Stocks from the top p-percentile
    top_p_momentum_stocks_dict_12_1 = top_p_percentage_stocks(momentum_12_1_df, p)
    # Stocks in the bottom p-percentile
    bottom_p_momentum_stocks_dict_12_1 = bottom_p_percentage_stocks(momentum_12_1_df, p)
    # Returns from longing as well as sorting at the same time
    sort_long_dict_12_1 = portfolio_return(returns_df, top_p_momentum_stocks_dict_12_1, bottom_p_momentum_stocks_dict_12_1)
    long = pd.DataFrame.from_dict(sort_long_dict_12_1, orient='index')
    
    long.rename({0:f'12_1_momentum_ls_{p}'}, axis='columns', inplace=True)
    strategy_returns[f'12_1_momentum_ls_{p}'] = long


'''-----------------------REVERSION----------------------------------------'''
def reversion_long(snp_stocks_data, momentum, returns_df,  p, strategy, strategy_returns):
    # Stocks in the top p-percentile 
    top_p_momentum_stocks_dict_inversion = bottom_p_percentage_stocks(momentum, p)

    # Returns from longing the top p-percentile
    long_dict_reversion = total_return(snp_stocks_data, top_p_momentum_stocks_dict_inversion, returns_df)
    long = pd.DataFrame.from_dict(long_dict_reversion, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long

def reversion_long_sort(snp_stocks_data, momentum, returns_df, p_long, p_sort, strategy, strategy_returns):
    # Stocks in the top p-percentile 
    top_p_momentum_stocks_dict_inversion = bottom_p_percentage_stocks(momentum, p_long)

    # Stocks in the bottom p-percentile
    bottom_p_momentum_stocks_dict_inversion = top_p_percentage_stocks(momentum, p_sort)

    # Retuens from longing as well as sorting at the same time
    sort_long_dict_reversion = portfolio_return(returns_df, top_p_momentum_stocks_dict_inversion, bottom_p_momentum_stocks_dict_inversion)
    long = pd.DataFrame.from_dict(sort_long_dict_reversion, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long


'''---------------------------VALUE STRATEGY------------------------------------'''
def price_strategy(stocks_data, returns_df, p, strategy, strategy_returns):
    x = value_strategy(stocks_data, p)
    m = total_return(stocks_data, x, returns_df)
    long = pd.DataFrame.from_dict(m, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long


def price_strategy_ls(stocks_data, returns_df, p_buy, p_sell, strategy, strategy_returns):
    x, y = value_strategy_ls(stocks_data, p_buy, p_sell)
    m = portfolio_return(returns_df, x, y)
    long = pd.DataFrame.from_dict(m, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long


'''-------------------------------SEASONALITY--------------------------------------------'''
def seasonality(stocks_data, return_df, strategy, strategy_returns,rebalance, y=1, m=1, n=1):
    result = calculate_return_signals(return_df, y, rebalance)
    buy_dict, sell_dict = generate_buy_sell_signals(result, m, n)
    buy_return = total_return(stocks_data, buy_dict, return_df)
    long = pd.DataFrame.from_dict(buy_return, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long


def seasonality_ls(stocks_data, return_df, strategy, strategy_returns, rebalance, y=1, m=1, n=1):
    result = calculate_return_signals(return_df, y, rebalance)
    buy_dict, sell_dict = generate_buy_sell_signals(result, m, n)
    buy_sell_return = portfolio_return(return_df, buy_dict, sell_dict)
    long = pd.DataFrame.from_dict(buy_sell_return, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long


'''-------------------------VOLATILITY-------------------------------------'''
def volatility_strategy(stocks_data, returns_df, p, strategy, strategy_returns, y=1 ):
    vol_df = calculate_vol(stocks_data, y)
    vol_df = vol_df.dropna(axis=1, how='all').T
    top = bottom_p_percentage_stocks(vol_df, p)
    long_dict_reversion = total_return(stocks_data, top, returns_df)
    long = pd.DataFrame.from_dict(long_dict_reversion, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long

def volatility_strategy_ls(stocks_data, returns_df,  p, strategy, strategy_returns,  y=1,):
    vol_df = calculate_vol(stocks_data, y).T
    vol_df = vol_df.dropna(axis=1, how='all')
    top = bottom_p_percentage_stocks(vol_df, p)
    btm = top_p_percentage_stocks(vol_df, p)
    sort_long_dict_reversion = portfolio_return(returns_df, top, btm)
    long = pd.DataFrame.from_dict(sort_long_dict_reversion, orient='index')
    long.rename({0:strategy}, axis='columns', inplace=True)
    strategy_returns[strategy] = long