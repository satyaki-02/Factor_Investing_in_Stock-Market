# Factor Investing in Stock Market

**This is my summer internship project on factor investing in the stock market under Shrish Trivedi.**  
In this project, we looked at 5 factors:

1. Momentum
2. Seasonality
3. Reversion
4. Volatility
5. Value

## Machine Learning Algorithms Used

We applied various Machine Learning algorithms for both classification and regression tasks:

### Classification Models
- Random Forest
- Logistic Regression
- Neural Network Classifier

### Regression Models
- Linear Regression
- Random Forest Regression
- Neural Network

## Portfolio Strategy

Based on the models, we created a portfolio under the assumption of **perfect liquidity** in the market and **zero transaction costs**.

The strategy involved:
- **Buying** stocks whose return predictions were positive (1 for classification models and >0 for regression models).
- **Short selling** stocks if the predicted return was negative.

## File Details

Below is a brief description of the main files used in the project:

- `helper.py`: Contains all the helper functions required for basic calculations.
- `factors.py`: Computes the factors and also plots the cumulative returns.
- Files starting with `snp_` or `nse_` refer to code for S&P 500 and Nifty 500 respectively.
- Files ending with `_weekly` or `_monthly` indicate code for weekly and monthly portfolio rebalancing.
