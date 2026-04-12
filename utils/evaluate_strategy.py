import numpy as np
import pandas as pd

def max_drawdown(pct_series):
    cumprod = (1 + pct_series).cumprod()
    # Get the largest percentage change 
    # from peak to subsequent trough
    window = 365
    rolling_peak = cumprod.rolling(window, min_periods=1).max()
    drawdown = (cumprod - rolling_peak) / rolling_peak
    return drawdown.min()

def longest_drawdown(pct_series):
    cumprod = (1 + pct_series).cumprod()
    # Get the numerical index difference 
    # from a peak to the number of rows
    # it took to reach peak again.
    window = 365
    rolling_peak = cumprod.rolling(window, min_periods=1).max()
    drawdown = (cumprod - rolling_peak) / rolling_peak
    
    longest = 0
    last_zero_idx = 0
    
    # loop through, and keep adding until
    for i, v in enumerate(drawdown):
        if v == 0:
            last_zero_idx = i
        else:
            current_length = i - last_zero_idx
            if current_length > longest:
                longest = current_length
    
    return longest
    
            
    
def evaluate_strategy(daily_pct_returns, benchmark_returns=None):
    cumprod = (1 + daily_pct_returns).cumprod()
    
    sharpe = (cumprod.iloc[-1]-1) / cumprod.std()
    
    longest_dd = longest_drawdown(daily_pct_returns)
    
    days = len(daily_pct_returns)
    ann_sharpe = daily_pct_returns.iloc[-1] / daily_pct_returns.std() * np.sqrt(365 / days)
    
    
    print("             PERFORMANCE             ")
    print("-------------------------------------")
    print(f"Final Return                : {((1 + daily_pct_returns).prod() - 1):.2%}")
    print(f"Avg Daily Return            : {daily_pct_returns.mean():.2%}")
    print(f"Median Daily Return         : {daily_pct_returns.median():.2%}")
    print(f"Stddev Return               : {daily_pct_returns.std():.2%}")
    print(f"Annualized Return           : {(1+daily_pct_returns.mean()) ** 365 - 1:.2%}")
    # print(f"Sharpe Ratio                : {sharpe:.2f}")
    print(f"Annualized Sharpe Ratio     : {(daily_pct_returns.mean()*365)/(daily_pct_returns.std() * np.sqrt(365)):.2f}")
    print(f"Max Drawdown                : {max_drawdown(daily_pct_returns):.2%}")
    print(f"Longest Drawdown (days)     : {longest_dd:,.0f}")
    print(f"Win Rate                    : {np.sum(daily_pct_returns>0) / np.sum(daily_pct_returns != 0):.2%}")
    print(f"Skewness                    : {daily_pct_returns.skew():.2f}")
    print(f"Kurtosis                    : {daily_pct_returns.kurtosis():.2f}")
    print(f"Days                        : {len(daily_pct_returns)}")
    print(f"Days in Trade               : {np.mean(daily_pct_returns!=0):.2%}")
    # if benchmark_returns is not None: get beta, alpha, and information ratio
    if benchmark_returns is not None:
        cov = np.cov(daily_pct_returns, benchmark_returns)
        beta = cov[0, 1] / cov[1, 1]
        alpha = (daily_pct_returns.mean() - beta * benchmark_returns.mean()) * 365
        active_returns = daily_pct_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(365)
        information_ratio = (active_returns.mean() * 365) / tracking_error if tracking_error != 0 else np.nan

        print(f"Beta                        : {beta:.2f}")
        print(f"Alpha (annualised)          : {alpha:.2%}")
        print(f"Information Ratio (ann.)    : {information_ratio:.2f}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    daily_returns = pd.Series(np.random.normal(0.0005, 0.02, 365*2))  # Simulated 2 years of daily returns
    evaluate_strategy(daily_returns)