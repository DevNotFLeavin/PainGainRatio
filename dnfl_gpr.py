import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

def apply_smoothing(time_series):
    """Enhanced smoothing method using Savitzky-Golay filter."""
    return pd.Series(
        savgol_filter(
            time_series.fillna(method='ffill').fillna(method='bfill'),
            window_length=21, 
            polyorder=3
        ),
        index=time_series.index
    )

def compute_performance_ratio(returns, window=90):
    """Compute Gain to Performance Ratio"""
    rolling_gains = returns.rolling(window).sum()
    rolling_negative_impact = returns.clip(upper=0).abs().rolling(window).sum()
    return rolling_gains / rolling_negative_impact

def compute_volatility_adjusted_performance(returns, high, low, close, window=90):
    """Compute Volatility-Adjusted Performance Ratio"""
    true_range1 = high - low
    true_range2 = abs(high - close.shift())
    true_range3 = abs(low - close.shift())
    true_range = pd.concat([true_range1, true_range2, true_range3], axis=1).max(axis=1)
    average_true_range = true_range.rolling(window).mean()
    
    # Volatility-adjusted returns
    adjusted_returns = returns / average_true_range
    return compute_performance_ratio(adjusted_returns, window)

def analyze_market_sensitivity(series, market_returns, window=90):
    """
    Analyze market sensitivity metrics.
    """
    upside_sensitivity = pd.Series(index=series.index, dtype=float)
    downside_sensitivity = pd.Series(index=series.index, dtype=float)
    market_independence = pd.Series(index=series.index, dtype=float)
    composite_sensitivity = pd.Series(index=series.index, dtype=float)
    
    for i in range(window, len(series)):
        # Extract windowed data
        y = series.iloc[i-window:i]
        x = market_returns.iloc[i-window:i]
        
        up_market_mask = x > 0
        down_market_mask = x < 0
        
        if sum(up_market_mask) > window//4 and sum(down_market_mask) > window//4:
            # Compute sensitivity during up and down markets
            upside_slope = stats.linregress(x[up_market_mask], y[up_market_mask])[0]
            downside_slope = stats.linregress(x[down_market_mask], y[down_market_mask])[0]
            
            upside_sensitivity.iloc[i] = upside_slope
            downside_sensitivity.iloc[i] = downside_slope
            
            # Calculate market independence and composite sensitivity
            composite_sensitivity.iloc[i] = upside_slope - downside_slope
            market_independence.iloc[i] = 1 - abs(upside_slope * downside_slope)
    
    return {
        'upside_sensitivity': upside_sensitivity,
        'downside_sensitivity': downside_sensitivity,
        'composite_sensitivity': composite_sensitivity,
        'market_independence': market_independence
    }

def comprehensive_asset_analysis(symbol, market_symbol, start_date='2020-01-01', end_date='2025-01-01', window=30):
    """Comprehensive asset performance analysis"""
    # Fetch asset and market data
    asset = yf.Ticker(symbol)
    data = asset.history(start=start_date, end=end_date)
    
    market_benchmark = yf.Ticker(market_symbol)
    market_benchmark_data = market_benchmark.history(start=start_date, end=end_date)
    
    # Calculate returns
    asset_returns = data['Close'].pct_change()
    market_returns = market_benchmark_data['Close'].pct_change()
    
    # Compute Performance Ratios
    performance_ratio = compute_performance_ratio(asset_returns, window)
    volatility_adjusted_ratio = compute_volatility_adjusted_performance(
        asset_returns, 
        data['High'], 
        data['Low'], 
        data['Close'], 
        window
    )
    
    # Analyze market sensitivities
    performance_sensitivity = analyze_market_sensitivity(performance_ratio, market_returns, window)
    volatility_sensitivity = analyze_market_sensitivity(volatility_adjusted_ratio, market_returns, window)
    
    # Apply smoothing to all series
    for measure in [performance_sensitivity, volatility_sensitivity]:
        for key in measure:
            measure[key] = apply_smoothing(measure[key])
    
    return {
        'Performance_Ratio': performance_sensitivity,
        'Volatility_Adjusted_Ratio': volatility_sensitivity,
        'prices': {
            'asset': data['Close'],
            'market': market_benchmark_data['Close']
        }
    }

def visualize_performance_analysis(symbol, results):
    """Create performance visualization"""
    color_palette = {
        'primary': '#2E8B57',  # Sea Green
        'secondary': '#FF6347',  # Tomato
        'highlight': '#4682B4',  # Steel Blue
        'price_line': '#8A2BE2'  # Blueviolet
    }
    
    # Normalize price series
    norm_asset = results['prices']['asset'] / results['prices']['asset'].iloc[0] * 100
    norm_market = results['prices']['market'] / results['prices']['market'].iloc[0] * 100
    
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, figure=fig)
    
    # Performance Ratio Sensitivity
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(results['Performance_Ratio']['upside_sensitivity'], 
             label='Upside Sensitivity', 
             color=color_palette['primary'], alpha=0.7)
    ax1.plot(results['Performance_Ratio']['downside_sensitivity'], 
             label='Downside Sensitivity', 
             color=color_palette['secondary'], alpha=0.7)
    ax1.plot(results['Performance_Ratio']['composite_sensitivity'], 
             label='Composite Sensitivity', 
             color=color_palette['highlight'], linewidth=1.2)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(norm_asset.index, norm_asset, '--', 
                  color=color_palette['price_line'], label=symbol, alpha=0.4)
    
    ax1.set_title(f'{symbol} Performance Ratio Analysis')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Market Independence Comparison
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(results['Performance_Ratio']['market_independence'], 
             label='Performance Independence', 
             color=color_palette['primary'], linewidth=1.2)
    ax2.plot(results['Volatility_Adjusted_Ratio']['market_independence'], 
             label='Volatility-Adjusted Independence', 
             color=color_palette['secondary'], linewidth=1.2)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(norm_asset.index, norm_asset, '--', 
                  color=color_palette['price_line'], label=symbol, alpha=0.4)
    
    ax2.set_title('Market Independence Measures')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Define asset and market symbols
    symbols = ['PEPE24478-USD', 'SOL-USD', 'DOGE-USD']
    market_symbol = 'BTC-USD'
    
    # Iterate through symbols for analysis
    for symbol in symbols:
        try:
            print(f"\nWorking {symbol} ...")
            
            results = comprehensive_asset_analysis(symbol, market_symbol)
            fig = visualize_performance_analysis(symbol, results)
            
            print("\nPerformance Analysis Summary:")
            for measure in ['Performance_Ratio', 'Volatility_Adjusted_Ratio']:
                print(f"\n{measure}:")
                for key in ['upside_sensitivity', 'downside_sensitivity', 'composite_sensitivity', 'market_independence']:
                    mean_val = results[measure][key].mean()
                    print(f"{key}: {mean_val:.3f}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
