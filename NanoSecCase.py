import pandas as pd
from pathlib import Path
import matplotlib
import numpy as np
from datetime import datetime, timedelta
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from typing import Tuple, Dict, Optional

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1) File Paths
BASE_DIR = Path(r"C:\Users\Vusal Ibrahimli\OneDrive\Desktop\NanoSec_Case")
SPOT_PATH = BASE_DIR / "trb_usdt_spot_export.csv"
PERP_PATH = BASE_DIR / "trb_usdt_futures_export.csv"
TRADE_PATH = BASE_DIR / "trb_usdt_trades_export.csv"

# 2) Data Loading & Preprocessing
def load_orderbook(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
    df['ms'] = (df['time'] - df['time'].min()).dt.total_seconds() * 1000
    return df

def load_trades(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(path)
    trades['time'] = pd.to_datetime(trades['time'], errors='coerce')
    trades = trades.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    trades['ms'] = (trades['time'] - trades['time'].min()).dt.total_seconds() * 1000

    trades_agg = trades.groupby('time').agg({
        'quantity': 'sum',
        'price': 'mean'
    }).reset_index()
    trades_agg['ms'] = (trades_agg['time'] - trades_agg['time'].min()).dt.total_seconds() * 1000

    return trades, trades_agg

spot, perp, trades, trades_agg = None, None, None, None
spot = load_orderbook(SPOT_PATH)
perp = load_orderbook(PERP_PATH)
trades, trades_agg = load_trades(TRADE_PATH)

print(f"Data loaded - Spot: {len(spot):,} records, Perp: {len(perp):,} records, Trades: {len(trades):,} records")

# 3) Exploratory Data Analysis
def plot_price_series(df: pd.DataFrame, label: str):
    plt.figure(figsize=(12, 5))
    plt.plot(df['time'], df['bid_price'], label=f'{label} Bid', alpha=0.8)
    plt.plot(df['time'], df['ask_price'], label=f'{label} Ask', alpha=0.8)
    plt.title(f'{label} Bid & Ask Prices')
    plt.xlabel('Time'); plt.ylabel('Price'); plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout(); plt.show()

def plot_comparison(a: pd.DataFrame, b: pd.DataFrame, price_col: str, label_a: str, label_b: str):
    plt.figure(figsize=(12, 5))
    plt.plot(a['time'], a[price_col], label=label_a, alpha=0.7)
    plt.plot(b['time'], b[price_col], label=label_b, alpha=0.7)
    plt.title(f'{label_a} vs {label_b} ({price_col})')
    plt.xlabel('Time'); plt.ylabel(price_col.replace('_', ' ').title()); plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tight_layout(); plt.show()

plot_price_series(spot, 'Spot')
plot_price_series(perp, 'Perp')
plot_comparison(spot, perp, 'mid_price', 'Spot', 'Perp')

# 4) Mid-Price Analysis
def analyze_price_relationship(spot_df: pd.DataFrame, perp_df: pd.DataFrame) -> pd.DataFrame:
    spot_mid = spot_df[['time','mid_price']].rename(columns={'mid_price':'spot_mid'})
    perp_mid = perp_df[['time','mid_price']].rename(columns={'mid_price':'perp_mid'})
    merged = pd.merge_asof(
        spot_mid.sort_values('time'),
        perp_mid.sort_values('time'),
        on='time', direction='nearest', tolerance=pd.Timedelta('5ms')
    ).dropna()
    merged['delta_mid'] = merged['perp_mid'] - merged['spot_mid']
    merged['delta_pct'] = (merged['delta_mid']/merged['spot_mid'])*10000
    return merged

merged = analyze_price_relationship(spot, perp)
plt.figure(figsize=(12,5))
plt.plot(merged['time'], merged['delta_mid'], alpha=0.7)
plt.title('Perp – Spot Mid-Price Delta Over Time')
plt.xlabel('Time'); plt.ylabel('Delta (perp_mid – spot_mid)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.tight_layout(); plt.show()

print(f"Price Delta Stats - Mean: {merged['delta_mid'].mean():.4f}, Std: {merged['delta_mid'].std():.4f}")
print(f"Price Delta (bps) - Mean: {merged['delta_pct'].mean():.2f}, Std: {merged['delta_pct'].std():.2f}")

# 5) Lead-Lag Analysis
def perform_lead_lag_analysis(merged_df: pd.DataFrame, max_lag: int=50) -> Dict:
    s = merged_df['spot_mid'] - merged_df['spot_mid'].mean()
    p = merged_df['perp_mid'] - merged_df['perp_mid'].mean()
    ccfs = [s.corr(p.shift(l)) for l in range(-max_lag, max_lag+1)]
    lags = np.arange(-max_lag, max_lag+1)
    max_corr_idx = np.argmax(np.abs(ccfs))
    optimal_lag, max_corr = lags[max_corr_idx], ccfs[max_corr_idx]

    plt.figure(figsize=(10,4))
    plt.stem(lags, ccfs)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(optimal_lag, color='red', linestyle='--', alpha=0.7,
                label=f'Max at lag {optimal_lag}')
    plt.title('Spot ↔ Perp Mid-Price Cross-Correlation')
    plt.xlabel('Lag (data points)'); plt.ylabel('Correlation'); plt.legend()
    plt.tight_layout(); plt.show()

    df_reg = merged_df.copy()
    for i in range(1,6):
        df_reg[f'spot_lag{i}'] = df_reg['spot_mid'].shift(i)
        df_reg[f'perp_lag{i}'] = df_reg['perp_mid'].shift(i)
    df_reg = df_reg.dropna()

    X_s2p = sm.add_constant(df_reg[[f'spot_lag{i}' for i in range(1,6)]])
    y_s2p = df_reg['perp_mid']
    model_s2p = sm.OLS(y_s2p, X_s2p).fit()

    X_p2s = sm.add_constant(df_reg[[f'perp_lag{i}' for i in range(1,6)]])
    y_p2s = df_reg['spot_mid']
    model_p2s = sm.OLS(y_p2s, X_p2s).fit()

    return {
        'optimal_lag': optimal_lag,
        'max_correlation': max_corr,
        'spot_to_perp_r2': model_s2p.rsquared,
        'perp_to_spot_r2': model_p2s.rsquared,
        'ccf_values': ccfs,
        'lags': lags
    }

lead_lag_results = perform_lead_lag_analysis(merged)
print(f"Lead-Lag Analysis Results:")
print(f"Optimal Lag: {lead_lag_results['optimal_lag']} (negative = spot leads)")
print(f"Max Correlation: {lead_lag_results['max_correlation']:.4f}")
print(f"Spot → Perp R²: {lead_lag_results['spot_to_perp_r2']:.4f}")
print(f"Perp → Spot R²: {lead_lag_results['perp_to_spot_r2']:.4f}")

# 6) Sudden-Move Statistics
def compute_window_stats(df: pd.DataFrame, windows_ms: list) -> pd.DataFrame:
    records = []; base = df[['ms','mid_price']].copy()
    for w in windows_ms:
        prev = base.copy(); prev['ms'] += w
        temp = pd.merge_asof(
            df.sort_values('ms'), prev.sort_values('ms'),
            on='ms', direction='backward', suffixes=('','_prev')
        ).dropna(subset=['mid_price_prev'])
        temp['pct_change_bps'] = (temp['mid_price'] - temp['mid_price_prev'])/temp['mid_price_prev']*10000
        a = temp['pct_change_bps'].abs()
        records.append({
            'window_ms': w,
            'samples': len(a),
            'p95_bps': a.quantile(0.95),
            'p99_bps': a.quantile(0.99),
            'mean_bps': a.mean(),
            'std_bps': a.std(),
            'max_bps': a.max()
        })
    return pd.DataFrame(records)

print("\n=== Computing Window Statistics ===")
windows = [1,2,3,5,10,20,50]
spot_stats = compute_window_stats(spot, windows)
perp_stats = compute_window_stats(perp, windows)

print("Spot Market Statistics:"); print(spot_stats.to_string(index=False))
print("\nPerp Market Statistics:"); print(perp_stats.to_string(index=False))

# 7) Signal Parameters
WINDOW_MS = 3
THRESHOLD_BPS = 5
SIGNAL_FWD_MS = 5
VOLUME_WINDOW_MS = 50
VOLUME_THRESHOLD = trades_agg['quantity'].quantile(0.95)

print(f"Signal Parameters:")
print(f"Window: {WINDOW_MS}ms, Threshold: {THRESHOLD_BPS}bps, Forward: {SIGNAL_FWD_MS}ms")
print(f"Volume Threshold: {VOLUME_THRESHOLD:.2f}")

# 8) Sudden Move Detection
def detect_sudden_moves(df: pd.DataFrame, window_ms: int, threshold_bps: float) -> pd.DataFrame:
    prev_data = df[['ms','mid_price','bid_price','ask_price']].copy()
    prev_data['ms'] += window_ms
    merged = pd.merge_asof(
        df.sort_values('ms'),
        prev_data.sort_values('ms'),
        on='ms', direction='backward',
        tolerance=window_ms*2, suffixes=('','_prev')
    ).dropna(subset=['mid_price_prev'])
    merged['pct_change_bps'] = (merged['mid_price']-merged['mid_price_prev'])/merged['mid_price_prev']*10000
    merged['bid_change_bps'] = (merged['bid_price']-merged['bid_price_prev'])/merged['bid_price_prev']*10000
    merged['ask_change_bps'] = (merged['ask_price']-merged['ask_price_prev'])/merged['ask_price_prev']*10000
    sudden_moves = merged[np.abs(merged['pct_change_bps'])>=threshold_bps].copy()
    sudden_moves['direction'] = np.where(sudden_moves['pct_change_bps']>0,'up','down')
    sudden_moves['leading_side'] = np.where(
        np.abs(sudden_moves['ask_change_bps'])>np.abs(sudden_moves['bid_change_bps']),'ask','bid'
    )
    return sudden_moves[['time','ms','mid_price','bid_price','ask_price',
                         'pct_change_bps','bid_change_bps','ask_change_bps',
                         'direction','leading_side']]

spot_moves = detect_sudden_moves(spot, WINDOW_MS, THRESHOLD_BPS)
perp_moves = detect_sudden_moves(perp, WINDOW_MS, THRESHOLD_BPS)

print(f"Sudden Moves Detected:")
print(f"Spot: {len(spot_moves):,} moves ({len(spot_moves)/len(spot)*100:.2f}%)")
print(f"Perp: {len(perp_moves):,} moves ({len(perp_moves)/len(perp)*100:.2f}%)")

# 9) Enhanced Signal Generation
def generate_enhanced_signals(moves_df: pd.DataFrame, target_market: pd.DataFrame,
                              forward_window_ms: int, threshold_bps: float) -> pd.DataFrame:
    moves = moves_df.copy(); moves['ms_future'] = moves['ms']+forward_window_ms
    target_data = target_market[['ms','mid_price','bid_price','ask_price']].rename(columns={
        'mid_price':'future_mid_price',
        'bid_price':'future_bid_price',
        'ask_price':'future_ask_price'
    }).sort_values('ms')
    signals = pd.merge_asof(
        moves.sort_values('ms_future'),
        target_data,
        left_on='ms_future', right_on='ms',
        direction='forward', tolerance=forward_window_ms
    ).dropna(subset=['future_mid_price'])
    signals['future_pct_change_bps'] = (
        (signals['future_mid_price']-signals['mid_price'])/signals['mid_price']*10000
    )
    signals['price_signal'] = (
        (np.abs(signals['future_pct_change_bps'])>=threshold_bps) &
        (np.sign(signals['future_pct_change_bps'])==np.sign(signals['pct_change_bps']))
    ).astype(int)
    signals['prediction'] = signals['future_mid_price']
    signals.rename(columns={'ms_x':'ms'}, inplace=True)
    return signals

spot2perp_signals = generate_enhanced_signals(spot_moves, perp, SIGNAL_FWD_MS, THRESHOLD_BPS)
perp2spot_signals = generate_enhanced_signals(perp_moves, spot, SIGNAL_FWD_MS, THRESHOLD_BPS)

print(f"Raw Signal Generation:")
print(f"Spot → Perp: {spot2perp_signals['price_signal'].sum():,} signals from {len(spot2perp_signals):,} moves")
print(f"Perp → Spot: {perp2spot_signals['price_signal'].sum():,} signals from {len(perp2spot_signals):,} moves")

# 10) Optimized Volume Confirmation
def add_volume_confirmation_vectorized(signals_df: pd.DataFrame, trades_df: pd.DataFrame,
                                       volume_window_ms: int, volume_threshold: float) -> pd.DataFrame:
    signals = signals_df.copy()
    if 'ms' not in signals.columns:
        signals['ms'] = (signals['time']-signals['time'].min()).dt.total_seconds()*1000
    half_window = volume_window_ms/2
    trades_df['volume_bin'] = ((trades_df['ms']+half_window)//volume_window_ms).astype(int)
    volume_by_bin = trades_df.groupby('volume_bin')['quantity'].sum().reset_index()
    signals['volume_bin'] = ((signals['ms']+half_window)//volume_window_ms).astype(int)
    signals = signals.merge(volume_by_bin, on='volume_bin', how='left')
    signals['quantity'] = signals['quantity'].fillna(0)
    signals['volume_confirmed'] = (signals['quantity']>=volume_threshold).astype(int)
    signals['final_signal'] = signals['price_signal'] * signals['volume_confirmed']
    return signals

spot2perp_final = add_volume_confirmation_vectorized(spot2perp_signals, trades_agg, VOLUME_WINDOW_MS, VOLUME_THRESHOLD)
perp2spot_final = add_volume_confirmation_vectorized(perp2spot_signals, trades_agg, VOLUME_WINDOW_MS, VOLUME_THRESHOLD)

# 11) Trade Signal Assignment
def assign_trade_signals(signals_df: pd.DataFrame, signal_type: str) -> pd.DataFrame:
    signals = signals_df.copy()
    if signal_type=='spot2perp':
        signals['trade_action'] = np.where(
            (signals['final_signal']==1)&(signals['direction']=='up'),'LONG',
            np.where((signals['final_signal']==1)&(signals['direction']=='down'),'SHORT','')
        )
    else:
        signals['trade_action'] = np.where(
            (signals['final_signal']==1)&(signals['direction']=='up'),'BUY',
            np.where((signals['final_signal']==1)&(signals['direction']=='down'),'SELL','')
        )
    return signals

spot2perp_final = assign_trade_signals(spot2perp_final, 'spot2perp')
perp2spot_final = assign_trade_signals(perp2spot_final, 'perp2spot')

# 12) Comprehensive Signal Reporting
def report_comprehensive_signals(signals_df: pd.DataFrame, signal_name: str) -> Dict:
    total_moves = len(signals_df)
    price_signals = signals_df['price_signal'].sum()
    final_signals = signals_df['final_signal'].sum()
    print(f"\n=== {signal_name} Signal Analysis ===")
    print(f"Total Sudden Moves: {total_moves:,}")
    print(f"Price-Valid Signals: {price_signals:,} ({price_signals/total_moves*100:.1f}%)")
    print(f"Volume-Confirmed Signals: {final_signals:,} ({final_signals/total_moves*100:.1f}%)")
    print(f"Volume Filter Rate: {final_signals/price_signals*100:.1f}% of price signals")
    active = signals_df[signals_df['final_signal']==1]
    if not active.empty:
        print(f"\nSample Signals:")
        cols = ['time','direction','trade_action','pct_change_bps','future_pct_change_bps','quantity']
        print(active[cols].head(10).to_string(index=False))
        stats = active.groupby('direction').agg({
            'final_signal':'count',
            'future_pct_change_bps':'mean',
            'leading_side':lambda x: x.mode().iloc[0]
        }).round(2)
        print(f"\nDirection Analysis:"); print(stats.to_string())
    return {
        'total_moves': total_moves,
        'price_signals': price_signals,
        'final_signals': final_signals,
        'signal_rate': final_signals/total_moves if total_moves>0 else 0
    }

spot2perp_stats = report_comprehensive_signals(spot2perp_final, 'Spot → Perp')
perp2spot_stats = report_comprehensive_signals(perp2spot_final, 'Perp → Spot')

# 13) Signal Distribution Over Time
def plot_signal_distribution(signals_df: pd.DataFrame, title: str, color='blue'):
    active = signals_df[signals_df['final_signal']==1]
    if active.empty:
        print(f"No active signals for {title}")
        return
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,10))
    ax1.scatter(active['time'],active['mid_price'],
                c=active['direction'].map({'up':'green','down':'red'}),
                alpha=0.7,s=30)
    ax1.set_title(f'{title} - Signal Timing and Direction')
    ax1.set_ylabel('Mid Price')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.grid(True,alpha=0.3)
    active['hour']=active['time'].dt.hour
    counts=active.groupby('hour').size()
    ax2.bar(counts.index,counts.values,color=color,alpha=0.7)
    ax2.set_title(f'{title} - Hourly Signal Frequency')
    ax2.set_xlabel('Hour of Day'); ax2.set_ylabel('Number of Signals'); ax2.grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()

plot_signal_distribution(spot2perp_final, "Spot → Perp Signals", 'blue')
plot_signal_distribution(perp2spot_final, "Perp → Spot Signals", 'red')

# 14) Market Leadership Analysis
def analyze_market_leadership(spot_stats: Dict, perp_stats: Dict, lead_lag: Dict) -> str:
    print(f"\n=== Market Leadership Analysis ===")
    print(f"Spot → Perp Signal Rate: {spot_stats['signal_rate']*100:.2f}% ({spot_stats['final_signals']} signals)")
    print(f"Perp → Spot Signal Rate: {perp_stats['signal_rate']*100:.2f}% ({perp_stats['final_signals']} signals)")
    lag = lead_lag['optimal_lag']
    if lag<0: print(f"Spot leads by {abs(lag)} data points")
    elif lag>0: print(f"Perp leads by {lag} data points")
    else: print("Simultaneous movement")
    print(f"Predictive Power: Spot→Perp R²={lead_lag['spot_to_perp_r2']:.4f}, Perp→Spot R²={lead_lag['perp_to_spot_r2']:.4f}")
    if spot_stats['signal_rate']>perp_stats['signal_rate'] and lead_lag['spot_to_perp_r2']>lead_lag['perp_to_spot_r2']:
        return "SPOT"
    elif perp_stats['signal_rate']>spot_stats['signal_rate'] and lead_lag['perp_to_spot_r2']>lead_lag['spot_to_perp_r2']:
        return "PERP"
    else:
        return "MIXED"

market_leader = analyze_market_leadership(spot2perp_stats, perp2spot_stats, lead_lag_results)
print(f"\nOverall Price Discovery Leader: {market_leader}")

# 15) Performance Metrics & Backtesting
def calculate_performance_metrics(signals_df: pd.DataFrame, trades_df: pd.DataFrame):
    active = signals_df[signals_df['final_signal']==1].sort_values('time').reset_index(drop=True)
    trades_for_merge = trades_df[['time','price']].dropna().sort_values('time')
    merged = pd.merge_asof(active, trades_for_merge, on='time', direction='forward').dropna(subset=['price'])
    if merged.empty:
        return {'error':'No signals matched to trades'}, None

    merged['entry_price'] = merged['mid_price']
    merged['exit_price']  = merged['price']
    merged['return'] = np.where(
        merged['direction']=='up',
        (merged['exit_price']-merged['entry_price'])/merged['entry_price'],
        (merged['entry_price']-merged['exit_price'])/merged['entry_price']
    )

    returns = merged['return']
    cumret = (1+returns).cumprod()
    drawdown = (cumret.cummax()-cumret)/cumret.cummax()

    metrics = {
        'total_signals': len(merged),
        'avg_return': returns.mean(),
        'std_return': returns.std(),
        'sharpe_ratio': (returns.mean()/returns.std())*np.sqrt(252*24*3600) if returns.std()>0 else 0,
        'max_drawdown': drawdown.max(),
        'hit_rate': (returns>0).mean(),
        'profit_factor': returns[returns>0].sum()/abs(returns[returns<0].sum()) if returns[returns<0].sum()<0 else np.inf,
        'total_return': cumret.iloc[-1]-1,
        'best_return': returns.max(),
        'worst_return': returns.min()
    }
    return metrics, merged

print(f"\n=== Performance Analysis ===")
spot2perp_perf, spot2perp_trades = calculate_performance_metrics(spot2perp_final, trades)
perp2spot_perf, perp2spot_trades = calculate_performance_metrics(perp2spot_final, trades)

def print_performance_metrics(metrics: Dict, name: str):
    if 'error' in metrics:
        print(f"{name}: {metrics['error']}")
        return
    print(f"\n{name} Performance:")
    print(f"Total Signals: {metrics['total_signals']:,}")
    print(f"Average Return: {metrics['avg_return']:.4f} ({metrics['avg_return']*100:.2f}%)")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Hit Rate: {metrics['hit_rate']:.2f} ({metrics['hit_rate']*100:.1f}%)")
    print(f"Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
    print(f"Best/Worst: {metrics['best_return']:.4f} / {metrics['worst_return']:.4f}")

print_performance_metrics(spot2perp_perf, "Spot → Perp Strategy")
print_performance_metrics(perp2spot_perf, "Perp → Spot Strategy")


# 16) Signal Performance Analysis
def plot_signal_performance(signals_df, trades_df, title):
    """Plot signal performance metrics"""
    active_signals = signals_df[signals_df['final_signal'] == 1].copy()
    if active_signals.empty:
        print(f"No active signals for {title}")
        return

    # Calculate returns for plotting
    trades_for_merge = trades_df[['time', 'price']].dropna().sort_values('time')
    signals_with_prices = pd.merge_asof(
        active_signals.sort_values('time'),
        trades_for_merge,
        on='time',
        direction='forward'
    ).dropna(subset=['price'])

    if signals_with_prices.empty:
        print(f"No signals matched with prices for {title}")
        return

    # Calculate returns
    signals_with_prices['return'] = np.where(
        signals_with_prices['direction'] == 'up',
        (signals_with_prices['price'] - signals_with_prices['mid_price']) / signals_with_prices['mid_price'],
        (signals_with_prices['mid_price'] - signals_with_prices['price']) / signals_with_prices['mid_price']
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Returns distribution
    ax1.hist(signals_with_prices['return'], bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(signals_with_prices['return'].mean(), color='red', linestyle='--',
                label=f'Mean: {signals_with_prices["return"].mean():.4f}')
    ax1.set_title(f'{title} - Return Distribution')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative returns
    cumulative_returns = (1 + signals_with_prices['return']).cumprod()
    ax2.plot(range(len(cumulative_returns)), cumulative_returns, linewidth=2)
    ax2.set_title(f'{title} - Cumulative Returns')
    ax2.set_xlabel('Signal Number')
    ax2.set_ylabel('Cumulative Return')
    ax2.grid(True, alpha=0.3)

    # Returns by direction
    direction_returns = signals_with_prices.groupby('direction')['return'].mean()
    ax3.bar(direction_returns.index, direction_returns.values, alpha=0.7)
    ax3.set_title(f'{title} - Returns by Direction')
    ax3.set_ylabel('Average Return')
    ax3.grid(True, alpha=0.3)

    # Hit rate by hour
    signals_with_prices['hour'] = signals_with_prices['time'].dt.hour
    signals_with_prices['hit'] = signals_with_prices['return'] > 0
    hourly_hit_rate = signals_with_prices.groupby('hour')['hit'].mean()
    ax4.plot(hourly_hit_rate.index, hourly_hit_rate.values, marker='o', linewidth=2)
    ax4.axhline(0.5, linestyle='--', alpha=0.5)
    ax4.set_title(f'{title} - Hit Rate by Hour')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Hit Rate')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return signals_with_prices


# 17) Market Microstructure Analysis
def plot_microstructure_analysis(spot_signals, perp_signals):
    """Analyze market microstructure patterns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Leading side analysis - Spot
    spot_active = spot_signals[spot_signals['final_signal'] == 1]
    if not spot_active.empty:
        leading_side_spot = spot_active['leading_side'].value_counts()
        ax1.pie(leading_side_spot.values, labels=leading_side_spot.index, autopct='%1.1f%%')
        ax1.set_title('Spot Signals - Leading Side')

    # Leading side analysis - Perp
    perp_active = perp_signals[perp_signals['final_signal'] == 1]
    if not perp_active.empty:
        leading_side_perp = perp_active['leading_side'].value_counts()
        ax2.pie(leading_side_perp.values, labels=leading_side_perp.index, autopct='%1.1f%%')
        ax2.set_title('Perp Signals - Leading Side')

    # Signal strength distribution
    if not spot_active.empty:
        ax3.hist(spot_active['pct_change_bps'], bins=30, alpha=0.7, label='Spot')
    if not perp_active.empty:
        ax3.hist(perp_active['pct_change_bps'], bins=30, alpha=0.7, label='Perp')
    ax3.set_title('Signal Strength Distribution')
    ax3.set_xlabel('Price Change (bps)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Volume vs Signal Success
    if not spot_active.empty and not perp_active.empty:
        combined = pd.concat([
            spot_active[['quantity', 'future_pct_change_bps']].assign(market='Spot'),
            perp_active[['quantity', 'future_pct_change_bps']].assign(market='Perp')
        ])
        combined['volume_bin'] = pd.qcut(combined['quantity'], q=5,
                                         labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
        perf = combined.groupby(['volume_bin', 'market'])['future_pct_change_bps'].mean().unstack()
        perf.plot(kind='bar', ax=ax4, alpha=0.7)
        ax4.set_title('Future Returns by Volume Bins')
        ax4.set_xlabel('Volume Bin')
        ax4.set_ylabel('Average Future Return (bps)')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 18) Advanced Signal Quality Analysis
def analyze_signal_quality(signals_df, title):
    """Analyze signal quality metrics"""
    active = signals_df[signals_df['final_signal'] == 1].copy()
    if active.empty:
        print(f"No active signals for {title}")
        return

    print(f"\n=== {title} Signal Quality Analysis ===")
    active['strength_bin'] = pd.qcut(np.abs(active['pct_change_bps']),
                                     q=5, labels=['Weak','Med-Weak','Medium','Med-Strong','Strong'])
    active['success'] = np.sign(active['pct_change_bps']) == np.sign(active['future_pct_change_bps'])

    quality_stats = active.groupby('strength_bin').agg({
        'success': ['count','mean'],
        'future_pct_change_bps': 'mean',
        'quantity': 'mean'
    }).round(3)
    print("Signal Quality by Strength:")
    print(quality_stats)

    active['hour'] = active['time'].dt.hour
    hourly = active.groupby('hour').agg({
        'final_signal':'count',
        'success':'mean',
        'future_pct_change_bps':'mean'
    }).round(3)
    print("\nHourly Signal Statistics (Top 5 hours):")
    print(hourly.nlargest(5, 'final_signal'))

    return active



spot_perf = plot_signal_performance(spot2perp_final, trades, "Spot → Perp Strategy")
perp_perf = plot_signal_performance(perp2spot_final, trades, "Perp → Spot Strategy")

plot_microstructure_analysis(spot2perp_final, perp2spot_final)

spot_quality = analyze_signal_quality(spot2perp_final, "Spot → Perp")
perp_quality = analyze_signal_quality(perp2spot_final, "Perp → Spot")



# 19) Momentum Quality Analysis
def analyze_momentum_quality(signals_df: pd.DataFrame, market_df: pd.DataFrame,
                             max_horizon_ms: int = 100) -> pd.DataFrame:
    active = signals_df[signals_df['final_signal'] == 1].copy()
    if active.empty:
        return pd.DataFrame()

    results = []
    for _, sig in active.iterrows():
        t0, dirn, p0 = sig['ms'], 1 if sig['direction']=='up' else -1, sig['mid_price']
        future = market_df[(market_df['ms']>t0)&(market_df['ms']<=t0+max_horizon_ms)].copy()
        if future.empty: continue
        future['dt'] = future['ms']-t0
        future['chg'] = (future['mid_price']-p0)/p0*10000
        dur, max_move, time2peak = 0, 0, 0
        for _, row in future.iterrows():
            aligned = row['chg']*dirn
            if aligned>max_move:
                max_move, time2peak = aligned, row['dt']
            if aligned>0:
                dur = row['dt']
            else:
                break
        peak_idx = future['chg'].abs().idxmax()
        if peak_idx is not None:
            pk_time = future.loc[peak_idx,'dt']; pk_move = future.loc[peak_idx,'chg']
            post = future[future['dt']>pk_time]
            final = post.iloc[-1]['chg'] if not post.empty else pk_move
            fade = (pk_move - final)/abs(pk_move) if pk_move!=0 else 0
        else:
            pk_time, pk_move, fade = 0, 0, 0

        results.append({
            'signal_time': sig['time'],
            'direction': sig['direction'],
            'momentum_duration_ms': dur,
            'max_favorable_move_bps': max_move,
            'time_to_peak_ms': time2peak,
            'peak_move_bps': pk_move,
            'fade_ratio': fade,
            'momentum_quality': 'High' if (dur>20 and max_move>5) else 'Low'
        })

    return pd.DataFrame(results)

def summarize_momentum_insights(spot_df, perp_df):
    print("\n=== MOMENTUM INSIGHTS SUMMARY ===")
    if not spot_df.empty:
        avg_dur = spot_df['momentum_duration_ms'].mean()
        hq = (spot_df['momentum_quality']=='High').sum()
        print(f"Spot: Avg dur {avg_dur:.1f}ms, {hq} high-quality")
    if not perp_df.empty:
        avg_dur = perp_df['momentum_duration_ms'].mean()
        hq = (perp_df['momentum_quality']=='High').sum()
        print(f"Perp: Avg dur {avg_dur:.1f}ms, {hq} high-quality")

# 20) Noise Detection & Characterization
def detect_and_characterize_noise(signals_df: pd.DataFrame, target_df: pd.DataFrame,
                                  threshold_bps: float=7, reversion_window_ms: int=10) -> pd.DataFrame:
    moves = signals_df.copy()
    analysis = []
    for _, mv in moves.iterrows():
        t0, dirn, chg = mv['ms'], 1 if mv['direction']=='up' else -1, mv['pct_change_bps']
        tgt = target_df[(target_df['ms']>=t0)&(target_df['ms']<=t0+reversion_window_ms)]
        if tgt.empty: continue
        init_price = tgt.iloc[0]['mid_price']
        responses = ((tgt['mid_price']-init_price)/init_price*10000).tolist()
        max_resp, min_resp = max(responses), min(responses)
        sig_resp = (dirn>0 and max_resp>threshold_bps*0.3) or (dirn<0 and min_resp<-threshold_bps*0.3)
        src_w = signals_df[(signals_df['ms']>=t0)&(signals_df['ms']<=t0+reversion_window_ms)]
        quick = False
        if len(src_w)>1:
            for price in src_w['mid_price'].iloc[1:]:
                if abs((price-mv['mid_price'])/mv['mid_price']*10000) < threshold_bps*0.5:
                    quick = True; break
        if not sig_resp and quick:
            ntype = "Pure Noise"
        elif not sig_resp:
            ntype = "Target Unresponsive"
        elif quick:
            ntype = "Source Reversion"
        else:
            ntype = "Valid Signal"
        analysis.append({
            'signal_time': mv['time'],
            'direction': mv['direction'],
            'noise_type': ntype,
            'is_noise': ntype!="Valid Signal"
        })
    return pd.DataFrame(analysis)

def analyze_noise_characteristics(noise_df: pd.DataFrame, title: str):
    if noise_df.empty:
        print(f"No noise data for {title}")
        return
    total = len(noise_df)
    noise_cnt = noise_df['is_noise'].sum()
    print(f"\n=== {title} NOISE ANALYSIS ===")
    print(f"Total Moves: {total}, Noise Events: {noise_cnt} ({noise_cnt/total*100:.1f}%)")
    breakdown = noise_df['noise_type'].value_counts()
    print("Noise Type Breakdown:")
    for t, c in breakdown.items():
        print(f"  {t}: {c} ({c/total*100:.1f}%)")
    return breakdown

def compare_noise_across_markets(spot_noise, perp_noise):
    print("\n=== CROSS-MARKET NOISE COMPARISON ===")
    sr, pr = spot_noise['is_noise'].mean(), perp_noise['is_noise'].mean()
    print(f"Spot Noise Rate: {sr*100:.1f}%, Perp Noise Rate: {pr*100:.1f}%")
    return pd.DataFrame({
        'Spot': spot_noise['noise_type'].value_counts(normalize=True),
        'Perp': perp_noise['noise_type'].value_counts(normalize=True)
    }).round(3)

def provide_noise_insights(spot_df, perp_df):
    print("\n=== NOISE INSIGHTS & RECOMMENDATIONS ===")
    if not spot_df.empty:
        pn = (spot_df['noise_type']=='Pure Noise').sum()
        print(f"Spot Pure Noise: {pn}/{len(spot_df)} ({pn/len(spot_df)*100:.1f}%)")
    if not perp_df.empty:
        pn = (perp_df['noise_type']=='Pure Noise').sum()
        print(f"Perp Pure Noise: {pn}/{len(perp_df)} ({pn/len(perp_df)*100:.1f}%)")


spot_mom = analyze_momentum_quality(spot2perp_final, perp)
perp_mom = analyze_momentum_quality(perp2spot_final, spot)
summarize_momentum_insights(spot_mom, perp_mom)

spot_noise = detect_and_characterize_noise(spot2perp_final, perp)
perp_noise = detect_and_characterize_noise(perp2spot_final, spot)
analyze_noise_characteristics(spot_noise, "Spot → Perp")
analyze_noise_characteristics(perp_noise, "Perp → Spot")
compare_noise_across_markets(spot_noise, perp_noise)
provide_noise_insights(spot_noise, perp_noise)
