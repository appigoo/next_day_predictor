"""
純 Python 技術指標模組
不依賴 TA-Lib 或 pandas-ta,確保 Streamlit Cloud 相容性
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """指數移動平均線"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """簡單移動平均線"""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相對強弱指標 RSI (Wilder's smoothing)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 指標"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真實波幅 ATR"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均趨向指標 ADX"""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=high.index)

    atr_val = atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.fillna(20)


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """布林通道"""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    return upper, mid, lower


def find_support_resistance(df: pd.DataFrame, lookback: int = 60, tolerance: float = 0.01):
    """
    找出支撐與壓力位
    基於近期高低點聚類
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    highs = recent['High'].values
    lows = recent['Low'].values
    closes = recent['Close'].values

    # 找局部極值 (peaks & troughs)
    def find_pivots(arr, window=3):
        pivots = []
        for i in range(window, len(arr) - window):
            if arr[i] == max(arr[i-window:i+window+1]):
                pivots.append(('high', i, arr[i]))
            if arr[i] == min(arr[i-window:i+window+1]):
                pivots.append(('low', i, arr[i]))
        return pivots

    high_pivots = [p[2] for p in find_pivots(highs) if p[0] == 'high']
    low_pivots = [p[2] for p in find_pivots(lows) if p[0] == 'low']

    # 聚類 (相近價位合併)
    def cluster_levels(levels, tol):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for v in levels[1:]:
            if abs(v - clusters[-1][-1]) / clusters[-1][-1] < tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [np.mean(c) for c in clusters]

    current_price = closes[-1]
    resistance_levels = cluster_levels(high_pivots, tolerance)
    support_levels = cluster_levels(low_pivots, tolerance)

    # 只保留當前價之上的壓力、之下的支撐
    resistance = sorted([r for r in resistance_levels if r > current_price])
    support = sorted([s for s in support_levels if s < current_price], reverse=True)

    return {
        'resistance': resistance,
        'support': support,
        'current': current_price
    }


def fibonacci_levels(high: float, low: float) -> dict:
    """計算斐波那契回撤位"""
    diff = high - low
    return {
        '0.0':   low,
        '0.236': low + diff * 0.236,
        '0.382': low + diff * 0.382,
        '0.5':   low + diff * 0.5,
        '0.618': low + diff * 0.618,
        '0.786': low + diff * 0.786,
        '1.0':   high
    }


# ==================== K 線型態識別 ====================

def _body(row):
    return abs(row['Close'] - row['Open'])

def _upper_shadow(row):
    return row['High'] - max(row['Close'], row['Open'])

def _lower_shadow(row):
    return min(row['Close'], row['Open']) - row['Low']

def _is_bullish(row):
    return row['Close'] > row['Open']

def _is_bearish(row):
    return row['Close'] < row['Open']

def _range(row):
    return row['High'] - row['Low']


def detect_candle_patterns(df: pd.DataFrame) -> dict:
    """
    識別主要 K 線型態 (純 Python)
    回傳最近出現的型態列表
    """
    patterns = []
    if len(df) < 5:
        return {'recent': patterns}

    for i in range(max(0, len(df) - 15), len(df)):
        row = df.iloc[i]
        date_str = df.index[i].strftime('%Y-%m-%d %H:%M') \
                    if hasattr(df.index[i], 'strftime') else str(i)
        body = _body(row)
        upper = _upper_shadow(row)
        lower = _lower_shadow(row)
        rng = _range(row)
        if rng == 0:
            continue

        # 1. 錘子線 / 上吊線 (Hammer / Hanging Man)
        if lower > body * 2 and upper < body * 0.3 and body / rng < 0.35:
            # 判斷前序趨勢
            if i >= 5:
                prior_trend = df['Close'].iloc[i-5:i].mean() - df['Close'].iloc[max(0,i-10):i-5].mean() \
                              if i >= 10 else 0
                if prior_trend < 0:
                    patterns.append({
                        'name': '錘子線 Hammer',
                        'type': 'bullish',
                        'date': date_str,
                        'description': '下影線長,實體小,下跌末端見底訊號'
                    })
                else:
                    patterns.append({
                        'name': '上吊線 Hanging Man',
                        'type': 'bearish',
                        'date': date_str,
                        'description': '高位下影長,見頂警告'
                    })

        # 2. 倒錘 / 流星 (Inverted Hammer / Shooting Star)
        elif upper > body * 2 and lower < body * 0.3 and body / rng < 0.35:
            if i >= 5:
                prior_trend = df['Close'].iloc[i-5:i].mean() - df['Close'].iloc[max(0,i-10):i-5].mean() \
                              if i >= 10 else 0
                if prior_trend > 0:
                    patterns.append({
                        'name': '流星 Shooting Star',
                        'type': 'bearish',
                        'date': date_str,
                        'description': '高位上影長,拋壓大,見頂訊號'
                    })
                else:
                    patterns.append({
                        'name': '倒錘 Inverted Hammer',
                        'type': 'bullish',
                        'date': date_str,
                        'description': '下跌末端上影長,反轉訊號'
                    })

        # 3. 十字星 Doji
        if body / rng < 0.1:
            patterns.append({
                'name': '十字星 Doji',
                'type': 'neutral',
                'date': date_str,
                'description': '多空平衡,趨勢可能反轉'
            })

        # 4. 大陽/大陰線 (Marubozu-ish)
        if body / rng > 0.85:
            if _is_bullish(row):
                patterns.append({
                    'name': '大陽線',
                    'type': 'bullish',
                    'date': date_str,
                    'description': '實體大,買方強勢'
                })
            else:
                patterns.append({
                    'name': '大陰線',
                    'type': 'bearish',
                    'date': date_str,
                    'description': '實體大,賣方強勢'
                })

        # 5. 吞噬型態 (Engulfing) - 需要前一根
        if i > 0:
            prev = df.iloc[i-1]
            if _is_bearish(prev) and _is_bullish(row):
                if row['Open'] < prev['Close'] and row['Close'] > prev['Open']:
                    patterns.append({
                        'name': '多頭吞噬 Bullish Engulfing',
                        'type': 'bullish',
                        'date': date_str,
                        'description': '陽線完全吞噬前陰線,強反轉訊號'
                    })
            elif _is_bullish(prev) and _is_bearish(row):
                if row['Open'] > prev['Close'] and row['Close'] < prev['Open']:
                    patterns.append({
                        'name': '空頭吞噬 Bearish Engulfing',
                        'type': 'bearish',
                        'date': date_str,
                        'description': '陰線完全吞噬前陽線,見頂訊號'
                    })

        # 6. 穿頭破腳 / 孕線 (Harami - 小身體在大身體內)
        if i > 0:
            prev = df.iloc[i-1]
            prev_body = _body(prev)
            if prev_body > 0 and body < prev_body * 0.5:
                # 當前 K 線身體在前 K 線身體內
                cur_hi_body = max(row['Close'], row['Open'])
                cur_lo_body = min(row['Close'], row['Open'])
                prev_hi_body = max(prev['Close'], prev['Open'])
                prev_lo_body = min(prev['Close'], prev['Open'])
                if cur_hi_body < prev_hi_body and cur_lo_body > prev_lo_body:
                    if _is_bearish(prev) and _is_bullish(row):
                        patterns.append({
                            'name': '多頭孕線 Bullish Harami',
                            'type': 'bullish',
                            'date': date_str,
                            'description': '下跌末端小陽孕線,反轉訊號'
                        })
                    elif _is_bullish(prev) and _is_bearish(row):
                        patterns.append({
                            'name': '空頭孕線 Bearish Harami',
                            'type': 'bearish',
                            'date': date_str,
                            'description': '上漲末端小陰孕線,見頂訊號'
                        })

    # 7. 連續 K 線型態 (V 型反轉、N 型等)
    if len(df) >= 10:
        last_10 = df.tail(10)
        # V 型反轉: 前半跌、後半漲、振幅夠大
        first_half = last_10.iloc[:5]
        second_half = last_10.iloc[5:]
        first_range = (first_half['High'].max() - first_half['Low'].min()) / first_half['Close'].iloc[0]
        dropped = first_half['Close'].iloc[-1] < first_half['Close'].iloc[0] * 0.97
        rebounded = second_half['Close'].iloc[-1] > second_half['Close'].iloc[0] * 1.03
        if dropped and rebounded and first_range > 0.05:
            patterns.append({
                'name': 'V 型反轉',
                'type': 'bullish',
                'date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else '',
                'description': '先跌後漲,強勢反轉型態'
            })

        # 倒 V 反轉
        rose = first_half['Close'].iloc[-1] > first_half['Close'].iloc[0] * 1.03
        fell = second_half['Close'].iloc[-1] < second_half['Close'].iloc[0] * 0.97
        if rose and fell and first_range > 0.05:
            patterns.append({
                'name': '倒 V 型反轉',
                'type': 'bearish',
                'date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else '',
                'description': '先漲後跌,高位反轉型態'
            })

    # 8. 連續陰/陽線 (3+ 根)
    if len(df) >= 3:
        last3 = df.tail(3)
        if all(_is_bullish(last3.iloc[j]) for j in range(3)):
            patterns.append({
                'name': '三陽線 (紅三兵)',
                'type': 'bullish',
                'date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else '',
                'description': '連續 3 根陽線,多頭強勢'
            })
        elif all(_is_bearish(last3.iloc[j]) for j in range(3)):
            patterns.append({
                'name': '三陰線 (黑三兵)',
                'type': 'bearish',
                'date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else '',
                'description': '連續 3 根陰線,空頭強勢,但常有技術反彈'
            })

    return {'recent': patterns}
