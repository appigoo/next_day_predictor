"""
韌性數據獲取模組
解決 Streamlit Cloud 上 Yahoo Finance rate limit 問題

策略:
1. yfinance (主) - 帶重試 + 指數退避
2. Stooq (備援) - 免費、無 rate limit
3. 磁碟快取 (diskcache) - 失敗時使用最後成功數據
4. Session state 快取 - 同一 session 不重複請求
"""
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO
import streamlit as st
import os

# 嘗試載入 diskcache(可選依賴)
try:
    import diskcache as dc
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.next_day_predictor_cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    _disk_cache = dc.Cache(CACHE_DIR, size_limit=100 * 1024 * 1024)  # 100MB
    DISK_CACHE_OK = True
except Exception:
    _disk_cache = {}
    DISK_CACHE_OK = False

# 輪換 User-Agent (降低被識別為 bot)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
]


def _cache_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}__{period}__{interval}"


def _save_to_cache(key: str, df: pd.DataFrame):
    """儲存到磁碟快取"""
    try:
        if DISK_CACHE_OK:
            _disk_cache.set(key, df, expire=86400 * 7)  # 7 天過期
        else:
            _disk_cache[key] = (df, datetime.now())
    except Exception:
        pass


def _load_from_cache(key: str, max_age_hours: int = 24):
    """從磁碟讀取快取"""
    try:
        if DISK_CACHE_OK:
            return _disk_cache.get(key)
        else:
            if key in _disk_cache:
                df, ts = _disk_cache[key]
                if (datetime.now() - ts).total_seconds() < max_age_hours * 3600:
                    return df
    except Exception:
        pass
    return None


def _fetch_yfinance(symbol: str, period: str, interval: str,
                    max_retries: int = 3) -> pd.DataFrame:
    """
    使用 yfinance 帶重試機制
    遇到 rate limit 指數退避等待
    """
    import yfinance as yf

    last_err = None
    for attempt in range(max_retries):
        try:
            # 隨機抖動,避免同步請求風暴
            if attempt > 0:
                wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(wait)

            # 建立 custom session 帶 User-Agent
            session = requests.Session()
            session.headers.update({
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            })

            ticker = yf.Ticker(symbol, session=session)
            df = ticker.history(period=period, interval=interval, timeout=15)

            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # rate limit 或 429 錯誤才重試,其他錯誤快速失敗
            if 'too many' in err_str or 'rate' in err_str or '429' in err_str:
                continue
            elif attempt == max_retries - 1:
                raise
    if last_err:
        raise last_err
    return pd.DataFrame()


def _fetch_stooq(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Stooq 備援數據源 (免費、無 rate limit)
    僅支援日線和週線
    """
    # Stooq 僅支援日線/週線/月線,分鐘級跳過
    if interval in ['1m', '5m', '15m', '30m', '1h']:
        return pd.DataFrame()

    interval_map = {
        '1d': 'd',
        '1wk': 'w',
        '1mo': 'm'
    }
    stq_interval = interval_map.get(interval, 'd')

    # Stooq 美股代碼加 .us 後綴
    stq_symbol = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stq_symbol}&i={stq_interval}"

    try:
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or 'No data' in r.text:
            return pd.DataFrame()

        df = pd.read_csv(StringIO(r.text))
        if df.empty or 'Date' not in df.columns:
            return pd.DataFrame()

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()

        # 標準化欄位名 (Stooq 已經是 Open/High/Low/Close/Volume)
        # 確認欄位存在
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        # 根據 period 截取
        period_days = {
            '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
            'max': 36500, '60d': 60, '7d': 7, '730d': 730
        }
        days = period_days.get(period, 365)
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]

        return df[required]
    except Exception:
        return pd.DataFrame()


def fetch_stock_data(symbol: str, period: str, interval: str,
                      use_cache: bool = True) -> tuple:
    """
    主數據獲取函數 - 多源備援
    回傳 (df, source_name, is_stale)
    """
    key = _cache_key(symbol, period, interval)

    # Step 1: 嘗試 yfinance (主源)
    try:
        df = _fetch_yfinance(symbol, period, interval, max_retries=3)
        if not df.empty:
            _save_to_cache(key, df)
            return df, 'yfinance', False
    except Exception as e:
        err_msg = str(e).lower()
        if 'too many' in err_msg or 'rate' in err_msg or '429' in err_msg:
            st.warning(f"⚠️ yfinance rate limit,嘗試備援源...")
        else:
            st.info(f"ℹ️ yfinance 失敗 ({str(e)[:50]}),嘗試備援源...")

    # Step 2: 嘗試 Stooq (備援,日/週線有效)
    if interval in ['1d', '1wk', '1mo']:
        df = _fetch_stooq(symbol, period, interval)
        if not df.empty:
            _save_to_cache(key, df)
            return df, 'Stooq (備援)', False

    # Step 3: 回落到磁碟快取
    if use_cache:
        cached = _load_from_cache(key, max_age_hours=48)
        if cached is not None and not cached.empty:
            return cached, 'Cache (快取,可能過時)', True

    # 全部失敗
    return pd.DataFrame(), 'Failed', True


def fetch_vix(use_cache: bool = True) -> tuple:
    """獲取 VIX (多源備援)"""
    key = "__VIX__daily"

    # Try yfinance
    try:
        df = _fetch_yfinance("^VIX", "5d", "1d", max_retries=2)
        if not df.empty:
            _save_to_cache(key, df)
            return df, 'yfinance', False
    except Exception:
        pass

    # Try Stooq (VIX 在 Stooq 為 ^vix)
    try:
        url = "https://stooq.com/q/d/l/?s=^vix&i=d"
        r = requests.get(url, headers={'User-Agent': random.choice(USER_AGENTS)},
                          timeout=15)
        if r.status_code == 200 and 'No data' not in r.text:
            df = pd.read_csv(StringIO(r.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index().tail(10)
            if not df.empty:
                _save_to_cache(key, df)
                return df, 'Stooq', False
    except Exception:
        pass

    # Cache fallback
    if use_cache:
        cached = _load_from_cache(key, max_age_hours=48)
        if cached is not None and not cached.empty:
            return cached, 'Cache', True

    return pd.DataFrame(), 'Failed', True


def clear_cache():
    """清除快取"""
    try:
        if DISK_CACHE_OK:
            _disk_cache.clear()
        else:
            _disk_cache.clear()
        return True
    except Exception:
        return False
