"""
韌性數據獲取模組 v3
===================
問題根源: yfinance 新版要求 curl_cffi session,舊版 requests session 不再有效
         Stooq 在 Streamlit Cloud 出口 IP 被封

解決策略 (4 層):
1. yfinance + curl_cffi session (修復新版 Yahoo API 要求)
2. Alpha Vantage API (免費 500次/天,無 IP 封鎖)
3. Finnhub API (免費 60次/分鐘,無 IP 封鎖)
4. 磁碟快取 (diskcache) - 所有源都掛時用舊數據
"""

import time
import random
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ── 磁碟快取 ────────────────────────────────────────────────────────────────
try:
    import diskcache as dc
    _CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ndp_cache")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    _disk = dc.Cache(_CACHE_DIR, size_limit=200 * 1024 * 1024)
    _DISK_OK = True
except Exception:
    _disk = {}
    _DISK_OK = False


def _disk_get(key):
    try:
        return _disk.get(key) if _DISK_OK else _disk.get(key)
    except Exception:
        return None


def _disk_set(key, val, expire=86400 * 7):
    try:
        if _DISK_OK:
            _disk.set(key, val, expire=expire)
        else:
            _disk[key] = (val, datetime.now())
    except Exception:
        pass


def _cache_key(symbol, period, interval):
    return f"{symbol}__{period}__{interval}"


_PERIOD_DAYS = {
    "5d": 5, "7d": 7, "1mo": 30, "3mo": 90, "60d": 60,
    "6mo": 180, "1y": 365, "2y": 730, "3y": 1095,
    "5y": 1825, "730d": 730, "max": 36500,
}


def _period_to_dates(period):
    days = _PERIOD_DAYS.get(period, 365)
    end = datetime.now()
    start = end - timedelta(days=days)
    return start, end


# ══════════════════════════════════════════════════════════════════════════════
# 源 1: yfinance + curl_cffi
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yfinance(symbol, period, interval, retries=3):
    import yfinance as yf

    session = None
    try:
        from curl_cffi import requests as cffi_req
        session = cffi_req.Session(impersonate="chrome")
    except ImportError:
        pass

    last_exc = None
    for attempt in range(retries):
        if attempt:
            time.sleep(2 ** attempt + random.uniform(0.3, 1.5))
        try:
            ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, timeout=20)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            if not any(k in msg for k in ("429", "too many", "rate", "timeout",
                                           "connection", "curl_cffi", "session")):
                break
    if last_exc:
        raise last_exc
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 源 2: Alpha Vantage
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_alpha_vantage(symbol, period, interval):
    key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
    if not key:
        return pd.DataFrame()

    _intra = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min"}
    _daily = {"1d": "TIME_SERIES_DAILY", "1wk": "TIME_SERIES_WEEKLY", "1mo": "TIME_SERIES_MONTHLY"}
    base = "https://www.alphavantage.co/query"

    try:
        if interval in _intra:
            params = {"function": "TIME_SERIES_INTRADAY", "symbol": symbol,
                      "interval": _intra[interval], "outputsize": "full",
                      "apikey": key, "datatype": "json"}
            r = requests.get(base, params=params, timeout=20)
            data = r.json()
            ts_key = f"Time Series ({_intra[interval]})"
            if ts_key not in data:
                return pd.DataFrame()
            raw = data[ts_key]
        elif interval in _daily:
            params = {"function": _daily[interval], "symbol": symbol,
                      "outputsize": "full", "apikey": key, "datatype": "json"}
            r = requests.get(base, params=params, timeout=20)
            data = r.json()
            ts_key = next((k for k in data if "Time Series" in k or
                           "Weekly" in k or "Monthly" in k), None)
            if not ts_key:
                return pd.DataFrame()
            raw = data[ts_key]
        else:
            return pd.DataFrame()

        rows = []
        for dt_str, v in raw.items():
            rows.append({"Date": pd.to_datetime(dt_str),
                         "Open": float(v.get("1. open", 0)),
                         "High": float(v.get("2. high", 0)),
                         "Low": float(v.get("3. low", 0)),
                         "Close": float(v.get("4. close", 0)),
                         "Volume": float(v.get("5. volume", 0))})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("Date").sort_index()
        start, _ = _period_to_dates(period)
        return df[df.index >= start]
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 源 3: Finnhub
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_finnhub(symbol, period, interval):
    key = st.secrets.get("FINNHUB_KEY", "")
    if not key:
        return pd.DataFrame()

    _res = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "1d": "D", "1wk": "W", "1mo": "M"}
    resolution = _res.get(interval)
    if not resolution:
        return pd.DataFrame()

    start, end = _period_to_dates(period)
    try:
        r = requests.get("https://finnhub.io/api/v1/stock/candle",
                         params={"symbol": symbol, "resolution": resolution,
                                 "from": int(start.timestamp()),
                                 "to": int(end.timestamp()), "token": key},
                         timeout=20)
        data = r.json()
        if data.get("s") != "ok" or not data.get("t"):
            return pd.DataFrame()
        df = pd.DataFrame({"Open": data["o"], "High": data["h"],
                           "Low": data["l"], "Close": data["c"],
                           "Volume": data["v"]},
                          index=pd.to_datetime(data["t"], unit="s"))
        df.index.name = "Date"
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 主函數
# ══════════════════════════════════════════════════════════════════════════════
def fetch_stock_data(symbol, period, interval, use_cache=True):
    """回傳 (df, source_label, is_stale)"""
    key = _cache_key(symbol, period, interval)

    # 源 1: yfinance
    try:
        df = _fetch_yfinance(symbol, period, interval)
        if not df.empty:
            _disk_set(key, df)
            return df, "yfinance ✅", False
    except Exception as e:
        st.info(f"ℹ️ yfinance 失敗 ({str(e)[:80]}),嘗試備援源...")

    # 源 2: Alpha Vantage
    if st.secrets.get("ALPHA_VANTAGE_KEY", ""):
        df = _fetch_alpha_vantage(symbol, period, interval)
        if not df.empty:
            _disk_set(key, df)
            return df, "Alpha Vantage 🔄", False
        st.info("ℹ️ Alpha Vantage 無數據,嘗試 Finnhub...")

    # 源 3: Finnhub
    if st.secrets.get("FINNHUB_KEY", ""):
        df = _fetch_finnhub(symbol, period, interval)
        if not df.empty:
            _disk_set(key, df)
            return df, "Finnhub 🔄", False
        st.info("ℹ️ Finnhub 無數據,嘗試快取...")

    # 源 4: 磁碟快取
    if use_cache:
        cached = _disk_get(key)
        if cached is not None:
            if isinstance(cached, tuple):
                df_c, ts = cached
                if (datetime.now() - ts).total_seconds() < 48 * 3600 and not df_c.empty:
                    return df_c, "記憶體快取 ⚠️", True
            elif not cached.empty:
                return cached, "磁碟快取 ⚠️", True

    return pd.DataFrame(), "全部失敗 ❌", True


def fetch_vix(use_cache=True):
    key = "__VIX__1d"
    try:
        df = _fetch_yfinance("^VIX", "5d", "1d", retries=2)
        if not df.empty:
            _disk_set(key, df)
            return df, "yfinance", False
    except Exception:
        pass
    if use_cache:
        cached = _disk_get(key)
        if cached is not None and not isinstance(cached, tuple) and not cached.empty:
            return cached, "快取", True
    return pd.DataFrame(), "失敗", True


def fetch_multi_stock(symbols):
    rows = []
    for sym in symbols:
        try:
            df, src, stale = fetch_stock_data(sym, "5d", "1d", use_cache=True)
            if df.empty or len(df) < 2:
                continue
            last, prev = df.iloc[-1], df.iloc[-2]
            from indicators import rsi
            rsi_val = rsi(df["Close"], 14).iloc[-1]
            rows.append({"Symbol": sym, "Price": last["Close"],
                         "Change %": (last["Close"] / prev["Close"] - 1) * 100,
                         "Volume (M)": last["Volume"] / 1e6,
                         "RSI": rsi_val,
                         "Range %": (last["High"] - last["Low"]) / last["Close"] * 100,
                         "Source": src})
        except Exception:
            pass
        time.sleep(0.3)
    return pd.DataFrame(rows)


def clear_cache():
    try:
        _disk.clear() if _DISK_OK else _disk.clear()
        return True
    except Exception:
        return False


def show_api_status():
    av = st.secrets.get("ALPHA_VANTAGE_KEY", "")
    fh = st.secrets.get("FINNHUB_KEY", "")
    st.markdown("**📡 備援 API 狀態**")
    st.markdown(f"{'✅' if av else '❌'} Alpha Vantage {'(已設定)' if av else '(未設定)'}")
    st.markdown(f"{'✅' if fh else '❌'} Finnhub {'(已設定)' if fh else '(未設定)'}")
    if not av and not fh:
        with st.expander("⚙️ 如何設定備援 API?"):
            st.markdown("""
**Alpha Vantage** — 免費 500次/天
1. [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) 填 email 即得
2. Streamlit Cloud → App → Settings → Secrets 加入:
```toml
ALPHA_VANTAGE_KEY = "你的KEY"
```
**Finnhub** — 免費 60次/分鐘
1. [finnhub.io/register](https://finnhub.io/register) 免費註冊
2. 同樣加入 Secrets:
```toml
FINNHUB_KEY = "你的KEY"
```
            """)
