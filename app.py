"""
下個交易日走勢推斷 - Multi-Stock Intelligent Dashboard
純 Python 技術指標 (無 TA-Lib / pandas-ta)
部署於 Streamlit Cloud
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import requests
import time

from indicators import (
    ema, sma, rsi, macd, atr, adx,
    bollinger_bands, find_support_resistance,
    detect_candle_patterns, fibonacci_levels
)
from analyzer import NextDayPredictor
from telegram_alerts import send_telegram_message, format_signal_message

# ==================== 頁面設定 ====================
st.set_page_config(
    page_title="下個交易日走勢推斷",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS - 乳白米色主題
st.markdown("""
<style>
    .stApp {
        background-color: #FAF7F0;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2C3E50;
        padding: 1rem 0;
        border-bottom: 3px solid #D4A574;
    }
    .scenario-card {
        background: #FFFFFF;
        border-left: 4px solid #D4A574;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .bullish { border-left-color: #27AE60 !important; }
    .bearish { border-left-color: #E74C3C !important; }
    .neutral { border-left-color: #F39C12 !important; }
    .metric-box {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .signal-strong-buy { color: #27AE60; font-weight: bold; }
    .signal-buy { color: #52BE80; font-weight: bold; }
    .signal-hold { color: #F39C12; font-weight: bold; }
    .signal-sell { color: #E67E22; font-weight: bold; }
    .signal-strong-sell { color: #E74C3C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== 頂部工具列 (持久) ====================
st.markdown('<div class="main-header">📊 下個交易日走勢推斷 Dashboard</div>',
            unsafe_allow_html=True)

top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([2, 2, 2, 2, 2])

with top_col1:
    symbol = st.selectbox(
        "🎯 選擇股票",
        ["TSLA", "AMZN", "AAPL", "NVDA", "GOOGL", "META", "MSFT", "SPY", "QQQ"],
        index=0
    )

with top_col2:
    timeframe = st.selectbox(
        "⏱️ 時間框架",
        ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"],
        index=5
    )

with top_col3:
    period_map = {
        "1m": "7d", "5m": "60d", "15m": "60d",
        "30m": "60d", "1h": "730d", "1d": "2y", "1wk": "5y"
    }
    period = period_map.get(timeframe, "1y")
    st.metric("📅 數據範圍", period)

with top_col4:
    auto_refresh = st.checkbox("🔄 自動刷新 (60s)", value=False)

with top_col5:
    telegram_enabled = st.checkbox("📱 Telegram 警報", value=False)

st.divider()

# ==================== 側邊欄設定 ====================
with st.sidebar:
    st.header("⚙️ 進階設定")

    st.subheader("📊 多股監控")
    multi_watch = st.multiselect(
        "監控清單",
        ["TSLA", "AMZN", "AAPL", "NVDA", "GOOGL", "META", "MSFT", "SPY", "QQQ", "UVXY"],
        default=["TSLA", "AMZN", "AAPL", "NVDA", "GOOGL", "META"]
    )

    st.subheader("🎚️ 技術指標參數")
    ema_fast = st.slider("EMA 快線", 5, 50, 20)
    ema_slow = st.slider("EMA 慢線", 20, 200, 50)
    ema_trend = st.slider("EMA 趨勢", 50, 300, 200)
    rsi_period = st.slider("RSI 週期", 7, 30, 14)
    atr_period = st.slider("ATR 週期", 7, 30, 14)

    st.subheader("🎯 場景機率權重")
    trend_weight = st.slider("趨勢權重", 0.0, 1.0, 0.35, 0.05)
    momentum_weight = st.slider("動能權重", 0.0, 1.0, 0.25, 0.05)
    pattern_weight = st.slider("型態權重", 0.0, 1.0, 0.25, 0.05)
    volume_weight = st.slider("量能權重", 0.0, 1.0, 0.15, 0.05)

    st.subheader("📱 Telegram 設定")
    if telegram_enabled:
        tg_token = st.text_input("Bot Token",
                                  value=st.secrets.get("TELEGRAM_BOT_TOKEN", ""),
                                  type="password")
        tg_chat_id = st.text_input("Chat ID",
                                    value=st.secrets.get("TELEGRAM_CHAT_ID", ""))

    st.subheader("ℹ️ 系統資訊")
    uk_tz = pytz.timezone('Europe/London')
    ny_tz = pytz.timezone('America/New_York')
    st.text(f"UK:  {datetime.now(uk_tz).strftime('%H:%M:%S')}")
    st.text(f"NYC: {datetime.now(ny_tz).strftime('%H:%M:%S')}")

# ==================== 數據載入 ====================
@st.cache_data(ttl=60)
def load_data(sym, tf, per):
    """載入股票數據"""
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period=per, interval=tf)
        if df.empty:
            return None, None
        df.index = pd.to_datetime(df.index)
        info = ticker.info
        return df, info
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None

@st.cache_data(ttl=300)
def load_vix():
    """載入 VIX 數據"""
    try:
        vix = yf.Ticker("^VIX").history(period="5d", interval="1d")
        return vix
    except:
        return None

with st.spinner(f"📡 載入 {symbol} {timeframe} 數據..."):
    df, info = load_data(symbol, timeframe, period)
    vix_df = load_vix()

if df is None or df.empty:
    st.error("❌ 無法載入數據,請檢查網路或股票代號")
    st.stop()

# ==================== 計算技術指標 ====================
df['EMA_fast'] = ema(df['Close'], ema_fast)
df['EMA_slow'] = ema(df['Close'], ema_slow)
df['EMA_trend'] = ema(df['Close'], ema_trend)
df['RSI'] = rsi(df['Close'], rsi_period)
df['ATR'] = atr(df['High'], df['Low'], df['Close'], atr_period)
macd_line, signal_line, hist = macd(df['Close'])
df['MACD'] = macd_line
df['MACD_signal'] = signal_line
df['MACD_hist'] = hist
df['ADX'] = adx(df['High'], df['Low'], df['Close'], 14)
bb_upper, bb_mid, bb_lower = bollinger_bands(df['Close'], 20, 2)
df['BB_upper'] = bb_upper
df['BB_mid'] = bb_mid
df['BB_lower'] = bb_lower
df['Vol_MA20'] = df['Volume'].rolling(20).mean()

# 支撐壓力
sr_levels = find_support_resistance(df, lookback=60)

# K線型態
patterns = detect_candle_patterns(df)

# 斐波那契
fib = fibonacci_levels(df['High'].tail(60).max(), df['Low'].tail(60).min())

# ==================== 執行預測分析 ====================
vix_value = vix_df['Close'].iloc[-1] if vix_df is not None and not vix_df.empty else None
vix_change = 0
if vix_df is not None and len(vix_df) >= 2:
    vix_change = (vix_df['Close'].iloc[-1] / vix_df['Close'].iloc[-2] - 1) * 100

predictor = NextDayPredictor(
    df=df,
    patterns=patterns,
    sr_levels=sr_levels,
    fib_levels=fib,
    vix=vix_value,
    vix_change=vix_change,
    weights={
        'trend': trend_weight,
        'momentum': momentum_weight,
        'pattern': pattern_weight,
        'volume': volume_weight
    }
)
prediction = predictor.predict()

# ==================== 頂部即時數據 ====================
current = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else current
price_change = current['Close'] - prev['Close']
price_change_pct = (price_change / prev['Close']) * 100

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric(f"💰 {symbol}",
              f"${current['Close']:.2f}",
              f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
with m2:
    st.metric("📊 成交量",
              f"{current['Volume']/1e6:.1f}M",
              f"{(current['Volume']/current['Vol_MA20']-1)*100:+.0f}% vs MA20"
              if not pd.isna(current['Vol_MA20']) else None)
with m3:
    st.metric("📈 RSI", f"{current['RSI']:.1f}",
              "超買" if current['RSI'] > 70 else "超賣" if current['RSI'] < 30 else "中性")
with m4:
    st.metric("🎯 ATR",
              f"${current['ATR']:.2f}",
              f"{(current['ATR']/current['Close']*100):.1f}%")
with m5:
    adx_val = current['ADX']
    trend_str = "強趨勢" if adx_val > 25 else "盤整" if adx_val < 20 else "趨勢中"
    st.metric("📐 ADX", f"{adx_val:.1f}", trend_str)
with m6:
    if vix_value:
        st.metric("😰 VIX", f"{vix_value:.2f}", f"{vix_change:+.2f}%")

st.divider()

# ==================== 主圖表 ====================
st.subheader(f"📈 {symbol} K線圖 + 技術指標")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.5, 0.15, 0.15, 0.2],
    subplot_titles=('價格', '成交量', 'MACD', 'RSI')
)

# K線
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name='K線',
    increasing_line_color='#26A69A',
    decreasing_line_color='#EF5350'
), row=1, col=1)

# EMA
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_fast'],
                          name=f'EMA{ema_fast}',
                          line=dict(color='#3498DB', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_slow'],
                          name=f'EMA{ema_slow}',
                          line=dict(color='#E67E22', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_trend'],
                          name=f'EMA{ema_trend}',
                          line=dict(color='#9B59B6', width=1.5)), row=1, col=1)

# 支撐壓力
for level in sr_levels.get('resistance', [])[:3]:
    fig.add_hline(y=level, line_dash="dash", line_color="#E74C3C",
                   annotation_text=f"R: {level:.2f}",
                   annotation_position="right", row=1, col=1)
for level in sr_levels.get('support', [])[:3]:
    fig.add_hline(y=level, line_dash="dash", line_color="#27AE60",
                   annotation_text=f"S: {level:.2f}",
                   annotation_position="right", row=1, col=1)

# 斐波那契重要位
fib_key_levels = {'0.382': fib.get('0.382'), '0.5': fib.get('0.5'),
                  '0.618': fib.get('0.618')}
for name, val in fib_key_levels.items():
    if val:
        fig.add_hline(y=val, line_dash="dot", line_color="#D4A574",
                       annotation_text=f"Fib {name}: {val:.2f}",
                       annotation_position="left", row=1, col=1)

# 成交量
vol_colors = ['#26A69A' if df['Close'].iloc[i] >= df['Open'].iloc[i]
              else '#EF5350' for i in range(len(df))]
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                     marker_color=vol_colors, showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Vol_MA20'],
                          name='Vol MA20', line=dict(color='#D4A574', width=1.5)),
              row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                          line=dict(color='#3498DB', width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal',
                          line=dict(color='#E67E22', width=1.5)), row=3, col=1)
macd_colors = ['#26A69A' if v >= 0 else '#EF5350' for v in df['MACD_hist']]
fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram',
                     marker_color=macd_colors, showlegend=False), row=3, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                          line=dict(color='#9B59B6', width=1.5)), row=4, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#E74C3C", row=4, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#27AE60", row=4, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="gray", row=4, col=1)

fig.update_layout(
    height=800,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    template='plotly_white',
    paper_bgcolor='#FAF7F0',
    plot_bgcolor='#FFFFFF',
    margin=dict(l=10, r=10, t=30, b=10)
)
fig.update_xaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# ==================== 下個交易日預測 (核心) ====================
st.subheader("🎯 下個交易日走勢推斷")

pred_col1, pred_col2 = st.columns([1, 1])

with pred_col1:
    # 整體信號
    overall = prediction['overall']
    signal_color = {
        'STRONG_BUY': '#27AE60', 'BUY': '#52BE80',
        'HOLD': '#F39C12', 'SELL': '#E67E22',
        'STRONG_SELL': '#E74C3C'
    }.get(overall['signal'], '#95A5A6')

    st.markdown(f"""
    <div style="background: {signal_color}; color: white; padding: 1.5rem;
                border-radius: 10px; text-align: center;">
        <h2 style="margin:0;">{overall['signal'].replace('_', ' ')}</h2>
        <p style="margin:0; font-size: 1.1rem;">綜合評分: {overall['score']:.1f}/100</p>
        <p style="margin:0;">置信度: {overall['confidence']:.0%}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 機率分佈")
    scenarios = prediction['scenarios']

    for sc in scenarios:
        css_class = "bullish" if sc['direction'] == 'up' else \
                    "bearish" if sc['direction'] == 'down' else "neutral"
        icon = "🟢" if sc['direction'] == 'up' else \
               "🔴" if sc['direction'] == 'down' else "🟡"

        st.markdown(f"""
        <div class="scenario-card {css_class}">
            <strong>{icon} {sc['name']} (機率 {sc['probability']:.0%})</strong><br>
            <span style="color: #7F8C8D;">觸發: {sc['trigger']}</span><br>
            <span style="color: #2C3E50;">目標: {sc['target']}</span><br>
            <span style="color: #95A5A6; font-size: 0.9rem;">理由: {sc['reason']}</span>
        </div>
        """, unsafe_allow_html=True)

with pred_col2:
    st.markdown("### ⚡ 交易建議")
    trade = prediction['trade_plan']

    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 8px;
                border-left: 4px solid {signal_color};">
        <p>🟢 <strong>進場區間:</strong> ${trade['entry_low']:.2f} - ${trade['entry_high']:.2f}</p>
        <p>🔴 <strong>停損:</strong> ${trade['stop_loss']:.2f}
           ({((trade['stop_loss']/current['Close']-1)*100):+.2f}%)</p>
        <p>🎯 <strong>目標 1:</strong> ${trade['target_1']:.2f}
           ({((trade['target_1']/current['Close']-1)*100):+.2f}%)</p>
        <p>🎯 <strong>目標 2:</strong> ${trade['target_2']:.2f}
           ({((trade['target_2']/current['Close']-1)*100):+.2f}%)</p>
        <p>📏 <strong>風險報酬比 R:R:</strong> {trade['rr_ratio']:.2f}</p>
        <p>📊 <strong>建議倉位:</strong> {trade['position_size']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 外部因素")
    for factor in prediction['external_factors']:
        st.markdown(f"- {factor}")

    st.markdown("### 📝 綜合判斷")
    st.info(prediction['summary'])

# ==================== K線型態分析 ====================
st.divider()
st.subheader("🔍 K線型態分析")

pat_col1, pat_col2, pat_col3 = st.columns(3)

with pat_col1:
    st.markdown("**最近 K 線型態**")
    recent_patterns = patterns.get('recent', [])
    if recent_patterns:
        for p in recent_patterns[-5:]:
            pat_icon = "🟢" if p['type'] == 'bullish' else \
                       "🔴" if p['type'] == 'bearish' else "🟡"
            st.markdown(f"{pat_icon} **{p['name']}** ({p['date']})")
            st.caption(p['description'])
    else:
        st.info("最近無明顯 K 線型態")

with pat_col2:
    st.markdown("**量能分析**")
    recent_vol = df['Volume'].tail(5).mean()
    prior_vol = df['Volume'].tail(20).head(15).mean()
    vol_ratio = recent_vol / prior_vol if prior_vol > 0 else 1

    if vol_ratio > 1.3:
        st.warning(f"🔥 近期放量 {vol_ratio:.2f}x (主力介入)")
    elif vol_ratio < 0.7:
        st.info(f"💤 近期縮量 {vol_ratio:.2f}x (觀望/健康回調)")
    else:
        st.success(f"⚖️ 量能平穩 {vol_ratio:.2f}x")

    # 趨勢中的成交量分析
    recent_up_vol = df[df['Close'] > df['Open']].tail(5)['Volume'].mean()
    recent_dn_vol = df[df['Close'] < df['Open']].tail(5)['Volume'].mean()
    if recent_up_vol > recent_dn_vol * 1.2:
        st.success("📈 漲時放量 > 跌時量 (多頭特徵)")
    elif recent_dn_vol > recent_up_vol * 1.2:
        st.error("📉 跌時放量 > 漲時量 (空頭特徵)")

with pat_col3:
    st.markdown("**斐波那契位置**")
    current_price = current['Close']
    for name, level in sorted(fib.items(), key=lambda x: -x[1]):
        distance = (current_price - level) / current_price * 100
        marker = "👉" if abs(distance) < 1 else "  "
        st.markdown(f"{marker} **Fib {name}**: ${level:.2f} ({distance:+.2f}%)")

# ==================== 多股監控面板 ====================
st.divider()
st.subheader("📊 多股即時監控")

@st.cache_data(ttl=60)
def get_multi_stock_summary(symbols):
    """獲取多股摘要"""
    results = []
    for s in symbols:
        try:
            t = yf.Ticker(s)
            hist = t.history(period="5d", interval="1d")
            if hist.empty or len(hist) < 2:
                continue
            last = hist.iloc[-1]
            prev = hist.iloc[-2]
            rsi_val = rsi(hist['Close'], 14).iloc[-1]
            results.append({
                'Symbol': s,
                'Price': last['Close'],
                'Change %': (last['Close']/prev['Close']-1)*100,
                'Volume (M)': last['Volume']/1e6,
                'RSI': rsi_val,
                'Range %': (last['High']-last['Low'])/last['Close']*100
            })
        except:
            continue
    return pd.DataFrame(results)

multi_df = get_multi_stock_summary(multi_watch)
if not multi_df.empty:
    def style_row(row):
        colors = []
        for col in row.index:
            if col == 'Change %':
                c = '#27AE60' if row[col] > 0 else '#E74C3C'
                colors.append(f'color: {c}; font-weight: bold')
            elif col == 'RSI':
                if row[col] > 70:
                    colors.append('color: #E74C3C; font-weight: bold')
                elif row[col] < 30:
                    colors.append('color: #27AE60; font-weight: bold')
                else:
                    colors.append('')
            else:
                colors.append('')
        return colors

    styled = multi_df.style.apply(style_row, axis=1).format({
        'Price': '${:.2f}',
        'Change %': '{:+.2f}%',
        'Volume (M)': '{:.1f}',
        'RSI': '{:.1f}',
        'Range %': '{:.2f}%'
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ==================== 多時間框架確認 ====================
st.divider()
st.subheader("⏱️ 多時間框架確認")

@st.cache_data(ttl=120)
def multi_timeframe_signals(sym):
    """多時間框架信號"""
    tfs = [('15m', '5d'), ('1h', '60d'), ('1d', '1y'), ('1wk', '3y')]
    signals = []
    for tf, per in tfs:
        try:
            d = yf.Ticker(sym).history(period=per, interval=tf)
            if len(d) < 50:
                continue
            ema20 = ema(d['Close'], 20)
            ema50 = ema(d['Close'], 50)
            r = rsi(d['Close'], 14)
            price = d['Close'].iloc[-1]

            trend = "↑ 多" if ema20.iloc[-1] > ema50.iloc[-1] else "↓ 空"
            above_ema = "在 EMA20 之上" if price > ema20.iloc[-1] else "在 EMA20 之下"
            rsi_now = r.iloc[-1]

            signals.append({
                '時間框架': tf,
                '價格': f"${price:.2f}",
                '趨勢': trend,
                '位置': above_ema,
                'RSI': f"{rsi_now:.1f}"
            })
        except:
            continue
    return pd.DataFrame(signals)

mtf_df = multi_timeframe_signals(symbol)
if not mtf_df.empty:
    st.dataframe(mtf_df, use_container_width=True, hide_index=True)

# ==================== Telegram 警報 ====================
if telegram_enabled and 'tg_token' in locals() and tg_token and tg_chat_id:
    if st.button("📱 發送當前信號到 Telegram"):
        msg = format_signal_message(symbol, current, prediction, timeframe)
        result = send_telegram_message(tg_token, tg_chat_id, msg)
        if result:
            st.success("✅ 已發送到 Telegram")
        else:
            st.error("❌ 發送失敗,請檢查 Token 和 Chat ID")

# ==================== 自動刷新 ====================
if auto_refresh:
    time.sleep(60)
    st.rerun()

# ==================== 頁腳 ====================
st.divider()
st.caption(f"⚠️ 本儀表板僅供參考,不構成投資建議 | "
           f"最後更新: {datetime.now(pytz.timezone('Europe/London')).strftime('%Y-%m-%d %H:%M:%S')} UK | "
           f"數據源: Yahoo Finance")
