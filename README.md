# 📊 下個交易日走勢推斷 Dashboard

純 Python 建構的智能股市儀表板,支援多股票、多時間框架監控,並透過 Telegram 推送交易信號。

## ✨ 主要功能

### 🎯 下個交易日預測引擎
- **3 情境機率分佈**:反彈 / 橫盤 / 續跌
- **4 維度評分**:趨勢 / 動能 / 型態 / 量能
- **VIX 環境調整**:根據恐慌指數動態調整
- **可執行交易計劃**:進場區間 / 停損 / 雙目標 / R:R 比

### 📈 技術分析
- **純 Python 指標**(無 TA-Lib / pandas-ta):
  - EMA (20/50/200)
  - RSI (Wilder's smoothing)
  - MACD
  - ATR / ADX
  - Bollinger Bands
- **自動支撐壓力辨識**(局部極值聚類)
- **斐波那契回撤位**
- **K 線型態辨識**(16+ 型態):
  - 錘子線 / 流星 / 十字星
  - 多頭吞噬 / 空頭吞噬
  - 多頭孕線 / 空頭孕線
  - V 型反轉 / 倒 V 反轉
  - 紅三兵 / 黑三兵
  - 大陽線 / 大陰線

### ⏱️ 多時間框架
1m / 5m / 15m / 30m / 1h / 1d / 1wk

### 📊 多股監控
TSLA, AMZN, AAPL, NVDA, GOOGL, META, MSFT, SPY, QQQ, UVXY

### 📱 Telegram 推送
一鍵發送當前信號到 Telegram,包含完整交易計劃

---

## 🚀 部署到 Streamlit Cloud

### 步驟 1: 上傳到 GitHub
```bash
git init
git add .
git commit -m "initial next_day_predictor"
git remote add origin https://github.com/你的帳號/next-day-predictor.git
git push -u origin main
```

**重要**:將 `.streamlit/secrets.toml` 加入 `.gitignore`,勿上傳真實 Token

### 步驟 2: Streamlit Cloud
1. 前往 https://share.streamlit.io/
2. 連接 GitHub,選擇這個 repository
3. 主檔案路徑填 `app.py`
4. 在 **Secrets** 管理介面加入:
```toml
TELEGRAM_BOT_TOKEN = "你的 Bot Token"
TELEGRAM_CHAT_ID = "你的 Chat ID"
```
5. 點 Deploy

### 步驟 3: 取得 Telegram Bot Token
1. 在 Telegram 搜尋 `@BotFather`
2. `/newbot` → 取得 Token
3. 用你的 Bot 發訊息給自己
4. 開啟 `https://api.telegram.org/bot<TOKEN>/getUpdates` 取得 chat_id

---

## 📁 檔案結構
```
next_day_predictor/
├── app.py                     # Streamlit 主程式
├── indicators.py              # 純 Python 技術指標
├── analyzer.py                # 預測核心引擎
├── telegram_alerts.py         # Telegram 推送
├── requirements.txt
├── .streamlit/
│   ├── config.toml           # 主題設定
│   └── secrets.toml.example  # Secrets 範本
└── README.md
```

---

## 🎨 設計原則

- **乳白米色主題**:低刺激、長時間監控友好
- **持久頂部工具列**:股票、時間框架、刷新、Telegram 開關始終可見
- **純 Python**:Streamlit Cloud 原生相容,無編譯依賴
- **快取機制**:`@st.cache_data(ttl=60)` 避免頻繁 API 呼叫

---

## 📊 預測引擎邏輯

```
綜合分 = 趨勢×0.35 + 動能×0.25 + 型態×0.25 + 量能×0.15 + VIX 調整

分數 > 50  → STRONG_BUY
分數 > 20  → BUY
分數 ∈ (-20, 20) → HOLD
分數 < -20 → SELL
分數 < -50 → STRONG_SELL
```

情境機率以綜合分映射,並依 ADX 強度調整橫盤權重。

---

## ⚠️ 免責聲明

本儀表板僅供技術分析教育參考,**不構成投資建議**。交易有風險,盈虧自負。
