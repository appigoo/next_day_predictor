"""
Telegram 信號推送模組
"""
import requests
from datetime import datetime
import pytz


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """發送訊息至 Telegram"""
    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False


def format_signal_message(symbol, current, prediction, timeframe) -> str:
    """格式化信號為 Telegram HTML"""
    uk_time = datetime.now(pytz.timezone('Europe/London')).strftime('%Y-%m-%d %H:%M')

    overall = prediction['overall']
    scenarios = prediction['scenarios']
    trade = prediction['trade_plan']

    signal_emoji = {
        'STRONG_BUY': '🟢🟢🟢',
        'BUY': '🟢',
        'HOLD': '🟡',
        'SELL': '🔴',
        'STRONG_SELL': '🔴🔴🔴'
    }.get(overall['signal'], '⚪')

    msg = f"""<b>📊 下個交易日走勢推斷</b>
━━━━━━━━━━━━━━━━
<b>🎯 {symbol} ({timeframe})</b>
💰 現價: <b>${current['Close']:.2f}</b>
📈 RSI: {current['RSI']:.1f} | ATR: ${current['ATR']:.2f}

<b>{signal_emoji} 信號: {overall['signal'].replace('_', ' ')}</b>
綜合分: {overall['score']:.1f}/100
置信度: {overall['confidence']:.0%}

<b>📊 情境機率</b>
"""
    for sc in scenarios:
        icon = "🟢" if sc['direction'] == 'up' else "🔴" if sc['direction'] == 'down' else "🟡"
        msg += f"{icon} {sc['name']}: <b>{sc['probability']:.0%}</b>\n"
        msg += f"   目標: {sc['target']}\n"

    msg += f"""
<b>⚡ 交易計劃</b>
🟢 進場: ${trade['entry_low']:.2f} - ${trade['entry_high']:.2f}
🔴 停損: ${trade['stop_loss']:.2f}
🎯 目標1: ${trade['target_1']:.2f}
🎯 目標2: ${trade['target_2']:.2f}
📏 R:R: {trade['rr_ratio']:.2f}
📊 倉位: {trade['position_size']}

<b>📝 綜合判斷</b>
{prediction['summary']}

<i>⏰ {uk_time} UK | ⚠️ 僅供參考非投資建議</i>
"""
    return msg
