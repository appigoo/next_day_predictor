"""
下個交易日預測核心引擎
基於:
  - 趨勢 (EMA 排列, ADX)
  - 動能 (RSI, MACD)
  - K 線型態
  - 量能分析
  - 支撐壓力 / 斐波那契
  - VIX 環境
綜合產出 3 個情境 (反彈/橫盤/續跌) 與機率分佈
"""
import pandas as pd
import numpy as np


class NextDayPredictor:
    def __init__(self, df, patterns, sr_levels, fib_levels, vix=None, vix_change=0,
                 weights=None):
        self.df = df
        self.patterns = patterns
        self.sr = sr_levels
        self.fib = fib_levels
        self.vix = vix
        self.vix_change = vix_change
        self.weights = weights or {
            'trend': 0.35, 'momentum': 0.25, 'pattern': 0.25, 'volume': 0.15
        }
        self.current = df.iloc[-1]
        self.prev = df.iloc[-2] if len(df) > 1 else self.current

    # ---------- 子評分模組 ----------

    def _score_trend(self):
        """趨勢評分 [-100, +100]"""
        score = 0
        reasons = []

        # EMA 排列
        ema_f = self.current['EMA_fast']
        ema_s = self.current['EMA_slow']
        ema_t = self.current['EMA_trend']
        price = self.current['Close']

        if pd.notna(ema_f) and pd.notna(ema_s) and pd.notna(ema_t):
            # 多頭排列
            if price > ema_f > ema_s > ema_t:
                score += 40
                reasons.append("✅ EMA 多頭排列 (價>快>慢>長)")
            elif price > ema_f and ema_f > ema_s:
                score += 25
                reasons.append("✅ 短期多頭排列")
            # 空頭排列
            elif price < ema_f < ema_s < ema_t:
                score -= 40
                reasons.append("❌ EMA 空頭排列")
            elif price < ema_f and ema_f < ema_s:
                score -= 25
                reasons.append("❌ 短期空頭排列")

            # 價格相對 EMA200
            if pd.notna(ema_t):
                if price > ema_t:
                    score += 15
                    reasons.append("✅ 價格在長期均線之上 (多頭市場)")
                else:
                    score -= 15
                    reasons.append("❌ 價格在長期均線之下 (空頭市場)")

        # ADX 強度
        adx_val = self.current['ADX']
        if pd.notna(adx_val):
            if adx_val > 25:
                # 趨勢強,放大現有趨勢分數
                score = score * 1.2
                reasons.append(f"💪 ADX={adx_val:.1f} 強趨勢")
            elif adx_val < 20:
                # 盤整,縮小趨勢分數
                score = score * 0.6
                reasons.append(f"😴 ADX={adx_val:.1f} 盤整市")

        return max(-100, min(100, score)), reasons

    def _score_momentum(self):
        """動能評分 [-100, +100]"""
        score = 0
        reasons = []

        rsi_v = self.current['RSI']
        macd_v = self.current['MACD']
        macd_sig = self.current['MACD_signal']
        macd_h = self.current['MACD_hist']

        # RSI
        if pd.notna(rsi_v):
            if rsi_v > 70:
                score -= 20
                reasons.append(f"⚠️ RSI={rsi_v:.1f} 超買")
            elif rsi_v > 55:
                score += 25
                reasons.append(f"✅ RSI={rsi_v:.1f} 多頭動能")
            elif rsi_v < 30:
                score += 20
                reasons.append(f"✅ RSI={rsi_v:.1f} 超賣,反彈機率高")
            elif rsi_v < 45:
                score -= 25
                reasons.append(f"❌ RSI={rsi_v:.1f} 空頭動能")

        # MACD
        if pd.notna(macd_v) and pd.notna(macd_sig):
            if macd_v > macd_sig:
                score += 20
                reasons.append("✅ MACD 金叉之上")
            else:
                score -= 20
                reasons.append("❌ MACD 死叉之下")

            # 柱狀圖變化 (動能轉折)
            if len(self.df) >= 3:
                hist_diff = self.df['MACD_hist'].iloc[-1] - self.df['MACD_hist'].iloc[-3]
                if hist_diff > 0 and macd_h < 0:
                    score += 15
                    reasons.append("✅ MACD 柱由負轉正中 (動能回升)")
                elif hist_diff < 0 and macd_h > 0:
                    score -= 15
                    reasons.append("❌ MACD 柱由正轉負中 (動能衰退)")

        return max(-100, min(100, score)), reasons

    def _score_pattern(self):
        """K 線型態評分 [-100, +100]"""
        score = 0
        reasons = []
        recent = self.patterns.get('recent', [])

        # 只看最近 5 個型態,越近權重越大
        recent_tail = recent[-5:]
        n = len(recent_tail)
        for idx, p in enumerate(recent_tail):
            weight = (idx + 1) / n  # 0.2, 0.4, ..., 1.0
            if p['type'] == 'bullish':
                score += 30 * weight
                reasons.append(f"🟢 {p['name']} ({p['date']})")
            elif p['type'] == 'bearish':
                score -= 30 * weight
                reasons.append(f"🔴 {p['name']} ({p['date']})")
            else:
                reasons.append(f"🟡 {p['name']} ({p['date']})")

        # 布林通道位置
        bb_u = self.current.get('BB_upper')
        bb_l = self.current.get('BB_lower')
        bb_m = self.current.get('BB_mid')
        price = self.current['Close']
        if pd.notna(bb_u) and pd.notna(bb_l):
            if price > bb_u:
                score -= 10
                reasons.append("⚠️ 突破布林上軌 (短期超漲)")
            elif price < bb_l:
                score += 10
                reasons.append("✅ 跌破布林下軌 (短期超跌,反彈機率)")

        # 斐波那契關鍵位
        fib_key = self.fib.get('0.5')
        if fib_key:
            distance = abs(price - fib_key) / price
            if distance < 0.01:  # 在 0.5 回撤位附近
                reasons.append(f"📐 當前正測試斐波那契 0.5 位 (${fib_key:.2f})")

        return max(-100, min(100, score)), reasons

    def _score_volume(self):
        """量能評分 [-100, +100]"""
        score = 0
        reasons = []

        vol = self.current['Volume']
        vol_ma = self.current.get('Vol_MA20')

        if pd.notna(vol_ma) and vol_ma > 0:
            ratio = vol / vol_ma
            # 近期 5 日 vs 前期 15 日
            recent_vol = self.df['Volume'].tail(5).mean()
            prior_vol = self.df['Volume'].tail(20).head(15).mean()
            trend_ratio = recent_vol / prior_vol if prior_vol > 0 else 1

            # 判斷漲跌時量能特徵
            recent5 = self.df.tail(5)
            up_days = recent5[recent5['Close'] > recent5['Open']]
            down_days = recent5[recent5['Close'] < recent5['Open']]
            up_vol = up_days['Volume'].mean() if len(up_days) > 0 else 0
            down_vol = down_days['Volume'].mean() if len(down_days) > 0 else 0

            # 關鍵:漲時量 vs 跌時量
            if up_vol > down_vol * 1.3:
                score += 25
                reasons.append("✅ 漲時量 > 跌時量 (多頭特徵)")
            elif down_vol > up_vol * 1.3:
                score -= 25
                reasons.append("❌ 跌時量 > 漲時量 (空頭特徵)")

            # 回調是否縮量
            price_trend = self.df['Close'].iloc[-1] - self.df['Close'].iloc[-5]
            if price_trend < 0 and trend_ratio < 0.85:
                score += 15
                reasons.append("✅ 回調縮量 (健康回調,非主力出貨)")
            elif price_trend > 0 and trend_ratio > 1.2:
                score += 15
                reasons.append("✅ 上漲放量 (資金介入)")
            elif price_trend < 0 and trend_ratio > 1.3:
                score -= 20
                reasons.append("❌ 下跌放量 (主力出貨警訊)")

            if ratio > 2.0:
                reasons.append(f"🔥 當日爆量 {ratio:.1f}x MA20")

        return max(-100, min(100, score)), reasons

    def _vix_adjustment(self):
        """VIX 環境調整"""
        adj = 0
        reasons = []
        if self.vix is None:
            return 0, reasons

        if self.vix > 30:
            adj -= 15
            reasons.append(f"⚠️ VIX={self.vix:.1f} 高恐慌,做多需謹慎")
        elif self.vix > 25:
            adj -= 8
            reasons.append(f"⚠️ VIX={self.vix:.1f} 波動偏高")
        elif self.vix < 15:
            adj += 5
            reasons.append(f"✅ VIX={self.vix:.1f} 低波動環境")

        if self.vix_change > 5:
            adj -= 8
            reasons.append(f"⚠️ VIX 當日 +{self.vix_change:.1f}% 大漲,避險情緒升")
        elif self.vix_change < -5:
            adj += 5
            reasons.append(f"✅ VIX 當日 {self.vix_change:.1f}% 下降,風險情緒改善")

        return adj, reasons

    # ---------- 情境機率計算 ----------

    def _calculate_scenarios(self, overall_score, atr_val, current_price):
        """根據綜合評分計算三情境機率"""
        # 基準機率分配
        # overall_score: -100 ~ +100
        # 正分 -> 偏多; 負分 -> 偏空
        norm = (overall_score + 100) / 200  # 0~1

        # 反彈機率: 線性映射 + sigmoid 平滑
        rebound_prob = 0.15 + norm * 0.55  # 15%~70%
        # 續跌機率: 相反
        fall_prob = 0.15 + (1 - norm) * 0.55
        # 橫盤: 剩餘
        # 但當 ADX 低時,橫盤機率提升
        adx_val = self.current.get('ADX', 20)
        if pd.notna(adx_val) and adx_val < 20:
            sideways_bonus = 0.15
        else:
            sideways_bonus = 0.05

        total_raw = rebound_prob + fall_prob
        rebound_prob = rebound_prob / total_raw * (1 - sideways_bonus)
        fall_prob = fall_prob / total_raw * (1 - sideways_bonus)
        sideways_prob = sideways_bonus + (1 - rebound_prob - fall_prob - sideways_bonus)

        # 歸一化
        total = rebound_prob + fall_prob + sideways_prob
        rebound_prob /= total
        fall_prob /= total
        sideways_prob /= total

        # 價位計算
        atr_v = atr_val if pd.notna(atr_val) and atr_val > 0 else current_price * 0.02

        resistances = self.sr.get('resistance', [])
        supports = self.sr.get('support', [])

        # 反彈情境
        r1 = resistances[0] if resistances else current_price + atr_v * 2
        r2 = resistances[1] if len(resistances) >= 2 else current_price + atr_v * 3.5
        s1 = supports[0] if supports else current_price - atr_v * 1.5
        s2 = supports[1] if len(supports) >= 2 else current_price - atr_v * 3

        scenarios = [
            {
                'name': '情境 A: 反彈',
                'direction': 'up',
                'probability': rebound_prob,
                'trigger': f"守住 ${s1:.2f} 支撐,開盤不跌破",
                'target': f"反彈至 ${r1:.2f} - ${r2:.2f}",
                'reason': self._rebound_reason()
            },
            {
                'name': '情境 B: 橫盤整理',
                'direction': 'sideways',
                'probability': sideways_prob,
                'trigger': f"在 ${s1:.2f} - ${r1:.2f} 區間震盪",
                'target': f"區間整理,等方向選擇",
                'reason': self._sideways_reason()
            },
            {
                'name': '情境 C: 續跌',
                'direction': 'down',
                'probability': fall_prob,
                'trigger': f"跌破 ${s1:.2f} 且放量",
                'target': f"下探 ${s2:.2f} - ${supports[2] if len(supports)>=3 else s2-atr_v:.2f}",
                'reason': self._fall_reason()
            }
        ]

        # 按機率排序
        scenarios.sort(key=lambda x: -x['probability'])
        return scenarios

    def _rebound_reason(self):
        reasons = []
        rsi_v = self.current.get('RSI', 50)
        if rsi_v < 40:
            reasons.append("RSI 偏低有反彈空間")
        if self.vix and self.vix < 20:
            reasons.append("VIX 低波動環境支持")
        # 量能
        recent_vol = self.df['Volume'].tail(5).mean()
        prior_vol = self.df['Volume'].tail(20).head(15).mean()
        if prior_vol > 0 and recent_vol / prior_vol < 0.9:
            reasons.append("回調縮量,非主力出貨")
        # 斐波那契
        fib_05 = self.fib.get('0.5')
        if fib_05:
            dist = abs(self.current['Close'] - fib_05) / self.current['Close']
            if dist < 0.015:
                reasons.append("正測試 0.5 斐波那契關鍵位")
        return "、".join(reasons) if reasons else "技術面存在反彈條件"

    def _sideways_reason(self):
        adx_val = self.current.get('ADX', 20)
        reasons = []
        if pd.notna(adx_val) and adx_val < 20:
            reasons.append(f"ADX={adx_val:.1f} 無明顯趨勢")
        reasons.append("多空分歧,等待突破方向")
        return "、".join(reasons)

    def _fall_reason(self):
        reasons = []
        price = self.current['Close']
        ema_t = self.current.get('EMA_trend')
        if pd.notna(ema_t) and price < ema_t:
            reasons.append("價格跌破長期均線")
        rsi_v = self.current.get('RSI', 50)
        if rsi_v < 45 and rsi_v > 30:
            reasons.append("RSI 偏弱未超賣")
        if self.vix and self.vix > 25:
            reasons.append(f"VIX={self.vix:.1f} 避險情緒升溫")
        # MACD 死叉
        macd_v = self.current.get('MACD')
        macd_s = self.current.get('MACD_signal')
        if pd.notna(macd_v) and pd.notna(macd_s) and macd_v < macd_s:
            reasons.append("MACD 死叉")
        return "、".join(reasons) if reasons else "空頭結構未改善"

    # ---------- 交易計劃 ----------

    def _build_trade_plan(self, overall_score, scenarios, current_price, atr_val):
        """根據綜合評分產出可執行交易計劃"""
        atr_v = atr_val if pd.notna(atr_val) and atr_val > 0 else current_price * 0.02
        supports = self.sr.get('support', [])
        resistances = self.sr.get('resistance', [])

        s1 = supports[0] if supports else current_price - atr_v * 1.5
        s2 = supports[1] if len(supports) >= 2 else s1 - atr_v
        r1 = resistances[0] if resistances else current_price + atr_v * 2
        r2 = resistances[1] if len(resistances) >= 2 else r1 + atr_v * 1.5

        if overall_score > 20:
            # 做多
            entry_high = current_price * 1.002
            entry_low = max(s1, current_price * 0.99)
            stop_loss = s1 * 0.995 if supports else current_price - atr_v * 1.5
            target_1 = r1
            target_2 = r2
            risk = current_price - stop_loss
            reward = target_1 - current_price
            rr = reward / risk if risk > 0 else 0

            # 倉位依強度調整
            if overall_score > 60:
                pos = "滿倉 (1.0x)"
            elif overall_score > 40:
                pos = "標準倉 (0.7x)"
            else:
                pos = "輕倉 (0.4x)"

        elif overall_score < -20:
            # 做空(或避開)
            entry_high = min(r1, current_price * 1.01)
            entry_low = current_price * 0.998
            stop_loss = r1 * 1.005 if resistances else current_price + atr_v * 1.5
            target_1 = s1
            target_2 = s2
            risk = stop_loss - current_price
            reward = current_price - target_1
            rr = reward / risk if risk > 0 else 0
            pos = "空倉觀望" if overall_score > -40 else "輕倉做空 (0.3x)"
        else:
            # 中性
            entry_high = current_price * 1.005
            entry_low = current_price * 0.995
            stop_loss = s1 if supports else current_price - atr_v
            target_1 = r1
            target_2 = r2
            risk = current_price - stop_loss if supports else atr_v
            reward = target_1 - current_price
            rr = reward / risk if risk > 0 else 0
            pos = "觀望為主,等明確訊號"

        return {
            'entry_low': entry_low,
            'entry_high': entry_high,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'rr_ratio': rr,
            'position_size': pos
        }

    # ---------- 外部因素 ----------

    def _external_factors(self):
        factors = []
        if self.vix is not None:
            if self.vix > 25:
                factors.append(f"⚠️ VIX={self.vix:.1f} 恐慌情緒明顯 (建議減倉)")
            elif self.vix > 20:
                factors.append(f"⚠️ VIX={self.vix:.1f} 波動偏高,需留意大盤拖累")
            else:
                factors.append(f"✅ VIX={self.vix:.1f} 安全區,風險資產有利")
            if self.vix_change > 3:
                factors.append(f"📈 VIX 當日 +{self.vix_change:.1f}% 需警惕避險盤")

        # 價格距離重要均線
        price = self.current['Close']
        ema_t = self.current.get('EMA_trend')
        if pd.notna(ema_t):
            dist = (price / ema_t - 1) * 100
            if abs(dist) < 1:
                factors.append(f"📐 價格正在 EMA200 附近 (${ema_t:.2f}),關鍵多空分水嶺")
            elif dist > 15:
                factors.append(f"⚠️ 價格高於 EMA200 {dist:.1f}% (偏離較大,回檔風險)")
            elif dist < -10:
                factors.append(f"✅ 價格低於 EMA200 {abs(dist):.1f}% (超跌反彈機會)")

        # 財報日提醒 (無 API 情況下提示檢查)
        factors.append("📅 請自行確認近期是否有財報 / 重大事件 (未發布則波動放大)")

        return factors

    # ---------- 主預測 ----------

    def predict(self):
        # 子評分
        trend_score, trend_reasons = self._score_trend()
        momentum_score, momentum_reasons = self._score_momentum()
        pattern_score, pattern_reasons = self._score_pattern()
        volume_score, volume_reasons = self._score_volume()
        vix_adj, vix_reasons = self._vix_adjustment()

        # 加權綜合
        w = self.weights
        raw_score = (
            trend_score * w['trend'] +
            momentum_score * w['momentum'] +
            pattern_score * w['pattern'] +
            volume_score * w['volume']
        )
        overall_score = raw_score + vix_adj
        overall_score = max(-100, min(100, overall_score))

        # 轉 0-100 便於顯示
        display_score = (overall_score + 100) / 2

        # 信號
        if overall_score > 50:
            signal = 'STRONG_BUY'
        elif overall_score > 20:
            signal = 'BUY'
        elif overall_score > -20:
            signal = 'HOLD'
        elif overall_score > -50:
            signal = 'SELL'
        else:
            signal = 'STRONG_SELL'

        # 置信度: 各子分一致性越高 = 越有信心
        sub_scores = [trend_score, momentum_score, pattern_score, volume_score]
        std = np.std(sub_scores)
        confidence = max(0.3, min(0.95, 1 - std / 100))

        # 情境
        current_price = self.current['Close']
        atr_val = self.current['ATR']
        scenarios = self._calculate_scenarios(overall_score, atr_val, current_price)

        # 交易計劃
        trade_plan = self._build_trade_plan(overall_score, scenarios, current_price, atr_val)

        # 外部因素
        external = self._external_factors()

        # 綜合判斷文字
        top_scenario = scenarios[0]
        summary = self._build_summary(signal, top_scenario, overall_score,
                                       trade_plan, current_price)

        return {
            'overall': {
                'signal': signal,
                'score': display_score,
                'raw_score': overall_score,
                'confidence': confidence
            },
            'sub_scores': {
                'trend': trend_score,
                'momentum': momentum_score,
                'pattern': pattern_score,
                'volume': volume_score,
                'vix_adj': vix_adj
            },
            'reasons': {
                'trend': trend_reasons,
                'momentum': momentum_reasons,
                'pattern': pattern_reasons,
                'volume': volume_reasons,
                'vix': vix_reasons
            },
            'scenarios': scenarios,
            'trade_plan': trade_plan,
            'external_factors': external,
            'summary': summary
        }

    def _build_summary(self, signal, top_scenario, score, trade_plan, price):
        """產出人類化綜合判斷"""
        direction_map = {
            'STRONG_BUY': '強烈看多',
            'BUY': '短線偏多',
            'HOLD': '中性觀望',
            'SELL': '短線偏空',
            'STRONG_SELL': '強烈看空'
        }
        direction = direction_map.get(signal, '中性')

        stop = trade_plan['stop_loss']
        target = trade_plan['target_1']

        summary = f"我的判斷:{direction} (綜合分 {score:+.1f})。"
        summary += f"最可能情境為「{top_scenario['name']}」(機率 {top_scenario['probability']:.0%})。"

        if signal in ['STRONG_BUY', 'BUY']:
            summary += f"建議做多進場於 ${price:.2f} 附近,嚴守 ${stop:.2f} 停損,目標 ${target:.2f}。"
            summary += f"若跌破停損需重新評估型態是否失效。"
        elif signal in ['STRONG_SELL', 'SELL']:
            summary += f"建議避開多頭或輕倉做空,關注 ${target:.2f} 支撐反應。"
            summary += f"若放量跌破支撐則型態確認,續空。"
        else:
            summary += f"多空動能相近,建議等待明確突破 (${trade_plan['target_1']:.2f} 或 ${trade_plan['stop_loss']:.2f}) 後再操作。"

        return summary
