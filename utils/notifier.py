"""
Telegram notification utility.
Sends trade result messages for wins and losses.
"""
from __future__ import annotations

import os
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

_BOT_TOKEN = ""
_CHAT_ID = ""


def init_notifier():
    global _BOT_TOKEN, _CHAT_ID
    _BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    if _BOT_TOKEN and _CHAT_ID:
        logger.info("Telegram notifier enabled")
    else:
        logger.warning("Telegram not configured – notifications disabled")


def _send(text: str):
    if not _BOT_TOKEN or not _CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        if not resp.ok:
            logger.warning(f"Telegram send failed: {resp.text}")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


def notify_trade_result(
    direction: str,
    symbol: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    pnl_usdt: float,
    capital_usdt: float,
    sl: float,
    tp: float,
):
    is_win = pnl_usdt >= 0
    emoji = "✅ 盈利" if is_win else "❌ 虧損"
    direction_label = "做多 LONG" if direction == "long" else "做空 SHORT"
    pnl_sign = "+" if pnl_usdt >= 0 else ""

    msg = (
        f"{emoji}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"幣種：<b>{symbol}</b>\n"
        f"方向：{direction_label}\n"
        f"進場價：{entry_price:.4f}\n"
        f"出場價：{exit_price:.4f}\n"
        f"止損：{sl:.4f} | 止盈：{tp:.4f}\n"
        f"數量：{qty:.6f}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"盈虧：<b>{pnl_sign}{pnl_usdt:.4f} USDT</b>\n"
        f"現有資金：{capital_usdt:.4f} USDT\n"
    )
    _send(msg)
    logger.info(f"Telegram notified: {emoji} {pnl_sign}{pnl_usdt:.4f} USDT")


def notify_risk_event(event: str):
    """Send risk management alerts (pause, survival mode, monthly target)."""
    _send(f"⚠️ 風控提醒\n{event}")


def notify_bot_start(symbol: str, capital: float, dry_run: bool):
    mode = "【模擬模式】" if dry_run else "【正式交易】"
    _send(
        f"🤖 交易機器人啟動 {mode}\n"
        f"幣種：{symbol}\n"
        f"資金：{capital:.2f} USDT"
    )
