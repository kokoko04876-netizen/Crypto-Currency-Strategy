"""
Exchange client wrapping ccxt for futures trading.
Supports Binance, Bybit, OKX (set EXCHANGE_NAME in .env).
"""
import os
import time
import ccxt
import pandas as pd
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_exchange(name: str, api_key: str, api_secret: str, passphrase: str) -> ccxt.Exchange:
    common_opts = {
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    exchanges = {
        "binance": ccxt.binance,
        "bybit": ccxt.bybit,
        "okx": ccxt.okx,
    }
    if name not in exchanges:
        raise ValueError(f"Unsupported exchange: {name}. Choose from {list(exchanges.keys())}")
    if name == "okx" and passphrase:
        common_opts["password"] = passphrase
    return exchanges[name](common_opts)


class ExchangeClient:
    def __init__(self, config: dict):
        self.cfg = config["trading"]
        name = os.getenv("EXCHANGE_NAME", "binance").lower()
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        passphrase = os.getenv("PASSPHRASE", "")
        self.symbol = os.getenv("SYMBOL", self.cfg["symbol"])
        self.exchange = _build_exchange(name, api_key, api_secret, passphrase)
        logger.info(f"Exchange: {name.upper()} | Symbol: {self.symbol}")

    # ── Market data ──────────────────────────────────────────────────────────

    def fetch_ohlcv(self, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Return OHLCV DataFrame with columns [open, high, low, close, volume]."""
        raw = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    def fetch_ticker(self) -> dict:
        return self.exchange.fetch_ticker(self.symbol)

    def get_current_price(self) -> float:
        ticker = self.fetch_ticker()
        return float(ticker["last"])

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Return available USDT balance."""
        balance = self.exchange.fetch_balance()
        usdt = balance.get("USDT", {}).get("free", 0.0)
        return float(usdt)

    def get_positions(self) -> list:
        """Return open positions for the configured symbol."""
        positions = self.exchange.fetch_positions([self.symbol])
        return [p for p in positions if float(p.get("contracts", 0)) != 0]

    # ── Orders ────────────────────────────────────────────────────────────────

    def set_leverage(self, leverage: int):
        try:
            self.exchange.set_leverage(leverage, self.symbol)
            logger.info(f"Leverage set to {leverage}x")
        except Exception as e:
            logger.warning(f"set_leverage failed (may already be set): {e}")

    def place_market_order(self, side: str, amount: float, params: dict = None) -> dict:
        """
        Place a market order.
        side: 'buy' (long) or 'sell' (short)
        amount: quantity in base asset units
        """
        params = params or {}
        logger.info(f"Placing MARKET {side.upper()} | qty={amount} | {self.symbol}")
        order = self.exchange.create_market_order(self.symbol, side, amount, params=params)
        logger.info(f"Order filled: {order.get('id')} @ {order.get('average')}")
        return order

    def place_limit_order(self, side: str, amount: float, price: float, params: dict = None) -> dict:
        params = params or {}
        logger.info(f"Placing LIMIT {side.upper()} | qty={amount} @ {price} | {self.symbol}")
        order = self.exchange.create_limit_order(self.symbol, side, amount, price, params=params)
        return order

    def place_stop_loss(self, side: str, amount: float, stop_price: float) -> Optional[dict]:
        """Place a stop-loss order (exchange-native stop market)."""
        close_side = "sell" if side == "buy" else "buy"
        params = {
            "stopPrice": stop_price,
            "reduceOnly": True,
        }
        try:
            order = self.exchange.create_order(
                self.symbol, "stop_market", close_side, amount, params=params
            )
            logger.info(f"Stop-loss placed @ {stop_price}")
            return order
        except Exception as e:
            logger.error(f"Stop-loss order failed: {e}")
            return None

    def place_take_profit(self, side: str, amount: float, tp_price: float) -> Optional[dict]:
        """Place a take-profit order (exchange-native take profit market)."""
        close_side = "sell" if side == "buy" else "buy"
        params = {
            "stopPrice": tp_price,
            "reduceOnly": True,
        }
        try:
            order = self.exchange.create_order(
                self.symbol, "take_profit_market", close_side, amount, params=params
            )
            logger.info(f"Take-profit placed @ {tp_price}")
            return order
        except Exception as e:
            logger.error(f"Take-profit order failed: {e}")
            return None

    def cancel_all_orders(self):
        try:
            self.exchange.cancel_all_orders(self.symbol)
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.warning(f"cancel_all_orders: {e}")

    def close_position(self, position: dict):
        """Market-close an open position."""
        side = position.get("side", "")
        amount = abs(float(position.get("contracts", 0)))
        if amount == 0:
            return
        close_side = "sell" if side == "long" else "buy"
        params = {"reduceOnly": True}
        self.cancel_all_orders()
        self.place_market_order(close_side, amount, params=params)
        logger.info(f"Position closed: {side} {amount} {self.symbol}")

    def close_all_positions(self):
        for pos in self.get_positions():
            self.close_position(pos)

    # ── Retry helper ──────────────────────────────────────────────────────────

    def with_retry(self, fn, retries: int = 3, delay: float = 2.0):
        for attempt in range(retries):
            try:
                return fn()
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Network error ({e}), retry {attempt+1}/{retries}")
                time.sleep(delay * (2 ** attempt))
