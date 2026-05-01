#!/usr/bin/env python3
"""
ICT Silver Bullet Automated Trading Bot
Strategy: A-Grade signals only (OB + FVG + OTE 0.705 + LTF trigger)

Usage:
  python main.py            # run live trading
  python main.py --status   # show bot state and exit
  python main.py --dry-run  # simulate without placing real orders
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date

import yaml
from dotenv import load_dotenv

from exchange.client import ExchangeClient
from strategy.ict_signals import ICTSignalDetector, SignalGrade
from strategy.position_calculator import PositionCalculator
from strategy.risk_manager import RiskManager
from utils.logger import get_logger, setup_logging
from utils.notifier import init_notifier, notify_bot_start, notify_risk_event, notify_trade_result
from utils.time_utils import SessionManager

# ── Bootstrap ─────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def bootstrap() -> dict:
    load_dotenv()
    cfg = load_config()
    log_cfg = cfg["logging"]
    setup_logging(
        log_file=log_cfg["file"],
        level_str=log_cfg["level"],
        max_bytes=log_cfg["max_bytes"],
        backup_count=log_cfg["backup_count"],
    )
    init_notifier()
    return cfg


logger = None   # initialized after bootstrap


# ── Bot core ──────────────────────────────────────────────────────────────────

SCAN_INTERVAL_SECONDS = 60         # check for signals every 60 s inside the window
OFF_HOURS_SLEEP = 30               # poll interval outside trading window
OPEN_POSITION_CHECK_INTERVAL = 10  # monitor open positions every 10 s


class TradingBot:
    def __init__(self, config: dict, dry_run: bool = False):
        self.cfg = config
        self.dry_run = dry_run
        self.exchange = ExchangeClient(config)
        self.session = SessionManager(config)
        self.risk = RiskManager(config)
        self.signals = ICTSignalDetector(config)
        self.calc = PositionCalculator(config)

        # Sync position calculator with live capital on startup
        bal = self._safe_balance()
        if bal > 0:
            self.calc.update_capital(bal)
            self.risk.state.capital_usdt = bal
            self.risk.save_state()

        # Monthly reset if needed
        self._maybe_monthly_reset()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _safe_balance(self) -> float:
        try:
            return self.exchange.get_balance()
        except Exception as e:
            logger.warning(f"Could not fetch balance: {e}")
            return 0.0

    def _maybe_monthly_reset(self):
        today = date.today()
        if today.day == 1 and self.risk.state.monthly_trades > 0:
            self.risk.monthly_reset()
            logger.info("New month – monthly stats reset")

    def _fetch_market_data(self) -> tuple:
        htf = self.cfg["signals"]["htf_timeframe"]
        ltf = self.cfg["signals"]["ltf_timeframe"]
        lookback = self.cfg["signals"]["lookback_candles"]
        df_htf = self.exchange.with_retry(
            lambda: self.exchange.fetch_ohlcv(htf, lookback)
        )
        df_ltf = self.exchange.with_retry(
            lambda: self.exchange.fetch_ohlcv(ltf, lookback)
        )
        return df_htf, df_ltf

    # ── Entry execution ───────────────────────────────────────────────────────

    def _execute_entry(self, signal, current_capital: float):
        direction = signal.direction.value   # "long" or "short"
        plan = self.calc.calculate(
            symbol=self.exchange.symbol,
            direction=direction,
            entry_price=signal.entry_price,
            current_capital=current_capital,
        )

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would place {direction.upper()} entry")
            logger.info(plan.summary())
            return None, plan

        # Set leverage
        self.exchange.set_leverage(plan.leverage)

        # Market entry
        side = "buy" if direction == "long" else "sell"
        entry_order = self.exchange.with_retry(
            lambda: self.exchange.place_market_order(side, plan.qty)
        )

        # Place SL and TP bracket
        self.exchange.place_stop_loss(side, plan.qty, plan.stop_loss)
        self.exchange.place_take_profit(side, plan.qty, plan.take_profit)

        logger.info(
            f"ENTRY: {direction.upper()} {plan.qty:.6f} @ ~{signal.entry_price:.4f} | "
            f"SL={plan.stop_loss:.4f} TP={plan.take_profit:.4f}"
        )
        return entry_order, plan

    # ── Position monitor loop ─────────────────────────────────────────────────

    def _monitor_position(self, plan):
        """Wait for position to close (SL or TP hit), record trade, send notification."""
        logger.info("Monitoring open position...")
        while True:
            time.sleep(OPEN_POSITION_CHECK_INTERVAL)

            # Force close time
            if self.session.is_after_force_close():
                logger.warning("Force-close time reached – closing position")
                self.exchange.close_all_positions()
                break

            positions = self.exchange.get_positions()
            if not positions:
                logger.info("Position closed (SL/TP triggered)")
                break

            pos = positions[0]
            unrealized = float(pos.get("unrealizedPnl", 0))
            logger.debug(f"Position open | unrealized PnL: {unrealized:.4f} USDT")

        # Calculate realized PnL (balance diff)
        new_balance = self._safe_balance()
        realized_pnl = new_balance - plan.capital_usdt
        logger.info(f"Realized PnL ≈ {realized_pnl:+.4f} USDT")

        # Determine approximate exit price from PnL
        if realized_pnl >= 0:
            exit_price = plan.take_profit
        else:
            exit_price = plan.stop_loss

        # ── Telegram notification (win or loss) ───────────────────────────────
        notify_trade_result(
            direction=plan.direction,
            symbol=plan.symbol,
            entry_price=plan.entry_price,
            exit_price=exit_price,
            qty=plan.qty,
            pnl_usdt=realized_pnl,
            capital_usdt=new_balance,
            sl=plan.stop_loss,
            tp=plan.take_profit,
        )

        self.risk.record_trade(realized_pnl)
        self.calc.update_capital(new_balance)
        return realized_pnl

    # ── Silver Bullet scanning loop ───────────────────────────────────────────

    def _silver_bullet_loop(self):
        logger.info("Silver Bullet window open – scanning for A-grade signals")

        while self.session.is_silver_bullet():
            # Risk check
            can, reason = self.risk.can_trade()
            if not can:
                logger.info(f"Cannot trade: {reason}")
                notify_risk_event(f"無法交易：{reason}")
                return

            # Market data
            try:
                df_htf, df_ltf = self._fetch_market_data()
            except Exception as e:
                logger.error(f"Failed to fetch market data: {e}")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # Signal evaluation
            signal = self.signals.evaluate(df_htf, df_ltf)
            logger.info(f"Signal: {signal.grade.value} | {signal.reason}")

            if signal.grade == SignalGrade.A:
                current_balance = self._safe_balance() or self.risk.state.capital_usdt
                entry_order, plan = self._execute_entry(signal, current_balance)

                if self.dry_run:
                    logger.info("[DRY-RUN] Simulating trade close")
                    sim_pnl = plan.expected_profit_net
                    self.risk.record_trade(sim_pnl)
                    notify_trade_result(
                        direction=plan.direction,
                        symbol=plan.symbol,
                        entry_price=plan.entry_price,
                        exit_price=plan.take_profit,
                        qty=plan.qty,
                        pnl_usdt=sim_pnl,
                        capital_usdt=self.risk.state.capital_usdt,
                        sl=plan.stop_loss,
                        tp=plan.take_profit,
                    )
                    return

                # Block until position closes
                self._monitor_position(plan)
                return          # max 1 trade per Silver Bullet window

            # No signal yet – wait and scan again
            time.sleep(SCAN_INTERVAL_SECONDS)

        logger.info("Silver Bullet window ended – no trade taken")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger.info("=" * 50)
        logger.info(" ICT Silver Bullet Bot  STARTED")
        if self.dry_run:
            logger.info(" *** DRY-RUN MODE – no real orders ***")
        logger.info("=" * 50)
        logger.info(self.risk.status_summary())

        notify_bot_start(
            symbol=self.exchange.symbol,
            capital=self.risk.state.capital_usdt,
            dry_run=self.dry_run,
        )

        while True:
            try:
                self.session.log_status()
                session = self.session.current_session()

                # Force close at configured time
                if self.session.is_force_close():
                    logger.warning("Force-close time – closing all positions")
                    if not self.dry_run:
                        self.exchange.close_all_positions()
                    notify_risk_event("04:00 強制平倉執行")
                    time.sleep(60)
                    continue

                if session == "pre_market":
                    logger.info("Pre-market window (20:00–21:30) – preparation only")
                    time.sleep(OFF_HOURS_SLEEP)

                elif session == "macro_window":
                    logger.info("Macro window (21:30–22:00) – observation only, no entry")
                    time.sleep(OFF_HOURS_SLEEP)

                elif session == "silver_bullet":
                    self._silver_bullet_loop()

                else:
                    # Off-hours – sleep until next Silver Bullet
                    secs = self.session.seconds_until_silver_bullet()
                    if secs > 60:
                        logger.info(
                            f"Off-hours – next Silver Bullet in {secs/3600:.1f}h"
                        )
                        time.sleep(min(secs - 60, 3600))
                    else:
                        time.sleep(OFF_HOURS_SLEEP)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(30)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICT Silver Bullet Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without real orders")
    parser.add_argument("--status", action="store_true", help="Print bot status and exit")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = bootstrap()
    global logger
    logger = get_logger("main")

    bot = TradingBot(cfg, dry_run=args.dry_run)

    if args.status:
        print(bot.risk.status_summary())
        sys.exit(0)

    bot.run()


if __name__ == "__main__":
    main()
