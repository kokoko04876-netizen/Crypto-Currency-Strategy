#!/usr/bin/env python3
"""
ICT Silver Bullet Automated Trading Bot
Full 7-element A-grade signal system.

Usage:
  python main.py            # live trading
  python main.py --dry-run  # simulate, no real orders
  python main.py --status   # print account state and exit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

from exchange.client import ExchangeClient
from strategy.ict_signals import ICTSignalDetector, SignalGrade
from strategy.news_filter import create_sample_events_file, is_blocked_by_news
from strategy.position_calculator import PositionCalculator
from strategy.risk_manager import RiskManager
from utils.logger import get_logger, setup_logging
from utils.notifier import init_notifier, notify_bot_start, notify_risk_event, notify_trade_result
from utils.time_utils import SessionManager


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


logger = None

SCAN_INTERVAL = 60
OFF_HOURS_SLEEP = 30
POS_MONITOR_INTERVAL = 10


class TradingBot:
    def __init__(self, config: dict, dry_run: bool = False):
        self.cfg = config
        self.dry_run = dry_run
        self.exchange = ExchangeClient(config)
        self.session = SessionManager(config)
        self.risk = RiskManager(config)
        self.signals = ICTSignalDetector(config)
        self.calc = PositionCalculator(config)
        self.tz = ZoneInfo(config["sessions"].get("timezone", "Asia/Taipei"))
        self.smt_symbol = config["trading"].get("smt_symbol", "ETH/USDT:USDT")

        create_sample_events_file(config["news"]["events_file"])

        bal = self._safe_balance()
        if bal > 0:
            self.calc.update_capital(bal)
            self.risk.state.capital_usdt = bal
            self.risk.save_state()

        self._maybe_monthly_reset()

    def _safe_balance(self) -> float:
        try:
            return self.exchange.get_balance()
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")
            return 0.0

    def _maybe_monthly_reset(self):
        if date.today().day == 1 and self.risk.state.monthly_trades > 0:
            self.risk.monthly_reset()
            logger.info("Monthly stats reset")

    def _fetch_all_data(self) -> tuple:
        lookback = self.cfg["signals"]["lookback_candles"]
        fetch = self.exchange.with_retry
        df_htf   = fetch(lambda: self.exchange.fetch_ohlcv(self.cfg["signals"]["htf_timeframe"], lookback))
        df_ltf   = fetch(lambda: self.exchange.fetch_ohlcv(self.cfg["signals"]["ltf_timeframe"], lookback))
        df_4h    = fetch(lambda: self.exchange.fetch_ohlcv(self.cfg["htf_bias"]["timeframe_4h"], self.cfg["htf_bias"]["lookback"]))
        df_daily = fetch(lambda: self.exchange.fetch_ohlcv(self.cfg["htf_bias"]["timeframe_daily"], 30))
        df_eth = None
        if self.cfg.get("smt", {}).get("enabled", False):
            try:
                orig = self.exchange.symbol
                self.exchange.symbol = self.smt_symbol
                df_eth = fetch(lambda: self.exchange.fetch_ohlcv(self.cfg["signals"]["htf_timeframe"], lookback))
                self.exchange.symbol = orig
            except Exception as e:
                logger.warning(f"ETH fetch failed (SMT skipped): {e}")
        return df_htf, df_ltf, df_4h, df_daily, df_eth

    def _execute_entry(self, signal, current_capital: float):
        direction = signal.direction.value
        plan = self.calc.calculate(
            symbol=self.exchange.symbol,
            direction=direction,
            entry_price=signal.entry_price,
            current_capital=current_capital,
        )
        min_rr = self.cfg["trading"].get("min_rr_ratio", 2.5)
        if plan.true_rr < min_rr:
            logger.warning(f"RR {plan.true_rr:.2f} < min {min_rr} — skipping entry")
            return None, None
        if self.dry_run:
            logger.info(f"[DRY-RUN] {direction.upper()} entry")
            logger.info(plan.summary())
            return None, plan
        self.exchange.set_leverage(plan.leverage)
        side = "buy" if direction == "long" else "sell"
        entry_order = self.exchange.with_retry(lambda: self.exchange.place_market_order(side, plan.qty))
        self.exchange.place_stop_loss(side, plan.qty, plan.stop_loss)
        self.exchange.place_take_profit(side, plan.qty, plan.take_profit)
        logger.info(f"ENTRY {direction.upper()} {plan.qty:.6f} @ ~{signal.entry_price:.4f} | SL={plan.stop_loss:.4f} TP={plan.take_profit:.4f}")
        return entry_order, plan

    def _monitor_position(self, plan):
        logger.info("Monitoring open position...")
        while True:
            time.sleep(POS_MONITOR_INTERVAL)
            if self.session.is_after_force_close():
                logger.warning("Force-close time — closing position")
                self.exchange.close_all_positions()
                notify_risk_event("04:00 強制平倉執行")
                break
            if not self.exchange.get_positions():
                logger.info("Position closed (SL/TP hit)")
                break
        new_bal = self._safe_balance()
        pnl = new_bal - plan.capital_usdt
        exit_price = plan.take_profit if pnl >= 0 else plan.stop_loss
        notify_trade_result(
            direction=plan.direction, symbol=plan.symbol,
            entry_price=plan.entry_price, exit_price=exit_price,
            qty=plan.qty, pnl_usdt=pnl, capital_usdt=new_bal,
            sl=plan.stop_loss, tp=plan.take_profit,
        )
        self.risk.record_trade(pnl)
        self.calc.update_capital(new_bal)
        logger.info(f"Trade closed | PnL ≈ {pnl:+.4f} USDT | Balance = {new_bal:.4f} USDT")
        return pnl

    def _silver_bullet_loop(self):
        logger.info("▶ Silver Bullet 22:00–00:00 | scanning...")
        while self.session.is_silver_bullet():
            can, reason = self.risk.can_trade()
            if not can:
                logger.info(f"Cannot trade: {reason}")
                notify_risk_event(f"無法交易：{reason}")
                return
            blocked, news_reason = is_blocked_by_news(self.cfg["news"], self.tz)
            if blocked:
                logger.warning(news_reason)
                notify_risk_event(f"消息面封鎖：{news_reason}")
                time.sleep(SCAN_INTERVAL)
                continue
            try:
                df_htf, df_ltf, df_4h, df_daily, df_eth = self._fetch_all_data()
            except Exception as e:
                logger.error(f"Data fetch error: {e}")
                time.sleep(SCAN_INTERVAL)
                continue
            signal = self.signals.evaluate(df_htf, df_ltf, df_4h, df_daily, df_eth)
            logger.info(f"Signal: [{signal.grade.value}] {signal.reason}")
            if signal.grade == SignalGrade.A:
                bal = self._safe_balance() or self.risk.state.capital_usdt
                entry_order, plan = self._execute_entry(signal, bal)
                if plan is None:
                    time.sleep(SCAN_INTERVAL)
                    continue
                if self.dry_run:
                    sim_pnl = plan.expected_profit_net
                    self.risk.record_trade(sim_pnl)
                    notify_trade_result(
                        direction=plan.direction, symbol=plan.symbol,
                        entry_price=plan.entry_price, exit_price=plan.take_profit,
                        qty=plan.qty, pnl_usdt=sim_pnl,
                        capital_usdt=self.risk.state.capital_usdt,
                        sl=plan.stop_loss, tp=plan.take_profit,
                    )
                    logger.info("[DRY-RUN] Simulated trade complete")
                    return
                self._monitor_position(plan)
                return
            time.sleep(SCAN_INTERVAL)
        logger.info("◀ Silver Bullet window closed — no trade taken")

    def run(self):
        logger.info("=" * 52)
        logger.info("  ICT Silver Bullet Bot — STARTED")
        if self.dry_run:
            logger.info("  *** DRY-RUN — no real orders ***")
        logger.info("=" * 52)
        logger.info(self.risk.status_summary())
        notify_bot_start(self.exchange.symbol, self.risk.state.capital_usdt, self.dry_run)

        while True:
            try:
                self.session.log_status()
                session = self.session.current_session()
                if self.session.is_force_close():
                    logger.warning("04:00 force-close — closing all positions")
                    if not self.dry_run:
                        self.exchange.close_all_positions()
                    notify_risk_event("04:00 強制平倉")
                    time.sleep(60)
                    continue
                if session == "pre_market":
                    logger.info("Layer 1: Pre-market (20:00–20:30) — preparation only")
                    time.sleep(OFF_HOURS_SLEEP)
                elif session == "mid_session":
                    logger.info("Layer 2: Mid-session (20:30–22:00) — observe structure & liquidity")
                    time.sleep(OFF_HOURS_SLEEP)
                elif session == "silver_bullet":
                    self._silver_bullet_loop()
                else:
                    secs = self.session.seconds_until_silver_bullet()
                    if secs > 120:
                        logger.info(f"Off-hours — Silver Bullet in {secs/3600:.1f}h")
                        time.sleep(min(secs - 60, 3600))
                    else:
                        time.sleep(OFF_HOURS_SLEEP)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="ICT Silver Bullet Bot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--config",  default="config.yaml")
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
