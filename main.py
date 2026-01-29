# -*- coding: utf-8 -*-
import os
import argparse
import logging
import sys
import time
import json
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import get_config, Config
from storage import get_db
from data_provider import DataFetcherManager
from analyzer import GeminiAnalyzer, AnalysisResult
from notification import NotificationService

# ================= 日志配置 =================

LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_SECTOR = "Macro"

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"stock_analysis_{today_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# ================= Pipeline =================

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()

        env_workers = os.getenv("MAX_CONCURRENT")
        self.max_workers = int(env_workers) if env_workers else (max_workers or self.config.max_workers or 1)

        self.portfolio = self._load_portfolio_config()
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()

        logger.info(f"AI-CIO 初始化完成 | 并发数={self.max_workers}")

    # ---------- 配置 ----------

    def _load_portfolio_config(self) -> dict:
        path = "portfolio.json"
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # === 兼容性修复 ===
                # 如果是列表（由 Telegram Bot 修改），转换为字典格式
                if isinstance(data, list):
                    return {str(code): {"code": str(code), "name": f"股票{str(code)}", "sector": "Unknown"} for code in data}
                
                return data
        except Exception as e:
            logger.error(f"加载 portfolio.json 失败: {e}")
            return {}

    # ---------- 新闻上下文 ----------

    def _get_trend_radar_context(self, code: str, json_path: str = "news_summary.json") -> dict:
        context = {"macro": "", "sector": "", "target_sector": DEFAULT_SECTOR}
        stock_info = self.portfolio.get(code, {})
        target_sector = stock_info.get("sector", DEFAULT_SECTOR)
        context["target_sector"] = target_sector

        if not os.path.exists(json_path):
            return context

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                news_items = json.load(f)

            macro_news, sector_news = [], []
            for item in news_items:
                cat = item.get("category", "Macro")
                line = f"- {item.get('title', '')}: {item.get('summary', '')}"
                if cat in ["Macro", "Finance"]:
                    macro_news.append(line)
                if cat == target_sector:
                    sector_news.append(line)

            context["macro"] = "\n".join(macro_news) if macro_news else "当前宏观面平静。"
            context["sector"] = (
                "\n".join(sector_news) if sector_news else f"当前{target_sector}板块无重大消息。"
            )
            return context
        except Exception as e:
            logger.warning(f"读取新闻上下文失败: {e}")
            return context

    # ---------- 技术指标 ----------

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}

        try:
            df = df.sort_values("date")
            close = df["close"]

            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]

            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd = (exp12 - exp26).iloc[-1]
            signal = (exp12 - exp26).ewm(span=9, adjust=False).mean().iloc[-1]

            return {
                "price": close.iloc[-1],
                "ma5": ma5,
                "ma20": ma20,
                "ma60": ma60,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": signal,
                "support": df["low"].tail(20).min(),
                "resistance": df["high"].tail(20).max(),
            }
        except Exception as e:
            logger.warning(f"技术指标计算失败: {e}")
            return {}

    # ---------- 数据抓取 ----------

    def fetch_and_save_stock_data(self, code: str) -> Optional[pd.DataFrame]:
        try:
            df, source = self.fetcher_manager.get_daily_data(code, days=100)
            if df is not None and not df.empty:
                self.db.save_daily_data(df, code, source)
                return df
            return None
        except Exception as e:
            logger.error(f"[{code}] 行情抓取失败: {e}")
            return None

    # ---------- 单股处理 ----------

    def process_single_stock(self, code: str, dry_run: bool = False) -> Optional[AnalysisResult]:
        match = re.search(r"\d{6}", code)
        if not match:
            return None

        stock_code = match.group(0)
        logger.info(f"========== 处理 A 股: {stock_code} ==========")

        try:
            time.sleep(2)  # 轻量限流

            df = self.fetch_and_save_stock_data(stock_code)
            if df is None:
                logger.error(f"[{stock_code}] 无行情数据，跳过")
                return None

            if dry_run:
                logger.info(f"[{stock_code}] dry-run 模式，跳过 AI 分析")
                return None

            stock_info = self.portfolio.get(
                stock_code, {"name": f"A股{stock_code}", "sector": DEFAULT_SECTOR}
            )
            stock_info.setdefault("code", stock_code)

            tech_data = self._calculate_technical_indicators(df)
            trend_context = self._get_trend_radar_context(stock_code)

            prompt = self.analyzer.generate_cio_prompt(stock_info, tech_data, trend_context)

            context = {
                "code": stock_code,
                "stock_name": stock_info.get("name", stock_code),
                "date": date.today().strftime("%Y-%m-%d"),
            }

            result = self.analyzer.analyze(context, custom_prompt=prompt)
            if result is None:
                logger.error(f"[{stock_code}] AI 返回为空，已丢弃")
                return None

            logger.info(f"[{stock_code}] AI 分析完成")
            return result

        except Exception as e:
            logger.exception(f"[{stock_code}] 处理异常: {e}")
            return None

    # ---------- 主执行 ----------

    def run(
        self,
        stock_codes: Optional[List[str]] = None,
        dry_run: bool = False,
        send_notification: bool = True,
    ) -> List[AnalysisResult]:

        if stock_codes is None:
            # 这里调用 keys() 现在是安全的了，因为我们已经在 load 时转成了 dict
            stock_codes = list(self.portfolio.keys()) if self.portfolio else self.config.stock_list

        results: List[AnalysisResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self.process_single_stock, code, dry_run): code
                for code in stock_codes
            }

            for future in as_completed(future_map):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    logger.exception(f"子线程异常: {e}")

        if results and send_notification and not dry_run:
            try:
                report = self.notifier.generate_dashboard_report(results)
                if self.notifier.is_available():
                    self.notifier.send_to_telegram(report)
            except Exception as e:
                logger.exception(f"发送通知失败，但不影响流程: {e}")

        logger.info(f"本次运行完成 | 成功分析 {len(results)} 只股票")
        return results

# ================= CLI =================

def parse_arguments():
    parser = argparse.ArgumentParser(description="AI-CIO")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="仅抓取数据，不进行 AI 分析")
    parser.add_argument("--stocks", type=str)
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--workers", type=int)
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = get_config()
    setup_logging(args.debug, config.log_dir)

    stock_list = [s.strip() for s in args.stocks.split(",")] if args.stocks else None

    pipeline = StockAnalysisPipeline(config, max_workers=args.workers)
    pipeline.run(
        stock_codes=stock_list,
        dry_run=args.dry_run,
        send_notification=not args.no_notify,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
