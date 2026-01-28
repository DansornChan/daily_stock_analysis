# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI-CIO 版
===================================
"""
import os
import argparse
import logging
import sys
import time
import json
import re
import pandas as pd  # <--- 关键修复：必须导入 pandas 以计算指标
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from config import get_config, Config
from storage import get_db, DatabaseManager
from data_provider import DataFetcherManager
from data_provider.akshare_fetcher import AkshareFetcher
from analyzer import GeminiAnalyzer, AnalysisResult
from notification import NotificationService, NotificationChannel
from feishu_doc import FeishuDocManager
from stock_analyzer import StockTrendAnalyzer

# 配置日志格式
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"stock_analysis_{today_str}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清理旧的 handler 防止重复打印
    root_logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)
    
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)
    
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
DEFAULT_SECTOR = "Macro"

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        
        # === [关键修复] 优先读取环境变量，确保 YAML 配置生效 ===
        env_workers = os.getenv("MAX_CONCURRENT")
        if env_workers:
            self.max_workers = int(env_workers)
        else:
            self.max_workers = max_workers or self.config.max_workers or 1
        
        # 加载投资组合
        self.portfolio = self._load_portfolio_config()
        
        # 初始化模块
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.akshare_fetcher = AkshareFetcher()
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()
        
        logger.info(f"AI-CIO 系统初始化完成，并发数: {self.max_workers}，监控标的: {len(self.portfolio)} 只")

    def _load_portfolio_config(self) -> dict:
        """从 portfolio.json 加载配置"""
        json_path = "portfolio.json"
        if not os.path.exists(json_path):
            logger.error(f"配置文件 {json_path} 不存在！请先创建。")
            return {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取配置文件失败: {e}")
            return {}

    def _get_trend_radar_context(self, code: str, json_path: str = 'news_summary.json') -> dict:
        """读取并结构化 TrendRadar 新闻"""
        context = {'macro': "", 'sector': "", 'target_sector': DEFAULT_SECTOR}
        stock_info = self.portfolio.get(code, {})
        target_sector = stock_info.get('sector', DEFAULT_SECTOR)
        context['target_sector'] = target_sector

        if not os.path.exists(json_path):
            return context

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                news_items = json.load(f)
            
            macro_news, sector_news = [], []
            for item in news_items:
                cat = item.get('category', 'Macro')
                line = f"- {item.get('title', '')}: {item.get('summary', '')}"
                if cat in ['Macro', 'Finance']: macro_news.append(line)
                if cat == target_sector: sector_news.append(line)

            context['macro'] = "\n".join(macro_news) if macro_news else "当前宏观面平静。"
            context['sector'] = "\n".join(sector_news) if sector_news else f"当前{target_sector}板块无重大消息。"
            return context
        except Exception as e:
            logger.warning(f"读取新闻失败: {e}")
            return context

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        """计算硬核技术指标 (MA, RSI, MACD)"""
        if df is None or df.empty: return {}
        try:
            df = df.sort_values('date')
            close = df['close']
            
            # 均线
            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # 趋势
            if ma5 > ma20 > ma60: trend = "多头排列 (强)"
            elif ma5 < ma20 < ma60: trend = "空头排列 (弱)"
            else: trend = "震荡整理"
            
            vol = df['volume']
            vol_ratio = vol.iloc[-1] / vol.rolling(5).mean().iloc[-1] if vol.rolling(5).mean().iloc[-1] > 0 else 0

            return {
                "price": close.iloc[-1],
                "change_pct": df['pct_change'].iloc[-1] if 'pct_change' in df else 0,
                "ma5": ma5, "ma20": ma20, "ma60": ma60,
                "rsi": rsi, "macd": macd.iloc[-1], "macd_signal": signal.iloc[-1],
                "support": df['low'].tail(20).min(), "resistance": df['high'].tail(20).max(),
                "trend": trend, "vol_ratio": vol_ratio
            }
        except Exception as e:
            logger.warning(f"指标计算失败: {e}")
            return {}

    def fetch_and_save_stock_data(self, code: str) -> bool:
        try:
            if self.db.has_today_data(code, date.today()): return True
            df, source = self.fetcher_manager.get_daily_data(code, days=100)
            if df is None or df.empty: return False
            self.db.save_daily_data(df, code, source)
            return True
        except Exception as e:
            logger.error(f"[{code}] 数据获取失败: {e}")
            return False

    def process_single_stock(self, code: str, skip_analysis: bool = False, single_notify: bool = False) -> Optional[AnalysisResult]:
        """处理单只 A 股全流程"""
        match = re.search(r'\d{6}', code)
        if not match: return None
        fetch_code = match.group(0)
        
        logger.info(f"========== 开始处理 A 股: {fetch_code} ==========")
        try:
            # 1. 抓取数据 (增加休眠防封)
            time.sleep(3)
            self.fetch_and_save_stock_data(fetch_code)
            
            if skip_analysis: return None

            # 2. 准备素材
            stock_info = self.portfolio.get(fetch_code, {"name": f"A股{fetch_code}", "sector": DEFAULT_SECTOR})
            df, _ = self.fetcher_manager.get_daily_data(fetch_code, days=100)
            tech_data = self._calculate_technical_indicators(df)
            trend_context = self._get_trend_radar_context(fetch_code)
            
            # 3. AI 分析
            prompt = self.analyzer.generate_cio_prompt(stock_info, tech_data, trend_context)
            base_context = {'code': fetch_code, 'stock_name': stock_info.get('name', fetch_code), 'date': date.today().strftime('%Y-%m-%d')}
            result = self.analyzer.analyze(base_context, custom_prompt=prompt)
            
            if result:
                logger.info(f"[{fetch_code}] 分析完成: {result.operation_advice}")
                if single_notify: self.notifier.send(self.notifier.generate_single_stock_report(result))
                
                # 限流：分析完强行等 10 秒
                logger.info("等待 API 配额重置 (10s)...")
                time.sleep(10)
            
            return result
        except Exception as e:
            logger.exception(f"[{fetch_code}] 处理异常: {e}")
            return None

    def run(self, stock_codes: Optional[List[str]] = None, dry_run: bool = False, send_notification: bool = True) -> List[AnalysisResult]:
        if stock_codes is None:
            stock_codes = list(self.portfolio.keys()) if self.portfolio else self.config.stock_list
                
        results = []
        # 使用单线程或指定线程运行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {executor.submit(self.process_single_stock, code, dry_run, False): code for code in stock_codes}
            for future in as_completed(future_to_code):
                res = future.result()
                if res: results.append(res)

        if results and send_notification and not dry_run:
            report = self.notifier.generate_dashboard_report(results)
            self.notifier.save_report_to_file(report)
            if self.notifier.is_available():
                self.notifier.send_to_telegram(report)
        return results

def main():
    args = parse_arguments()
    config = get_config()
    setup_logging(args.debug, config.log_dir)
    cmd_stocks = [c.strip() for c in args.stocks.split(',')] if args.stocks else None
    pipeline = StockAnalysisPipeline(config, max_workers=args.workers)
    pipeline.run(stock_codes=cmd_stocks, dry_run=args.dry_run, send_notification=not args.no_notify)
    return 0

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI-CIO Stock Analysis')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stocks', type=str)
    parser.add_argument('--no-notify', action='store_true')
    parser.add_argument('--workers', type=int)
    return parser.parse_args()

if __name__ == "__main__":
    sys.exit(main())
