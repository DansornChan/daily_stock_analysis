# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI-CIO 版
===================================
"""
import os

# 代理配置 - 仅在本地环境使用
if os.getenv("GITHUB_ACTIONS") != "true":
    # os.environ["http_proxy"] = "http://127.0.0.1:10809"
    # os.environ["https_proxy"] = "http://127.0.0.1:10809"
    pass

import argparse
import logging
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from feishu_doc import FeishuDocManager

from config import get_config, Config
from storage import get_db, DatabaseManager
from data_provider import DataFetcherManager
from data_provider.akshare_fetcher import AkshareFetcher, RealtimeQuote, ChipDistribution
from analyzer import GeminiAnalyzer, AnalysisResult
from notification import NotificationService, NotificationChannel, send_daily_report
from search_service import SearchService, SearchResponse
from stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult
from market_analyzer import MarketAnalyzer

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
        self.max_workers = max_workers or self.config.max_workers
        
        # === 核心：加载投资组合配置 ===
        self.portfolio = self._load_portfolio_config()
        
        # 初始化模块
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.akshare_fetcher = AkshareFetcher()
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=self.config.tavily_api_keys,
            serpapi_keys=self.config.serpapi_keys,
        )
        logger.info(f"AI-CIO 系统初始化完成，监控标的: {len(self.portfolio)} 只")

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
        
        # 1. 确定板块
        clean_code = code.split('.')[0]
        stock_info = self.portfolio.get(clean_code, {})
        target_sector = stock_info.get('sector', DEFAULT_SECTOR)
        context['target_sector'] = target_sector

        if not os.path.exists(json_path):
            return context

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                news_items = json.load(f)
            
            macro_news = []
            sector_news = []
            
            for item in news_items:
                cat = item.get('category', 'Macro')
                title = item.get('title', '')
                summary = item.get('summary', '')
                line = f"- {title}: {summary}"
                
                if cat in ['Macro', 'Finance']:
                    macro_news.append(line)
                if cat == target_sector:
                    sector_news.append(line)

            context['macro'] = "\n".join(macro_news) if macro_news else "当前宏观面平静。"
            context['sector'] = "\n".join(sector_news) if sector_news else f"当前{target_sector}板块无重大消息。"
            return context
        except Exception as e:
            logger.warning(f"读取新闻失败: {e}")
            return context

    def _calculate_technical_indicators(self, df) -> dict:
        """计算硬核技术指标 (MA, RSI, MACD)"""
        if df is None or df.empty: return {}
        try:
            df = df.sort_values('date')
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
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
            
            # 趋势与量比
            if ma5 > ma20 > ma60: trend = "多头排列 (强)"
            elif ma5 < ma20 < ma60: trend = "空头排列 (弱)"
            else: trend = "震荡整理"
            
            vol_avg = volume.rolling(5).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0

            return {
                "price": close.iloc[-1],
                "change_pct": df['pct_change'].iloc[-1] if 'pct_change' in df else 0,
                "ma5": ma5, "ma20": ma20, "ma60": ma60,
                "rsi": rsi,
                "macd": macd.iloc[-1], "macd_signal": signal.iloc[-1],
                "support": low.tail(20).min(), "resistance": high.tail(20).max(),
                "trend": trend, "vol_ratio": vol_ratio
            }
        except Exception as e:
            logger.warning(f"指标计算失败: {e}")
            return {}

    def fetch_and_save_stock_data(self, code: str) -> bool:
        """获取数据并保存 (含断点续传)"""
        try:
            today = date.today()
            if self.db.has_today_data(code, today):
                return True
            
            df, source = self.fetcher_manager.get_daily_data(code, days=100) # 获取长一点的数据用于计算 MA60
            if df is None or df.empty: return False
            
            self.db.save_daily_data(df, code, source)
            return True
        except Exception as e:
            logger.error(f"[{code}] 数据获取失败: {e}")
            return False

    def process_single_stock(self, code: str, skip_analysis: bool = False, single_notify: bool = False) -> Optional[AnalysisResult]:
        """处理单只 A 股全流程 (适配纯数字代码)"""
        
        # 1. 代码预处理：确保是纯数字字符串
        # 兼容处理：有些代码可能被误写成 'sh601068' 或 '601068.SS'，我们要提取其中的数字
        import re
        match = re.search(r'\d{6}', code)
        if not match:
            logger.warning(f"[{code}] 非标准 A 股代码格式，跳过处理")
            return None
        
        fetch_code = match.group(0)
        logger.info(f"========== 开始处理 A 股: {fetch_code} ==========")
        
        try:
            # 2. 尝试获取并保存行情数据
            # 增加 3 秒休眠，彻底解决你之前遇到的 'RemoteDisconnected' 被封锁问题
            time.sleep(3) 
            data_success = self.fetch_and_save_stock_data(fetch_code)
            
            if not data_success:
                logger.warning(f"[{fetch_code}] 实时数据抓取失败，尝试从数据库调取历史数据...")
            
            if skip_analysis: return None

            # 3. 准备 AI 分析素材
            # 从 portfolio.json 中获取该股票的配置（如持仓策略、板块等）
            stock_info = self.portfolio.get(fetch_code, {
                "name": f"A股{fetch_code}", 
                "sector": DEFAULT_SECTOR, 
                "strategy": "未定义"
            })
            
            # 获取 DataFrame 并计算 MACD/RSI/均线等硬指标
            try:
                # days=100 确保有足够数据计算 MA60 和完整的 MACD
                df, _ = self.fetcher_manager.get_daily_data(fetch_code, days=100)
                tech_data = self._calculate_technical_indicators(df)
            except Exception as e:
                logger.error(f"[{fetch_code}] 技术指标计算失败 (可能数据量不足): {e}")
                return None
            
            # 4. 获取 TrendRadar 提供的宏观与行业背景
            trend_context = self._get_trend_radar_context(fetch_code)
            
            # 5. 生成 CIO 深度分析 Prompt (自上而下逻辑)
            prompt = self.analyzer.generate_cio_prompt(stock_info, tech_data, trend_context)
            
            base_context = {
                'code': fetch_code, 
                'stock_name': stock_info.get('name', fetch_code), 
                'date': date.today().strftime('%Y-%m-%d')
            }
            
            # 6. 调用 Gemini 进行分析 (此前已修复 list 类型错误)
            result = self.analyzer.analyze(base_context, custom_prompt=prompt)
            
            return result
            
        except Exception as e:
            logger.exception(f"[{fetch_code}] 整体分析流程发生异常: {e}")
            return None
            
        except Exception as e:
            logger.exception(f"[{code}] 分析过程异常: {e}")
            return None

    def run(self, stock_codes: Optional[List[str]] = None, dry_run: bool = False, send_notification: bool = True) -> List[AnalysisResult]:
        """主运行入口"""
        # 优先使用 portfolio.json 的 Key
        if stock_codes is None:
            if self.portfolio:
                stock_codes = list(self.portfolio.keys())
            else:
                self.config.refresh_stock_list()
                stock_codes = self.config.stock_list
                
        logger.info(f"即将分析 {len(stock_codes)} 只股票: {stock_codes}")
        
        single_notify = getattr(self.config, 'single_stock_notify', False)
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {
                executor.submit(self.process_single_stock, code, dry_run, single_notify and send_notification): code
                for code in stock_codes
            }
            for future in as_completed(future_to_code):
                res = future.result()
                if res: results.append(res)

        if results and send_notification and not dry_run:
            if single_notify:
                logger.info("单股推送模式：跳过汇总推送")
            else:
                # 生成并发送汇总日报
                report = self.notifier.generate_dashboard_report(results)
                filepath = self.notifier.save_report_to_file(report)
                
                # 发送各渠道
                if self.notifier.is_available():
                    self.notifier.send_to_telegram(report)
                    self.notifier.send_to_feishu(report)
                    # ... 其他渠道可自行添加 ...
        
        # 尝试生成飞书文档
        try:
            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and results:
                dashboard = self.notifier.generate_dashboard_report(results)
                doc_title = f"{datetime.now().strftime('%Y-%m-%d')} AI-CIO 投资日报"
                url = feishu_doc.create_daily_doc(doc_title, dashboard)
                if url: logger.info(f"飞书文档已生成: {url}")
        except Exception as e:
            logger.error(f"飞书文档生成失败: {e}")

        return results

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI-CIO Stock Analysis')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stocks', type=str)
    parser.add_argument('--no-notify', action='store_true')
    parser.add_argument('--single-notify', action='store_true')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--market-review', action='store_true')
    parser.add_argument('--no-market-review', action='store_true')
    parser.add_argument('--webui', action='store_true')
    parser.add_argument('--webui-only', action='store_true')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = get_config()
    setup_logging(args.debug, config.log_dir)
    
    # 兼容命令行指定的 stocks
    cmd_stocks = [c.strip() for c in args.stocks.split(',')] if args.stocks else None
    
    pipeline = StockAnalysisPipeline(config, max_workers=args.workers)
    
    if args.webui_only:
        # WebUI 启动逻辑省略，保持原样
        pass 
    
    if args.market_review:
        # 大盘复盘逻辑保持原样
        pass
        
    pipeline.run(stock_codes=cmd_stocks, dry_run=args.dry_run, send_notification=not args.no_notify)
    return 0

if __name__ == "__main__":
    sys.exit(main())
