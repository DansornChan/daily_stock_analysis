# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 主调度程序
===================================

职责：
1. 协调各模块完成股票分析流程
2. 实现低并发的线程池调度
3. 全局异常处理，确保单股失败不影响整体
4. 提供命令行入口

使用方式：
    python main.py              # 正常运行
    python main.py --debug      # 调试模式
    python main.py --dry-run    # 仅获取数据不分析

交易理念（已融入分析）：
- 严进策略：不追高，乖离率 > 5% 不买入
- 趋势交易：只做 MA5>MA10>MA20 多头排列
- 效率优先：关注筹码集中度好的股票
- 买点偏好：缩量回踩 MA5/MA10 支撑
"""
import os

# 代理配置 - 仅在本地环境使用，GitHub Actions 不需要
if os.getenv("GITHUB_ACTIONS") != "true":
    pass

import argparse
import logging
import sys
import time
import json  # <--- 已添加
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
from analyzer import GeminiAnalyzer, AnalysisResult, STOCK_NAME_MAP
from notification import NotificationService, NotificationChannel, send_daily_report
from search_service import SearchService, SearchResponse
from stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult
from market_analyzer import MarketAnalyzer

# 配置日志格式
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    """配置日志系统"""
    level = logging.DEBUG if debug else logging.INFO
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"stock_analysis_{today_str}.log"
    debug_log_file = log_path / f"stock_analysis_debug_{today_str}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)
    debug_handler = RotatingFileHandler(debug_log_file, maxBytes=50 * 1024 * 1024, backupCount=3, encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(debug_handler)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.info(f"日志系统初始化完成，日志目录: {log_path.absolute()}")


logger = logging.getLogger(__name__)

# ==========================================
# 新增配置：股票行业映射表
# Key: 股票代码 (去除后缀), Value: 行业标签
# ==========================================
STOCK_SECTOR_MAP = {
    "603098": "Industrial", "NVDA": "Tech", "AAPL": "Tech",
    "TSLA": "Energy", "00700": "Tech", "600519": "Consumer",
    "BTC": "Crypto", "SPY": "Macro", "QQQ": "Macro", "300300": "Macro"
}
DEFAULT_SECTOR = "Macro"

class StockAnalysisPipeline:
    """股票分析主流程调度器"""
    
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        """初始化调度器"""
        self.config = config or get_config()
        self.max_workers = max_workers or self.config.max_workers
        
        # 初始化各模块 (这些必须在 __init__ 内部完成)
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.akshare_fetcher = AkshareFetcher()
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()
        
        # 初始化搜索服务
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=self.config.tavily_api_keys,
            serpapi_keys=self.config.serpapi_keys,
        )
        
        logger.info(f"调度器初始化完成，最大并发数: {self.max_workers}")
        if self.search_service.is_available:
            logger.info("搜索服务已启用 (Tavily/SerpAPI)")
        else:
            logger.warning("搜索服务未启用（未配置 API Key）")

    # === 新增方法：读取并筛选 TrendRadar 新闻 ===
    def _get_trend_radar_context(self, code: str, json_path: str = 'news_summary.json') -> str:
        """读取上游 Action 生成的新闻文件，并根据行业进行筛选"""
        if not os.path.exists(json_path):
            return ""

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                news_items = json.load(f)
            
            clean_code = code.split('.')[0] 
            target_sector = STOCK_SECTOR_MAP.get(clean_code, DEFAULT_SECTOR)
            
            filtered_news = []
            for item in news_items:
                category = item.get('category', 'Macro') 
                if category == 'Macro' or category == target_sector:
                    title = item.get('title', '无标题')
                    summary = item.get('summary', '')
                    filtered_news.append(f"- 【{category}】{title}: {summary}")

            if not filtered_news:
                return ""
            
            return "【来自 TrendRadar 的行业与宏观简报】\n" + "\n".join(filtered_news) + "\n"
            
        except Exception as e:
            logger.warning(f"[{code}] 读取 TrendRadar 新闻失败: {e}")
            return ""
    # ==========================================
    
    def fetch_and_save_stock_data(self, code: str, force_refresh: bool = False) -> Tuple[bool, Optional[str]]:
        """获取并保存单只股票数据"""
        try:
            today = date.today()
            if not force_refresh and self.db.has_today_data(code, today):
                logger.info(f"[{code}] 今日数据已存在，跳过获取")
                return True, None
            
            logger.info(f"[{code}] 开始从数据源获取数据...")
            df, source_name = self.fetcher_manager.get_daily_data(code, days=30)
            
            if df is None or df.empty:
                return False, "获取数据为空"
            
            saved_count = self.db.save_daily_data(df, code, source_name)
            logger.info(f"[{code}] 数据保存成功（来源: {source_name}，新增 {saved_count} 条）")
            return True, None
        except Exception as e:
            error_msg = f"获取/保存数据失败: {str(e)}"
            logger.error(f"[{code}] {error_msg}")
            return False, error_msg
    
    def analyze_stock(self, code: str) -> Optional[AnalysisResult]:
        """分析单只股票"""
        try:
            stock_name = STOCK_NAME_MAP.get(code, '')
            
            # Step 1: 获取实时行情
            realtime_quote: Optional[RealtimeQuote] = None
            try:
                realtime_quote = self.akshare_fetcher.get_realtime_quote(code)
                if realtime_quote:
                    if realtime_quote.name: stock_name = realtime_quote.name
                    logger.info(f"[{code}] {stock_name} 实时行情: 价格={realtime_quote.price}")
            except Exception as e:
                logger.warning(f"[{code}] 获取实时行情失败: {e}")
            
            if not stock_name: stock_name = f'股票{code}'
            
            # Step 2: 获取筹码分布
            chip_data: Optional[ChipDistribution] = None
            try:
                chip_data = self.akshare_fetcher.get_chip_distribution(code)
                if chip_data: logger.info(f"[{code}] 筹码分布: 获利={chip_data.profit_ratio:.1%}")
            except Exception as e:
                logger.warning(f"[{code}] 获取筹码分布失败: {e}")
            
            # Step 3: 趋势分析
            trend_result: Optional[TrendAnalysisResult] = None
            try:
                context = self.db.get_analysis_context(code)
                if context and 'raw_data' in context:
                    import pandas as pd
                    raw_data = context['raw_data']
                    if isinstance(raw_data, list) and len(raw_data) > 0:
                        df = pd.DataFrame(raw_data)
                        trend_result = self.trend_analyzer.analyze(df, code)
                        logger.info(f"[{code}] 趋势分析: {trend_result.trend_status.value}")
            except Exception as e:
                logger.warning(f"[{code}] 趋势分析失败: {e}")
            
            # Step 4: 多维度情报搜索
            news_context = None
            if self.search_service.is_available:
                logger.info(f"[{code}] 开始多维度情报搜索...")
                intel_results = self.search_service.search_comprehensive_intel(
                    stock_code=code, stock_name=stock_name, max_searches=3
                )
                if intel_results:
                    news_context = self.search_service.format_intel_report(intel_results, stock_name)
                    logger.info(f"[{code}] 情报搜索完成")
            else:
                logger.info(f"[{code}] 搜索服务不可用，跳过")

            # === 【插入点】注入 TrendRadar 新闻上下文 ===
            trend_news = self._get_trend_radar_context(code)
            if trend_news:
                logger.info(f"[{code}] 已注入 TrendRadar 行业舆情")
                if news_context is None:
                    news_context = ""
                news_context = trend_news + "\n" + news_context
            # ========================================
            
            # Step 5: 获取分析上下文
            context = self.db.get_analysis_context(code)
            if context is None:
                logger.warning(f"[{code}] 无法获取分析上下文，跳过分析")
                return None
            
            # Step 6: 增强上下文数据
            enhanced_context = self._enhance_context(
                context, realtime_quote, chip_data, trend_result, stock_name
            )
            
            # Step 7: 调用 AI 分析
            result = self.analyzer.analyze(enhanced_context, news_context=news_context)
            return result
            
        except Exception as e:
            logger.error(f"[{code}] 分析失败: {e}")
            logger.exception(f"[{code}] 详细错误信息:")
            return None
    
    def _enhance_context(self, context, realtime_quote, chip_data, trend_result, stock_name=""):
        enhanced = context.copy()
        if stock_name: enhanced['stock_name'] = stock_name
        elif realtime_quote and realtime_quote.name: enhanced['stock_name'] = realtime_quote.name
        
        if realtime_quote:
            enhanced['realtime'] = {
                'name': realtime_quote.name,
                'price': realtime_quote.price,
                'volume_ratio': realtime_quote.volume_ratio,
                'volume_ratio_desc': self._describe_volume_ratio(realtime_quote.volume_ratio),
                'turnover_rate': realtime_quote.turnover_rate,
                'pe_ratio': realtime_quote.pe_ratio,
                'pb_ratio': realtime_quote.pb_ratio,
                'total_mv': realtime_quote.total_mv,
                'circ_mv': realtime_quote.circ_mv,
                'change_60d': realtime_quote.change_60d,
            }
        
        if chip_data:
            current_price = realtime_quote.price if realtime_quote else 0
            enhanced['chip'] = {
                'profit_ratio': chip_data.profit_ratio,
                'avg_cost': chip_data.avg_cost,
                'concentration_90': chip_data.concentration_90,
                'concentration_70': chip_data.concentration_70,
                'chip_status': chip_data.get_chip_status(current_price),
            }
        
        if trend_result:
            enhanced['trend_analysis'] = {
                'trend_status': trend_result.trend_status.value,
                'ma_alignment': trend_result.ma_alignment,
                'trend_strength': trend_result.trend_strength,
                'bias_ma5': trend_result.bias_ma5,
                'bias_ma10': trend_result.bias_ma10,
                'volume_status': trend_result.volume_status.value,
                'volume_trend': trend_result.volume_trend,
                'buy_signal': trend_result.buy_signal.value,
                'signal_score': trend_result.signal_score,
                'signal_reasons': trend_result.signal_reasons,
                'risk_factors': trend_result.risk_factors,
            }
        return enhanced
    
    def _describe_volume_ratio(self, volume_ratio: float) -> str:
        if volume_ratio < 0.5: return "极度萎缩"
        elif volume_ratio < 0.8: return "明显萎缩"
        elif volume_ratio < 1.2: return "正常"
        elif volume_ratio < 2.0: return "温和放量"
        elif volume_ratio < 3.0: return "明显放量"
        else: return "巨量"
    
    def process_single_stock(self, code: str, skip_analysis: bool = False, single_stock_notify: bool = False) -> Optional[AnalysisResult]:
        logger.info(f"========== 开始处理 {code} ==========")
        try:
            success, error = self.fetch_and_save_stock_data(code)
            if not success: logger.warning(f"[{code}] 数据获取失败: {error}")
            
            if skip_analysis:
                logger.info(f"[{code}] 跳过 AI 分析（dry-run 模式）")
                return None
            
            result = self.analyze_stock(code)
            if result:
                logger.info(f"[{code}] 分析完成: {result.operation_advice}, 评分 {result.sentiment_score}")
                if single_stock_notify and self.notifier.is_available():
                    try:
                        single_report = self.notifier.generate_single_stock_report(result)
                        if self.notifier.send(single_report): logger.info(f"[{code}] 单股推送成功")
                        else: logger.warning(f"[{code}] 单股推送失败")
                    except Exception as e:
                        logger.error(f"[{code}] 单股推送异常: {e}")
            return result
        except Exception as e:
            logger.exception(f"[{code}] 处理过程发生未知异常: {e}")
            return None
    
    def run(self, stock_codes: Optional[List[str]] = None, dry_run: bool = False, send_notification: bool = True) -> List[AnalysisResult]:
        start_time = time.time()
        if stock_codes is None:
            self.config.refresh_stock_list()
            stock_codes = self.config.stock_list
        
        if not stock_codes:
            logger.error("未配置自选股列表")
            return []
        
        logger.info(f"===== 开始分析 {len(stock_codes)} 只股票 =====")
        logger.info(f"股票列表: {', '.join(stock_codes)}")
        
        single_stock_notify = getattr(self.config, 'single_stock_notify', False)
        results: List[AnalysisResult] = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {
                executor.submit(self.process_single_stock, code, dry_run, single_stock_notify and send_notification): code
                for code in stock_codes
            }
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    result = future.result()
                    if result: results.append(result)
                except Exception as e:
                    logger.error(f"[{code}] 任务执行失败: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"===== 分析完成 =====")
        logger.info(f"成功: {len(results)}, 耗时: {elapsed_time:.2f} 秒")
        
        if results and send_notification and not dry_run:
            if single_stock_notify:
                logger.info("单股推送模式：仅保存报告到本地")
                self._send_notifications(results, skip_push=True)
            else:
                self._send_notifications(results)
        return results
    
    def _send_notifications(self, results: List[AnalysisResult], skip_push: bool = False) -> None:
        try:
            logger.info("生成决策仪表盘日报...")
            report = self.notifier.generate_dashboard_report(results)
            filepath = self.notifier.save_report_to_file(report)
            logger.info(f"决策仪表盘日报已保存: {filepath}")
            
            if skip_push: return
            
            if self.notifier.is_available():
                channels = self.notifier.get_available_channels()
                success = False
                if NotificationChannel.WECHAT in channels:
                    content = self.notifier.generate_wechat_dashboard(results)
                    success = self.notifier.send_to_wechat(content) or success
                
                for channel in channels:
                    if channel == NotificationChannel.WECHAT: continue
                    if channel == NotificationChannel.FEISHU: success = self.notifier.send_to_feishu(report) or success
                    elif channel == NotificationChannel.TELEGRAM: success = self.notifier.send_to_telegram(report) or success
                    elif channel == NotificationChannel.EMAIL: success = self.notifier.send_to_email(report) or success
                    elif channel == NotificationChannel.CUSTOM: success = self.notifier.send_to_custom(report) or success
                
                if success: logger.info("推送成功")
                else: logger.warning("推送失败")
            else:
                logger.info("通知渠道未配置")
        except Exception as e:
            logger.error(f"发送通知失败: {e}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='A股自选股智能分析系统')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--dry-run', action='store_true', help='仅获取数据')
    parser.add_argument('--stocks', type=str, help='指定股票代码')
    parser.add_argument('--no-notify', action='store_true', help='不发送推送')
    parser.add_argument('--single-notify', action='store_true', help='启用单股推送')
    parser.add_argument('--workers', type=int, default=None, help='并发线程数')
    parser.add_argument('--schedule', action='store_true', help='启用定时任务')
    parser.add_argument('--market-review', action='store_true', help='仅运行大盘复盘')
    parser.add_argument('--no-market-review', action='store_true', help='跳过大盘复盘')
    parser.add_argument('--webui', action='store_true', help='启动WebUI')
    parser.add_argument('--webui-only', action='store_true', help='仅启动WebUI服务')
    return parser.parse_args()


def run_market_review(notifier: NotificationService, analyzer=None, search_service=None) -> Optional[str]:
    logger.info("开始执行大盘复盘分析...")
    try:
        market_analyzer = MarketAnalyzer(search_service=search_service, analyzer=analyzer)
        review_report = market_analyzer.run_daily_review()
        if review_report:
            date_str = datetime.now().strftime('%Y%m%d')
            filepath = notifier.save_report_to_file(f"# 🎯 大盘复盘\n\n{review_report}", f"market_review_{date_str}.md")
            logger.info(f"大盘复盘报告已保存: {filepath}")
            if notifier.is_available():
                notifier.send(f"🎯 大盘复盘\n\n{review_report}")
            return review_report
    except Exception as e:
        logger.error(f"大盘复盘分析失败: {e}")
    return None


def run_full_analysis(config: Config, args: argparse.Namespace, stock_codes: Optional[List[str]] = None):
    try:
        if getattr(args, 'single_notify', False): config.single_stock_notify = True
        pipeline = StockAnalysisPipeline(config=config, max_workers=args.workers)
        
        results = pipeline.run(stock_codes=stock_codes, dry_run=args.dry_run, send_notification=not args.no_notify)
        
        market_report = ""
        if config.market_review_enabled and not args.no_market_review:
            review_result = run_market_review(pipeline.notifier, pipeline.analyzer, pipeline.search_service)
            if review_result: market_report = review_result
        
        try:
            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and (results or market_report):
                logger.info("正在创建飞书云文档...")
                tz_cn = timezone(timedelta(hours=8))
                now = datetime.now(tz_cn)
                doc_title = f"{now.strftime('%Y-%m-%d %H:%M')} 大盘复盘"
                full_content = ""
                if market_report: full_content += f"# 📈 大盘复盘\n\n{market_report}\n\n---\n\n"
                if results:
                    dashboard_content = pipeline.notifier.generate_dashboard_report(results)
                    full_content += f"# 🚀 个股决策仪表盘\n\n{dashboard_content}"
                
                doc_url = feishu_doc.create_daily_doc(doc_title, full_content)
                if doc_url:
                    logger.info(f"飞书云文档创建成功: {doc_url}")
                    pipeline.notifier.send(f"[{now.strftime('%Y-%m-%d %H:%M')}] 复盘文档创建成功: {doc_url}")
        except Exception as e:
            logger.error(f"飞书文档生成失败: {e}")
            
    except Exception as e:
        logger.exception(f"分析流程执行失败: {e}")


def main() -> int:
    args = parse_arguments()
    config = get_config()
    setup_logging(debug=args.debug, log_dir=config.log_dir)
    
    logger.info("=" * 60)
    logger.info("A股自选股智能分析系统 启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    config.validate()
    
    stock_codes = None
    if args.stocks:
        stock_codes = [code.strip() for code in args.stocks.split(',') if code.strip()]
    
    start_webui = (args.webui or args.webui_only or config.webui_enabled) and os.getenv("GITHUB_ACTIONS") != "true"
    if start_webui:
        try:
            from webui import run_server_in_thread
            run_server_in_thread(host=config.webui_host, port=config.webui_port)
        except Exception as e:
            logger.error(f"启动 WebUI 失败: {e}")
    
    if args.webui_only:
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            return 0

    try:
        if args.market_review:
            notifier = NotificationService()
            search_service = None
            analyzer = None
            if config.bocha_api_keys or config.tavily_api_keys:
                search_service = SearchService(bocha_keys=config.bocha_api_keys, tavily_keys=config.tavily_api_keys)
            if config.gemini_api_key:
                analyzer = GeminiAnalyzer(api_key=config.gemini_api_key)
            run_market_review(notifier, analyzer, search_service)
            return 0
        
        if args.schedule or config.schedule_enabled:
            from scheduler import run_with_schedule
            run_with_schedule(lambda: run_full_analysis(config, args, stock_codes), schedule_time=config.schedule_time, run_immediately=True)
            return 0
        
        run_full_analysis(config, args, stock_codes)
        
        if start_webui:
            try:
                while True: time.sleep(1)
            except KeyboardInterrupt: pass
        
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
