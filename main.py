#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CIO 多资产智能分析系统
支持：
- 腾讯实时行情
- 历史K线
- Gemini AI 分析
- Telegram 推送

增强健壮性：
- 兼容未知命令行参数（如 --workers 1）
- 单个数据源/组件失败不影响整体运行
"""

import os
import sys
import argparse
import logging
import time
import json
import re
import pandas as pd

from datetime import datetime, date
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Optional

# ================= 编码修复 =================
try:
    import codecs
    if sys.stdout.encoding != "UTF-8" and hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    if sys.stderr.encoding != "UTF-8" and hasattr(sys.stderr, "buffer"):
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)
except:
    pass

os.environ["PYTHONIOENCODING"] = "utf-8"

# ================= 核心模块导入（带降级处理）=================

from config import get_config, Config
from storage import get_db

# ---------- 数据源管理器：使用已修正的 base.py，自动跳过不可用 fetcher ----------
try:
    from data_provider import DataFetcherManager
    _fetcher_manager = DataFetcherManager()
    logger = logging.getLogger(__name__)
    logger.info("✅ DataFetcherManager 初始化成功")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ DataFetcherManager 初始化失败: {e}，将使用降级数据源")
    # 降级方案：手动创建只包含最稳定数据源的管理器
    try:
        from data_provider.efinance_fetcher import EfinanceFetcher
        from data_provider.akshare_fetcher import AkshareFetcher
        from data_provider.base import DataFetcherManager
        _fetcher_manager = DataFetcherManager(fetchers=[
            EfinanceFetcher(),
            AkshareFetcher(),
        ])
        logger.info("✅ 降级数据源管理器初始化成功")
    except Exception as e2:
        logger.error(f"❌ 降级数据源也失败: {e2}，将无法获取历史数据")
        _fetcher_manager = None

# ---------- AI 分析器：尝试初始化，失败则标记不可用 ----------
try:
    from analyzer import GeminiAnalyzer, AnalysisResult
    _analyzer = GeminiAnalyzer()
    logger.info("✅ GeminiAnalyzer 初始化成功")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ GeminiAnalyzer 初始化失败: {e}，AI分析功能不可用")
    GeminiAnalyzer = None
    AnalysisResult = None
    _analyzer = None

# ---------- 通知服务：尝试初始化，失败则标记不可用 ----------
try:
    from notification import NotificationService
    _notifier = NotificationService()
    logger.info("✅ NotificationService 初始化成功")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ NotificationService 初始化失败: {e}，推送功能不可用")
    NotificationService = None
    _notifier = None

# ================= 日志配置 =================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_SECTOR = "Macro"

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    log_file = log_path / f"stock_analysis_{today_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)


# ================= Pipeline =================

class StockAnalysisPipeline:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

        self.portfolio = self._load_portfolio_config()
        self.db = get_db()
        self.fetcher_manager = _fetcher_manager
        self.analyzer = _analyzer
        self.notifier = _notifier

        if self.analyzer is None:
            logger.warning("⚠️ AI分析器未初始化，将跳过AI分析")
        if self.notifier is None:
            logger.warning("⚠️ 通知服务未初始化，将跳过推送")

        logger.info("✅ AI-CIO 初始化完成（降级模式可能已启用）")

    # ---------- 加载持仓 ----------

    def _load_portfolio_config(self) -> dict:
        env_stock_list = os.getenv("STOCK_LIST", "")

        if env_stock_list:
            stock_codes = [
                code.strip() for code in env_stock_list.split(",") if code.strip()
            ]
            return {
                str(code): {
                    "code": str(code),
                    "cost": 0,
                    "shares": 0,
                    "name": f"资产{code}",
                    "sector": DEFAULT_SECTOR,
                }
                for code in stock_codes
            }

        path = "portfolio.json"
        if not os.path.exists(path):
            logger.warning("未找到 portfolio.json，将使用空持仓")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            final_data = {}
            if isinstance(data, dict):
                for code, info in data.items():
                    info = info if isinstance(info, dict) else {}
                    info["code"] = str(code)
                    info.setdefault("cost", 0)
                    info.setdefault("shares", 0)
                    info.setdefault("name", f"资产{code}")
                    info.setdefault("sector", DEFAULT_SECTOR)
                    final_data[str(code)] = info

            return final_data

        except Exception as e:
            logger.error(f"加载 portfolio.json 失败: {e}")
            return {}

    # ---------- 技术指标 ----------

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        df = df.sort_values("date").copy()
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        amount = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)

        current_price = float(close.iloc[-1])
        support = float(low.tail(20).min())
        resistance = float(high.tail(20).max())

        support_distance_pct = None
        resistance_distance_pct = None
        if current_price > 0:
            support_distance_pct = (current_price - support) / current_price * 100
            resistance_distance_pct = (resistance - current_price) / current_price * 100

        # 简化版资金流向：上涨日成交额-下跌日成交额（近5日/20日）
        price_change = close.diff().fillna(0)
        signed_amount = amount.where(price_change >= 0, -amount)
        money_flow_5d = float(signed_amount.tail(5).sum())
        money_flow_20d = float(signed_amount.tail(20).sum())
        avg_amount_5d = float(amount.tail(5).mean()) if len(amount) else 0.0
        money_flow_strength = (money_flow_5d / avg_amount_5d) if avg_amount_5d else 0.0

        return {
            "price": current_price,
            "ma5": close.rolling(5).mean().iloc[-1],
            "ma20": close.rolling(20).mean().iloc[-1],
            "ma60": close.rolling(60).mean().iloc[-1],
            "rsi": None,
            "macd": None,
            "support": support,
            "support_distance_pct": support_distance_pct,
            "resistance": resistance,
            "resistance_distance_pct": resistance_distance_pct,
            "money_flow_5d": money_flow_5d,
            "money_flow_20d": money_flow_20d,
            "money_flow_strength": money_flow_strength,
            "volume_mean_5d": float(volume.tail(5).mean()) if len(volume) else 0.0,
        }


    def _build_trend_context(self, stock_info: dict, asset_code: str) -> dict:
        """构建新闻/政策面上下文（可选，失败不影响主流程）。"""
        target_sector = stock_info.get("sector", DEFAULT_SECTOR)
        context = {
            "macro": "暂无显著宏观政策新闻，建议结合盘面实时跟踪。",
            "sector": f"{target_sector}暂无显著政策催化，需观察后续公告与行业数据。",
            "target_sector": target_sector,
        }

        try:
            from search_service import get_search_service

            search_service = get_search_service()
            if not search_service or not search_service.is_available:
                return context

            response = search_service.search_stock_news(
                stock_code=asset_code,
                stock_name=stock_info.get("name", f"资产{asset_code}"),
                max_results=3,
            )
            if response.success and response.results:
                snippets = [f"{item.title}：{item.snippet}" for item in response.results[:3]]
                context["sector"] = "；".join(snippets)
                context["macro"] = "请重点提炼上述新闻中的政策导向、监管表态与行业支持/约束信号。"
        except Exception as e:
            logger.debug(f"新闻/政策上下文构建失败，使用默认上下文: {e}")

        return context

    # ---------- 单资产处理 ----------

    def process_single_stock(self, code: str) -> Optional[object]:
        """
        处理单个资产分析流程
        :param code: 股票代码（可带前缀，如 'sz000858'；也可纯数字）
        """
        raw_code = str(code).strip()
        match = re.search(r"\d{6}", raw_code)
        if not match:
            logger.warning(f"无效代码: {raw_code}")
            return None

        asset_code = match.group(0)
        logger.info(f"========== 处理资产: {asset_code} (原始: {raw_code}) ==========")

        # ---------- 1. 腾讯实时行情（独立导入，异常不影响主流程）----------
        realtime_data = {}
        try:
            # 延迟导入，避免初始化阶段强制依赖
            from data_provider.tencent_fetcher import TencentFetcher
            tencent = TencentFetcher()
            realtime_data = tencent.get_realtime_quote(raw_code) or {}
            if realtime_data.get("name"):
                logger.info(f"腾讯行情获取成功: {realtime_data['name']}")
        except ImportError:
            logger.debug("TencentFetcher 未安装或不存在，跳过腾讯实时行情")
        except Exception as e:
            logger.warning(f"腾讯行情获取失败: {e}")

        # ---------- 2. 历史K线数据 ----------
        df = None
        source = None
        if self.fetcher_manager is not None:
            try:
                df, source = self.fetcher_manager.get_daily_data(asset_code, days=120)
                logger.info(f"历史数据源: {source}")
            except Exception as e:
                logger.warning(f"历史数据获取失败: {e}")
        else:
            logger.warning("数据源管理器不可用，无法获取历史数据")

        # ---------- 3. 资产基础信息 ----------
        stock_info = self.portfolio.get(
            asset_code,
            {
                "name": f"资产{asset_code}",
                "sector": DEFAULT_SECTOR,
                "cost": 0,
                "shares": 0,
            }
        )
        stock_info["code"] = asset_code

        if realtime_data.get("name"):
            stock_info["name"] = realtime_data["name"]

        # ---------- 4. 技术指标 ----------
        if df is not None and not df.empty:
            tech_data = self._calculate_technical_indicators(df)
        else:
            logger.warning("无历史K线，使用最小技术结构")
            tech_data = {
                "price": realtime_data.get("price", 0),
                "ma5": None,
                "ma20": None,
                "ma60": None,
                "rsi": None,
                "macd": None,
                "support": None,
                "support_distance_pct": None,
                "resistance": None,
                "resistance_distance_pct": None,
                "money_flow_5d": None,
                "money_flow_20d": None,
                "money_flow_strength": None,
                "volume_mean_5d": None,
            }

        if realtime_data.get("price") is not None:
            tech_data["price"] = realtime_data["price"]

        # ---------- 5. AI 分析（如果分析器可用）----------
        if self.analyzer is None:
            logger.warning("AI分析器不可用，跳过分析")
            return None

        try:
            base_prompt = self.analyzer.generate_cio_prompt(
                stock_info,
                tech_data,
                self._build_trend_context(stock_info, asset_code),
            )

            context = {
                "code": asset_code,
                "stock_name": stock_info["name"],
                "date": date.today().strftime("%Y-%m-%d"),
            }

            result = self.analyzer.analyze(context, custom_prompt=base_prompt)
            return result
        except Exception as e:
            logger.error(f"AI分析失败: {e}", exc_info=True)
            return None

    # ---------- 主执行 ----------

    def run(self, stock_codes: Optional[List[str]] = None):
        if not stock_codes:
            stock_codes = list(self.portfolio.keys())

        results = []

        for code in stock_codes:
            res = self.process_single_stock(code)
            if res:
                results.append(res)
            time.sleep(5)

        if results and self.notifier is not None:
            try:
                report = self.notifier.generate_dashboard_report(results)
                if report and self.notifier.is_available():
                    self.notifier.send_to_telegram(report)
            except Exception as e:
                logger.error(f"推送报告失败: {e}")

        return results


# ================= CLI =================

def parse_args():
    """
    使用 parse_known_args 忽略未知参数（如 CI 环境传递的 --workers 1）
    """
    parser = argparse.ArgumentParser(description="AI-CIO 多资产分析系统")
    parser.add_argument("--stocks", type=str, help="资产代码，用逗号分隔")
    parser.add_argument("--debug", action="store_true")
    # 解析已知参数，忽略未知参数
    args, unknown = parser.parse_known_args()
    if unknown:
        logger = logging.getLogger(__name__)
        logger.warning(f"忽略未知命令行参数: {unknown}")
    return args


def main():
    args = parse_args()
    setup_logging(args.debug)

    logger.info("🚀 启动 AI-CIO 多资产分析系统")

    try:
        pipeline = StockAnalysisPipeline()

        stock_codes = []
        if args.stocks:
            stock_codes = [
                code.strip() for code in args.stocks.split(",") if code.strip()
            ]

        results = pipeline.run(stock_codes)
        logger.info(f"🎉 分析完成，共 {len(results)} 个结果")
        return 0

    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
