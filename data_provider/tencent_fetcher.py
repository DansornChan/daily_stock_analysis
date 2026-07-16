# -*- coding: utf-8 -*-
"""
===================================
TencentFetcher - 腾讯财经数据源 (Priority 5)
===================================

数据来源：腾讯财经（通过 tencent-stock-api 库）
仓库：https://github.com/ArSrNa/tencent-stock-api

说明：
- 项目内部股票代码通常是 6 位纯数字（如 600519 / 000001）
- 腾讯 API 需要前缀格式：sh600519 / sz000001 / hk00700 / usAAPL
- 本文件负责把项目代码自动转换为腾讯 API 需要的格式
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from tencent.stock import get_kline, get_quote

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS

logger = logging.getLogger(__name__)


class TencentFetcher(BaseFetcher):
    """腾讯财经数据源实现。"""

    name = "TencentFetcher"
    priority = 5

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从腾讯财经获取 K 线原始数据。"""
        code = self._convert_stock_code(stock_code)

        try:
            logger.info(
                "[TencentFetcher] 调用 get_kline(%s, period=day, start=%s, end=%s)",
                code,
                start_date,
                end_date,
            )
            # tencent-stock-api 文档中日线周期为 day
            df = get_kline(code, period="day", start_date=start_date, end_date=end_date)
        except TypeError:
            # 兼容部分版本参数名为 start/end
            df = get_kline(code, period="day", start=start_date, end=end_date)
        except Exception as e:
            raise DataFetchError(f"腾讯财经获取 K 线失败 {stock_code}: {e}") from e

        if df is None or df.empty:
            raise DataFetchError(f"腾讯财经未返回数据: {code}")

        df = df.copy()
        df.columns = [str(col).strip().lower() for col in df.columns]

        # 字段兼容：不同版本字段名可能不同
        if "amount" not in df.columns:
            if "turnover" in df.columns:
                df["amount"] = df["turnover"]
            elif "volume" in df.columns and "close" in df.columns:
                df["amount"] = pd.to_numeric(df["volume"], errors="coerce") * pd.to_numeric(
                    df["close"], errors="coerce"
                )

        if "pct_chg" not in df.columns:
            if "change_percent" in df.columns:
                df["pct_chg"] = df["change_percent"]
            elif "change_pct" in df.columns:
                df["pct_chg"] = df["change_pct"]

        if "date" not in df.columns:
            for candidate in ("time", "trade_date", "datetime"):
                if candidate in df.columns:
                    df["date"] = df[candidate]
                    break

        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化腾讯 K 线字段到项目统一格式。"""
        normalized_df = df.copy()
        normalized_df["code"] = stock_code

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in normalized_df.columns]
        return normalized_df[existing_cols]

    def get_realtime_quote(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取实时行情（扩展能力，不参与统一策略接口）。"""
        code = self._convert_stock_code(stock_code)
        try:
            raw = get_quote(code)
        except Exception as e:
            logger.error("[TencentFetcher] 获取实时行情失败 %s: %s", stock_code, e)
            return None

        return {
            "code": raw.get("code", code),
            "name": raw.get("name", ""),
            "price": raw.get("price", 0.0),
            "change": raw.get("change", 0.0),
            "change_percent": raw.get("change_percent", 0.0),
            "volume": raw.get("volume", 0),
            "turnover": raw.get("turnover", 0.0),
            "high": raw.get("high", 0.0),
            "low": raw.get("low", 0.0),
            "open": raw.get("open", 0.0),
            "pre_close": raw.get("pre_close", 0.0),
            "timestamp": raw.get("time", ""),
        }

    @staticmethod
    def _convert_stock_code(stock_code: str) -> str:
        """将项目内代码转换成腾讯 API 支持格式。"""
        code = str(stock_code).strip()
        lower_code = code.lower()

        # 已经是腾讯格式
        if lower_code.startswith(("sh", "sz", "hk", "us")):
            market = lower_code[:2]
            body = code[2:]
            if market == "hk":
                return f"hk{body.zfill(5)}"
            if market in ("sh", "sz"):
                return f"{market}{body.zfill(6)}"
            return f"us{body.upper()}"

        # 兼容 baostock/yfinance 常见格式
        if "." in lower_code:
            if lower_code.startswith("sh."):
                return f"sh{code.split('.')[-1].zfill(6)}"
            if lower_code.startswith("sz."):
                return f"sz{code.split('.')[-1].zfill(6)}"
            if lower_code.endswith(".ss"):
                return f"sh{code.split('.')[0].zfill(6)}"
            if lower_code.endswith(".sz"):
                return f"sz{code.split('.')[0].zfill(6)}"

        if code.isdigit():
            # A 股 6 位
            if len(code) == 6:
                if code.startswith(("6", "9", "5")):
                    return f"sh{code}"
                return f"sz{code}"
            # 港股 5 位
            if len(code) == 5:
                return f"hk{code}"

        # 美股字母代码
        if code.isalpha():
            return f"us{code.upper()}"

        raise DataFetchError(f"不支持的股票代码格式: {stock_code}")
