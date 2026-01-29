# -*- coding: utf-8 -*-
import logging
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from config import get_config

logger = logging.getLogger(__name__)

# ================= 数据结构 =================

@dataclass
class AnalysisResult:
    code: str
    name: str
    date: str
    sentiment_score: int
    operation_advice: str
    risk_alert: str
    trend_prediction: str
    analysis_summary: str
    buy_reason: str = ""
    sell_reason: str = ""

    def get_emoji(self):
        if self.sentiment_score >= 80:
            return "🔴"
        if self.sentiment_score <= 40:
            return "🟢"
        return "🟡"

# ================= Analyzer =================

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.config = get_config()
        self.api_key = api_key or self.config.gemini_api_key

        if not self.api_key:
            logger.warning("Gemini API Key 未配置，AI 分析将被跳过")
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                google_api_key=self.api_key,
                temperature=0.1,
                timeout=120
            )

    # ---------- Prompt ----------

    def generate_cio_prompt(
        self,
        stock_info: Dict[str, Any],
        tech_data: Dict[str, Any],
        trend_context: Dict[str, Any]
    ) -> str:
        stock_name = stock_info.get("name", "未知股票")
        stock_code = stock_info.get("code", "Unknown")
        
        # === 获取持仓信息 ===
        cost = float(stock_info.get("cost", 0))
        shares = int(stock_info.get("shares", 0))
        current_price = float(tech_data.get("price", 0))
        
        # 计算持仓状态字符串
        position_context = ""
        if shares > 0 and cost > 0 and current_price > 0:
            profit_pct = (current_price - cost) / cost * 100
            status_str = "盈利" if profit_pct > 0 else "亏损"
            position_context = (
                f"【用户持仓状态】\n"
                f"用户当前持有该股 {shares} 股，持仓成本 {cost} 元。\n"
                f"当前浮动{status_str}比例：{profit_pct:.2f}%。\n"
                f"请基于此持仓状态给出具体操作建议（是应该止盈、止损、补仓做T还是继续持有？）。"
            )
        else:
            position_context = (
                "【用户持仓状态】\n"
                "用户当前为空仓（无持仓）。\n"
                "请基于空仓视角给出建议（当前是否为合适的建仓时机？如果是，建议什么价格区间介入？）。"
            )
        # ==================

        return f"""
你是一位经验丰富、风控优先的基金经理（CIO）。

请基于以下信息进行自上而下分析，并【严格只返回 JSON，不要包含任何解释性文字】。

【宏观背景】
{trend_context.get("macro", "无")}

【行业背景（{trend_context.get("target_sector", "未知")}）】
{trend_context.get("sector", "无")}

【个股技术面】
股票：{stock_name}（{stock_code}）
现价：{tech_data.get("price", "N/A")}
MA5 / MA20 / MA60：{tech_data.get("ma5", 0):.2f} / {tech_data.get("ma20", 0):.2f} / {tech_data.get("ma60", 0):.2f}
RSI：{tech_data.get("rsi", 0):.2f}
MACD：{tech_data.get("macd", 0):.2f}

{position_context}

【返回格式要求（必须严格遵守）】
{{
  "stock_name": "股票真实中文简称",
  "sentiment_score": 0-100 的整数,
  "operation_advice": "结合持仓状态的操作建议（如：亏损5%请耐心持有/盈利10%建议分批止盈/空仓建议观望）",
  "core_view": "一句话核心判断",
  "analysis_summary": "完整分析逻辑，必须包含对当前持仓（或空仓）的操作指导",
  "risk_alert": "主要风险提示",
  "trend_prediction": "未来 1-4 周趋势判断"
}}
"""


    # ---------- 核心分析 ----------

    def analyze(
        self,
        context: Dict[str, Any],
        custom_prompt: Optional[str] = None
    ) -> Optional[AnalysisResult]:

        if not self.llm:
            return None

        try:
            result_obj = self.llm.invoke(custom_prompt or "请分析股票")
            content = result_obj.content

            # 统一转字符串
            if isinstance(content, list):
                content = "\n".join(
                    str(x.get("text", x)) if isinstance(x, dict) else str(x)
                    for x in content
                )
            else:
                content = str(content)

            # 提取 JSON
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start == -1 or json_end == -1:
                raise ValueError("未检测到 JSON 输出")

            json_text = content[json_start: json_end + 1]
            data = json.loads(json_text)

            # === 自动纠正股票名称 ===
            ai_name = data.get("stock_name")
            final_name = ai_name if ai_name else context.get("stock_name", "Unknown")
            # =====================

            score = int(data.get("sentiment_score", 50))
            score = max(0, min(100, score))

            core_view = data.get("core_view", "见详细分析")

            return AnalysisResult(
                code=context.get("code", "Unknown"),
                name=final_name, # 使用 AI 识别的名称
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("operation_advice", "持有观望"),
                risk_alert=data.get("risk_alert", "暂无"),
                trend_prediction=data.get("trend_prediction", "震荡"),
                analysis_summary=data.get("analysis_summary", ""),
                buy_reason=core_view,
                sell_reason=core_view
            )

        except Exception as e:
            logger.error(f"AI 分析失败，使用兜底结果: {e}")

            # -------- 保底返回 --------
            return AnalysisResult(
                code=context.get("code", "Unknown"),
                name=context.get("stock_name", "Unknown"),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="持有观望",
                risk_alert="AI 输出异常，建议人工复核",
                trend_prediction="不确定",
                analysis_summary="AI 分析失败或输出格式异常",
                buy_reason="暂无明确买入信号",
                sell_reason="暂无明确卖出信号"
            )
