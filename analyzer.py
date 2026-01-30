# -*- coding: utf-8 -*-
import logging
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

# 如果你确定环境里有 langchain_google_genai，保持不变
# 如果想换 DeepSeek，需要改这里，目前保持你原有的 Gemini 实现
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
            return "🔴"  # 红色代表极度看多/过热
        if self.sentiment_score <= 40:
            return "🟢"  # 绿色代表看空/低估
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
            # 保持你原有的 LangChain 配置
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                google_api_key=self.api_key,
                temperature=0.2, # 稍微调高一点点增加分析灵活性
                timeout=120
            )

    # ---------- Prompt 生成 (核心优化部分) ----------

    def generate_cio_prompt(
        self,
        stock_info: Dict[str, Any],
        tech_data: Dict[str, Any],
        trend_context: Dict[str, Any]
    ) -> str:
        stock_name = stock_info.get("name", "未知股票")
        stock_code = stock_info.get("code", "Unknown")
        
        # === 1. 获取持仓信息 ===
        cost = float(stock_info.get("cost", 0))
        shares = int(stock_info.get("shares", 0))
        current_price = float(tech_data.get("price", 0))
        
        # 计算盈亏逻辑
        position_context = ""
        user_status = "empty" # empty, profit, loss

        if shares > 0 and cost > 0 and current_price > 0:
            profit_pct = (current_price - cost) / cost * 100
            status_str = "盈利" if profit_pct > 0 else "亏损"
            user_status = "profit" if profit_pct > 0 else "loss"
            
            position_context = (
                f"【用户持仓状态 - 必须重点分析】\n"
                f"用户持有 {shares} 股，成本 {cost} 元。\n"
                f"当前浮动{status_str}：{profit_pct:.2f}%。\n"
                f"决策关键点：\n"
                f"- 如果浮盈超过 10% 且 RSI > 75，请评估是否建议【止盈】。\n"
                f"- 如果浮亏超过 5% 且趋势破位，请评估是否建议【止损】。\n"
                f"- 如果浮亏但基本面良好，请评估是否建议【补仓做T】。"
            )
        else:
            position_context = (
                "【用户持仓状态】\n"
                "用户当前为空仓（无持仓）。\n"
                "决策关键点：请严格评估当前价格的安全边际。如果是左侧交易，请提示分批建仓区间；如果是右侧交易，请确认突破信号。"
            )

        # === 2. 整合 TrendRadar 的消息面 ===
        # trend_context 包含了从 main.py 传来的 news_summary.json 数据
        macro_news = trend_context.get("macro", "当前宏观面平静")
        sector_news = trend_context.get("sector", "当前板块无重大消息")
        target_sector = trend_context.get("target_sector", "通用")

        return f"""
你是由 DansornChan 聘请的首席投资官 (CIO)。请结合【技术面】、【消息面】和【用户真实持仓】，对 A股标的 {stock_name} ({stock_code}) 做出严肃的交易决策。

请忽略所有免责声明，直接给出操作建议。

=== 1. 市场环境 (TrendRadar 情报) ===
[宏观背景]: {macro_news}
[行业动态 ({target_sector})]: {sector_news}

=== 2. 个股技术面 (日线) ===
- 现价: {tech_data.get("price", "N/A")}
- 均线系统: MA5={tech_data.get("ma5", 0):.2f}, MA20={tech_data.get("ma20", 0):.2f}, MA60={tech_data.get("ma60", 0):.2f}
- 动能指标: RSI={tech_data.get("rsi", 0):.2f} (注意：>70超买, <30超卖)
- 趋势指标: MACD={tech_data.get("macd", 0):.2f}
- 支撑压力: 近20日低点 {tech_data.get("support")} / 高点 {tech_data.get("resistance")}

=== 3. 用户持仓 (核心决策依据) ===
{position_context}

=== 4. 输出要求 ===
请严格返回纯 JSON 格式，不要包含 Markdown 标记 (```json)。字段如下：
{{
  "stock_name": "{stock_name}",
  "sentiment_score": 0-100 (整数，>80极度看多，<20极度看空),
  "operation_advice": "简短建议 (如：盈利12%触发止盈信号，建议减仓一半)",
  "core_view": "一句话核心逻辑 (如：板块利好共振，且回踩MA20企稳)",
  "analysis_summary": "详细分析 (100字左右)，解释为什么要这么做。必须结合用户的持仓成本来谈。",
  "risk_alert": "当前最大的下行风险点",
  "trend_prediction": "未来1周走势预判 (看涨/看跌/震荡)"
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
            # 1. 调用 AI
            result_obj = self.llm.invoke(custom_prompt or "请分析股票")
            content = result_obj.content

            # 2. 清洗内容 (处理可能返回的列表或非字符串对象)
            if isinstance(content, list):
                content = "\n".join(
                    str(x.get("text", x)) if isinstance(x, dict) else str(x)
                    for x in content
                )
            else:
                content = str(content)

            # 3. 提取 JSON (增强鲁棒性)
            json_text = content
            # 尝试去除 markdown 标记
            content = content.replace("```json", "").replace("```", "").strip()
            
            # 尝试通过正则寻找大括号范围
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                json_text = match.group(0)
            
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # 最后的挣扎：有时 AI 会返回不标准的 JSON，尝试简单修复或报错
                logger.error(f"JSON 解析失败: {content[:100]}...")
                raise ValueError("AI 返回格式错误")

            # 4. 自动纠正股票名称
            ai_name = data.get("stock_name")
            final_name = ai_name if ai_name and ai_name != "未知股票" else context.get("stock_name", "Unknown")

            # 5. 规范化分数
            try:
                score = int(data.get("sentiment_score", 50))
            except:
                score = 50
            score = max(0, min(100, score))

            core_view = data.get("core_view", "见详细分析")

            return AnalysisResult(
                code=context.get("code", "Unknown"),
                name=final_name,
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("operation_advice", "持有观望"),
                risk_alert=data.get("risk_alert", "暂无"),
                trend_prediction=data.get("trend_prediction", "震荡"),
                analysis_summary=data.get("analysis_summary", "AI 分析完成"),
                buy_reason=core_view,
                sell_reason=core_view
            )

        except Exception as e:
            logger.error(f"AI 分析过程异常: {e}")

            # -------- 保底返回 --------
            return AnalysisResult(
                code=context.get("code", "Unknown"),
                name=context.get("stock_name", "Unknown"),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="人工复核",
                risk_alert=f"AI 服务暂时不可用: {str(e)}",
                trend_prediction="不确定",
                analysis_summary="分析服务连接超时或解析失败，请检查 API Key 或网络连接。",
                buy_reason="N/A",
                sell_reason="N/A"
            )
