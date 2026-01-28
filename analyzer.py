# -*- coding: utf-8 -*-
import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    code: str
    name: str
    date: str
    sentiment_score: int
    operation_advice: str
    risk_alert: str
    trend_prediction: str
    analysis_summary: str  # <--- 修改点：从 summary 改为 analysis_summary
    
    def get_emoji(self):
        if self.sentiment_score >= 80: return "🔴"  # 强烈看多
        if self.sentiment_score <= 40: return "🟢"  # 看空/风险
        return "🟡"  # 观望

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.config = get_config()
        self.api_key = api_key or self.config.gemini_api_key
        
        if not self.api_key:
            logger.warning("Gemini API Key 未配置")
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                google_api_key=self.api_key,
                temperature=0.2, 
                timeout=60
            )

    def generate_cio_prompt(self, stock_info: dict, tech_data: dict, trend_context: dict) -> str:
        """生成 AI-CIO (首席投资官) 专用提示词"""
        macro_text = trend_context.get('macro', '当前无重大宏观消息。')
        sector_text = trend_context.get('sector', '当前板块无重大特定消息。')
        
        prompt = f"""
        # 角色设定
        你是一位拥有20年经验的**宏观对冲基金经理 (CIO)**。你的投资哲学是 **"自上而下 (Top-Down)"**：先看宏观天象，再看行业赛道，最后看个股形态。
        你极其厌恶风险，只有当"宏观逻辑"与"技术形态"共振时，你才会建议买入。

        # 1. 输入数据
        
        ## A. 宏观与行业情报 (TrendRadar)
        * **宏观环境**: {macro_text}
        * **{stock_info.get('sector', '未知')} 板块动态**: {sector_text}

        ## B. 标地资产技术面 ({stock_info.get('name')} - {stock_info.get('code')})
        * **持仓策略**: {stock_info.get('strategy', '未定义')} (成本: {stock_info.get('cost', 0)})
        * **当前价格**: {tech_data.get('price', 'N/A')} (涨跌幅: {tech_data.get('change_pct', 0):.2f}%)
        * **趋势状态**: {tech_data.get('trend', '未知')}
        * **均线系统**: MA5={tech_data.get('ma5', 0):.2f}, MA20={tech_data.get('ma20', 0):.2f}, MA60={tech_data.get('ma60', 0):.2f}
        * **关键指标**: 
            - RSI(14): {tech_data.get('rsi', 50):.2f} (>70超买, <30超卖)
            - MACD: {tech_data.get('macd', 0):.2f} (信号线: {tech_data.get('macd_signal', 0):.2f})
            - 量比: {tech_data.get('vol_ratio', 0):.2f} (>1.5为放量)
        * **关键点位**: 强支撑 {tech_data.get('support', 0)}, 强阻力 {tech_data.get('resistance', 0)}

        # 2. 分析任务 (请严格按步骤推理)

        ## 第一步：宏观一致性检查 (Consistency Check)
        * 判断当前宏观环境（利率、通胀、地缘）对该板块是"顺风"(Tailwind) 还是 "逆风"(Headwind)？
        * **警示**: 如果宏观是逆风，但技术面在上涨，这是否是"诱多"陷阱？

        ## 第二步：技术面深度诊断
        * **趋势力度**: 均线是发散还是纠缠？MACD是否背离？
        * **量价配合**: 上涨是否放量？下跌是否缩量？
        * **持仓建议**: 现价距离成本价的位置，结合支撑压力位，盈亏比如何？

        ## 第三步：交易指令 (Output)
        请输出最终决策，必须包含：
        1. **核心观点**: 一句话总结。
        2. **评分**: 0-100分。
        3. **操作建议**: [强力买入/逢低吸纳/持有观望/逢高减仓/清仓止损]。
        4. **关键点位**: 止损位、阻力位。
        
        请用**专业、犀利、客观**的金融术语回答。
        """
        return prompt

    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None, custom_prompt: Optional[str] = None) -> Optional[AnalysisResult]:
        if not self.llm:
            return None
            
        try:
            if custom_prompt:
                final_prompt = custom_prompt
            else:
                return None

            # 调用 AI
            result_obj = self.llm.invoke(final_prompt)
            response = result_obj.content
            
            # 类型转换修复
            if isinstance(response, list):
                response = "\n".join([str(item) for item in response])
            elif not isinstance(response, str):
                response = str(response)
            
            # 解析 AI 返回
            score_match = re.search(r'评分[:：]\s*(\d+)', response)
            score = int(score_match.group(1)) if score_match else 50
            
            advice_match = re.search(r'操作建议[:：]\s*\[?(.*?)\]?', response)
            advice = advice_match.group(1).strip() if advice_match else "观望"

            return AnalysisResult(
                code=context.get('code', 'Unknown'),
                name=context.get('stock_name', 'Unknown'),
                date=context.get('date', ''),
                sentiment_score=score,
                operation_advice=advice,
                risk_alert="详见总结",
                trend_prediction="详见总结",
                analysis_summary=response  # <--- 修改点：从 summary 改为 analysis_summary
            )
            
        except Exception as e:
            logger.error(f"AI 分析异常: {e}")
            return None
