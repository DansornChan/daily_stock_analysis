# -*- coding: utf-8 -*-
import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
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
    analysis_summary: str  # <--- 统一命名，解决 AttributeError
    
    def get_emoji(self):
        if self.sentiment_score >= 80: return "🔴"
        if self.sentiment_score <= 40: return "🟢"
        return "🟡"

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.config = get_config()
        self.api_key = api_key or self.config.gemini_api_key
        
        if not self.api_key:
            logger.warning("Gemini API Key 未配置")
            self.llm = None
        else:
            # 降低温度，增加稳定性
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                google_api_key=self.api_key,
                temperature=0.1, 
                timeout=120
            )

    def generate_cio_prompt(self, stock_info: dict, tech_data: dict, trend_context: dict) -> str:
        macro_text = trend_context.get('macro', '无重大消息')
        sector_text = trend_context.get('sector', '无重大消息')
        
        return f"""
        你是一位资深基金经理(CIO)。请基于以下数据进行自上而下的深度复盘：
        
        【宏观/行业背景】: {macro_text} | {sector_text}
        【个股技术面】: {stock_info['name']}({stock_info['code']}) 现价{tech_data.get('price')}
        指标: MA5/20/60={tech_data.get('ma5')}/{tech_data.get('ma20')}/{tech_data.get('ma60')}, RSI={tech_data.get('rsi')}, MACD={tech_data.get('macd')}
        
        请输出：
        1. 评分: 0-100
        2. 操作建议: [强力买入/逢低吸纳/持有观望/逢高减仓/清仓止损]
        3. 核心观点: 一句话总结
        4. 详细逻辑: 结合宏观与技术面。
        """

    def analyze(self, context: Dict[str, Any], custom_prompt: Optional[str] = None) -> Optional[AnalysisResult]:
        if not self.llm: return None
        try:
            # 执行 AI 调用
            result_obj = self.llm.invoke(custom_prompt or "分析股票")
            response = result_obj.content
            
            # 强制转换为字符串，解决 'list' 报错
            if isinstance(response, list):
                response = "\n".join([str(x.get('text', x) if isinstance(x, dict) else x) for x in response])
            else:
                response = str(response)

            # 解析评分
            score_match = re.search(r'评分[:：]\s*(\d+)', response)
            score = int(score_match.group(1)) if score_match else 50
            
            # 解析建议
            advice_match = re.search(r'操作建议[:：]\s*\[?(.*?)\]?(\n|$)', response)
            advice = advice_match.group(1).strip() if advice_match else "观望"

            return AnalysisResult(
                code=context.get('code', 'Unknown'),
                name=context.get('stock_name', 'Unknown'),
                date=context.get('date', ''),
                sentiment_score=score,
                operation_advice=advice,
                risk_alert="见分析",
                trend_prediction="见分析",
                analysis_summary=response
            )
        except Exception as e:
            logger.error(f"AI 分析异常: {e}")
            return None
