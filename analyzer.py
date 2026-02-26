#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 分析引擎 - DeepSeek 分析器
"""

import logging
import json
import re
import time
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_config

logger = logging.getLogger(__name__)

# ================= 数据结构 =================

@dataclass
class AnalysisResult:
    """分析结果数据结构"""
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
        """根据情绪分数返回表情符号"""
        if self.sentiment_score >= 80:
            return "🔴"
        if self.sentiment_score <= 40:
            return "🟢"
        return "🟡"


# ================= DeepSeek Analyzer =================

class DeepSeekAnalyzer:
    """DeepSeek 分析引擎"""

    def __init__(self):
        """初始化分析器"""
        self.config = get_config()
        self.llm = None
        self.model_name = "deepseek-chat"
        
        try:
            from openai import OpenAI
            
            # 获取 API 密钥
            api_key = None
            base_url = "https://api.deepseek.com/v1"
            
            if self.config.deepseek_api_key:
                api_key = self.config.deepseek_api_key
                logger.info("✅ 使用 DEEPSEEK_API_KEY 配置")
            elif self.config.openai_api_key:
                api_key = self.config.openai_api_key
                if self.config.openai_base_url:
                    base_url = self.config.openai_base_url
                logger.info(f"✅ 使用 OpenAI 兼容模式，Base URL: {base_url}")
            
            if not api_key:
                logger.error("❌ 未配置 AI API 密钥")
                return
            
            # 设置模型
            if self.config.deepseek_model:
                self.model_name = self.config.deepseek_model
            elif self.config.openai_model:
                self.model_name = self.config.openai_model
            
            # 创建客户端
            self.llm = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=30,
            )
            
            logger.info(f"✅ DeepSeek 分析引擎初始化完成，使用模型: {self.model_name}")
            
        except ImportError:
            logger.error("❌ 未安装 openai 包")
        except Exception as e:
            logger.error(f"❌ DeepSeek 初始化失败: {e}")

    def generate_cio_prompt(self, stock_info, tech_data, trend_context):
        """生成首席投资官提示词"""
        # 获取股票信息
        stock_name = stock_info.get("name", "未知股票")
        stock_code = stock_info.get("code", "Unknown")
        
        # 持仓信息
        cost = float(stock_info.get("cost", 0))
        shares = int(stock_info.get("shares", 0))
        current_price = float(tech_data.get("price", 0))
        
        # 持仓上下文
        if shares > 0 and cost > 0 and current_price > 0:
            profit_pct = (current_price - cost) / cost * 100
            profit_status = "盈利" if profit_pct > 0 else "亏损"
            position_context = f"用户持仓 {shares} 股，成本 {cost:.2f} 元，当前价格 {current_price:.2f} 元，当前{profit_status} {abs(profit_pct):.2f}%。"
        else:
            position_context = "用户当前为空仓，请评估安全边际与建仓方式。"
        
        # 新闻上下文
        macro_news = trend_context.get("macro", "当前宏观面平静")
        sector_news = trend_context.get("sector", "板块暂无重大消息")
        target_sector = trend_context.get("target_sector", "通用")
        
        # 格式化技术指标
        def format_value(val):
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)):
                return f"{val:.2f}"
            return str(val)
        
        ma5 = format_value(tech_data.get("ma5"))
        ma20 = format_value(tech_data.get("ma20"))
        ma60 = format_value(tech_data.get("ma60"))
        rsi = format_value(tech_data.get("rsi"))
        macd = format_value(tech_data.get("macd"))
        support = format_value(tech_data.get("support"))
        support_distance_pct = format_value(tech_data.get("support_distance_pct"))
        resistance = format_value(tech_data.get("resistance"))
        resistance_distance_pct = format_value(tech_data.get("resistance_distance_pct"))
        money_flow_5d = format_value(tech_data.get("money_flow_5d"))
        money_flow_20d = format_value(tech_data.get("money_flow_20d"))
        money_flow_strength = format_value(tech_data.get("money_flow_strength"))
        
        # 构建提示词
        prompt = f"""
你是一位专业的 A 股首席投资官（CIO），拥有 20 年投资经验。

请基于以下三方面信息，对 {stock_name}（{stock_code}）给出明确的交易决策：
1. 市场环境与新闻面
2. 技术分析指标
3. 用户实际持仓情况

=== 市场环境 ===
宏观面：{macro_news}
行业面（{target_sector}）：{sector_news}

=== 技术面分析（日线）===
当前价格：{current_price:.2f}
移动平均线：
  - MA5（短期）：{ma5}
  - MA20（中期）：{ma20}
  - MA60（长期）：{ma60}
相对强弱指数（RSI）：{rsi}
MACD：{macd}
关键技术位：
  - 支撑位：{support}
  - 距离支撑位：{support_distance_pct}%
  - 阻力位：{resistance}
  - 距离阻力位：{resistance_distance_pct}%
资金信号：
  - 近5日净资金流（估算）：{money_flow_5d}
  - 近20日净资金流（估算）：{money_flow_20d}
  - 资金强度（5日净流/5日均额）：{money_flow_strength}

=== 用户持仓情况 ===
{position_context}

=== 输出要求 ===
请以JSON格式返回分析结果，包含以下字段：
{{
  "stock_name": "{stock_name}",
  "sentiment_score": 0-100的整数（80+积极，40-谨慎，中间中性）,
  "operation_advice": "具体的操作建议（如：加仓、减仓、持有、观望等）",
  "core_view": "一句话核心逻辑",
  "analysis_summary": "详细分析（必须结合：撑压位置、资金进出、持仓成本、新闻政策面）",
  "position_advice": "围绕当前持仓给出增持/减持/持有建议与仓位理由（若空仓给出分批建仓建议）",
  "risk_alert": "需要关注的主要风险",
  "trend_prediction": "未来1周走势预测"
}}

请重点说明：
1) 当前价与支撑/压力的距离对风险收益比的影响；
2) 资金净流入/流出与趋势是否共振；
3) 若用户有持仓成本，是否应增持、减持、还是继续持有，并给出触发条件。

请确保只返回有效的JSON格式，不要包含其他解释性文字。
"""
        
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_ai_api(self, prompt):
        """调用AI API（带重试机制）"""
        if not self.llm:
            raise ValueError("AI客户端未初始化")
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一位专业的股票分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"AI API调用失败，将重试: {e}")
            raise

    def analyze(self, context, custom_prompt=None):
        """分析股票数据"""
        if not self.llm:
            logger.error("AI客户端未初始化，无法进行分析")
            return self._create_error_result(context, "AI客户端未初始化")

        try:
            if not custom_prompt:
                return self._create_error_result(context, "未提供分析提示词")
            
            logger.info(f"开始AI分析，股票: {context.get('code', '未知')}")
            start_time = time.time()
            
            ai_response = self._call_ai_api(custom_prompt)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"AI分析完成，耗时: {elapsed_time:.2f}秒")
            
            if not ai_response:
                logger.error("AI返回空响应")
                return self._create_error_result(context, "AI返回空响应")
            
            # 解析AI响应
            result = self._parse_ai_response(ai_response, context)
            
            if result:
                logger.info(f"股票 {result.code} 分析完成，情绪分数: {result.sentiment_score}")
            else:
                logger.warning(f"股票 {context.get('code')} 分析结果解析失败")
                
            return result
            
        except Exception as e:
            logger.error(f"AI分析过程失败: {e}")
            return self._create_error_result(context, str(e))

    def _parse_ai_response(self, ai_response, context):
        """解析AI返回的响应"""
        try:
            # 提取JSON
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                logger.error(f"无法从AI响应中提取JSON: {ai_response[:200]}...")
                return None
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # 验证必要字段
            required_fields = ["sentiment_score", "operation_advice", "core_view", 
                             "analysis_summary", "risk_alert", "trend_prediction"]
            
            for field in required_fields:
                if field not in data:
                    logger.warning(f"AI响应缺少字段: {field}")
                    data[field] = "未知"
            
            # 处理情绪分数
            try:
                score = int(data["sentiment_score"])
                score = max(0, min(100, score))
            except:
                score = 50
                logger.warning("情绪分数解析失败，使用默认值50")
            
            # 获取股票名称
            stock_name = data.get("stock_name", "")
            if not stock_name:
                stock_name = context.get("stock_name", f"股票{context.get('code', '')}")
            
            # 创建分析结果
            return AnalysisResult(
                code=context.get("code", ""),
                name=stock_name,
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("position_advice") or data.get("operation_advice", "观望"),
                risk_alert=data.get("risk_alert", ""),
                trend_prediction=data.get("trend_prediction", ""),
                analysis_summary=data.get("analysis_summary", ""),
                buy_reason=data.get("core_view", ""),
                sell_reason=data.get("core_view", ""),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"解析AI响应时出错: {e}")
            return None

    def _create_error_result(self, context, error_msg):
        """创建错误结果"""
        return AnalysisResult(
            code=context.get("code", ""),
            name=context.get("stock_name", ""),
            date=context.get("date", ""),
            sentiment_score=50,
            operation_advice="人工复核",
            risk_alert=error_msg,
            trend_prediction="不确定",
            analysis_summary="AI分析失败，需要人工检查",
        )


# ================= 向后兼容 =================

GeminiAnalyzer = DeepSeekAnalyzer
