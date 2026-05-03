# src/agents/agent_risk.py
import logging
from configs.config import STRATEGY_CONFIG, AGENT_CONFIG

class RiskControlAgent:
    def __init__(self):
        # 从统一配置中读取风控阈值
        self.stop_loss = STRATEGY_CONFIG["stop_loss_threshold"]       # 例如 -0.08
        self.take_profit = STRATEGY_CONFIG["take_profit_threshold"]   # 例如 0.15
        
        # 中线大盘风控配置
        self.enable_midterm = AGENT_CONFIG["risk_agent"]["enable_midterm_control"]
        self.fever_threshold = AGENT_CONFIG["risk_agent"]["market_fever_threshold"]
        
        self.logger = logging.getLogger(__name__)

    def check_portfolio_risk(self, portfolio, current_prices):
        """
        个股微观风控：检查当前持仓是否触发止盈或止损
        :param portfolio: 当前持仓列表，格式例如 [{'code': 'sh600519', 'buy_price': 100.0, 'shares': 100}]
        :param current_prices: 当日最新价格字典，格式例如 {'sh600519': 95.0}
        :return: 需要卖出的股票代码列表及卖出原因
        """
        sell_signals = []
        
        for position in portfolio:
            code = position['code']
            buy_price = position['buy_price']
            
            if code not in current_prices:
                continue
                
            current_price = current_prices[code]
            profit_pct = (current_price - buy_price) / buy_price
            
            # 止损逻辑
            if profit_pct <= self.stop_loss:
                self.logger.warning(f"🛡️ [风控智能体] 触发止损！{code} 当前收益 {profit_pct:.2%}，低于阈值 {self.stop_loss:.2%}")
                sell_signals.append({'code': code, 'reason': 'stop_loss'})
            
            # 止盈逻辑
            elif profit_pct >= self.take_profit:
                self.logger.info(f"💰 [风控智能体] 触发止盈！{code} 当前收益 {profit_pct:.2%}，高于阈值 {self.take_profit:.2%}")
                sell_signals.append({'code': code, 'reason': 'take_profit'})
                
        return sell_signals

    def check_market_risk(self, market_metrics):
        """
        大盘宏观风控：判断市场是否过热或处于主跌浪
        :param market_metrics: 市场情绪指标字典，例如 {'market_fever': 0.85}
        :return: bool (True 表示需要空仓或减仓避险)
        """
        if not self.enable_midterm:
            return False
            
        fever = market_metrics.get("market_fever", 0)
        if fever > self.fever_threshold:
            self.logger.warning(f"🚨 [风控智能体] 市场情绪过热 ({fever} > {self.fever_threshold})，建议停止开仓！")
            return True
            
        return False