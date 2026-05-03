# src/engine/backtest_engine.py
import pandas as pd
import logging
from configs.config import STRATEGY_CONFIG

# 引入重构后的三大智能体
from src.agents.agent_macro import MacroDynamicAgent
from src.agents.agent_ranking import DeepRankingAgent
from src.agents.agent_risk import RiskControlAgent

class BacktestEngine:
    def __init__(self):
        # 1. 基础配置
        self.start_date = STRATEGY_CONFIG["start_date"]
        self.end_date = STRATEGY_CONFIG["end_date"]
        self.capital = STRATEGY_CONFIG["initial_capital"]
        self.max_holdings = STRATEGY_CONFIG["max_holdings"]
        self.commission = STRATEGY_CONFIG["commission_rate"]
        
        # 2. 状态记录
        self.portfolio = []    # 持仓列表: [{'code': 'sh...', 'buy_price': 10.0, 'shares': 100}]
        self.watch_list = []   # 明日观察池 (由昨日盘后生成)
        
        # 3. 初始化智能体矩阵
        self.macro_agent = MacroDynamicAgent()
        self.ranking_agent = DeepRankingAgent()
        self.risk_agent = RiskControlAgent()
        
        # 4. 日志配置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _get_simulated_prices(self, date_str, codes):
        """模拟获取当日价格（实际项目中需从 data_pipeline 获取）"""
        return {code: 10.0 for code in codes} # 假设全是10元，仅作演示

    def run(self):
        self.logger.info(f"🚀 启动 Multi-Agent 回测引擎...")
        self.logger.info(f"💵 初始资金: {self.capital} | 回测区间: {self.start_date} 至 {self.end_date}")
        
        # 生成交易日历 (实际需根据真实行情数据日历生成)
        trading_calendar = pd.date_range(start=self.start_date, end=self.end_date, freq='B')

        for current_date in trading_calendar:
            date_str = current_date.strftime("%Y-%m-%d")
            self.logger.info(f"\n========== 【交易日: {date_str}】 ==========")

            # ---------------------------------------------------------
            # 阶段一：盘前宏观感知与自上而下选股 (Top-Down)
            # ---------------------------------------------------------
            # 1. 宏观智能体：定位当日主线板块
            top_sectors = self.macro_agent.analyze_sectors(date_str)
            
            # 2. 排序智能体：仅在最强板块内进行标的深度排序
            current_day_sector_rankings = {}
            for sector in top_sectors:
                # 模拟获取该板块的全部股票
                sector_stocks = [f"stock_{sector}_1", f"stock_{sector}_2", f"stock_{sector}_3", f"stock_{sector}_4", "stock_dummy"] 
                
                # 仅针对板块内进行 Transformer 打分排序，拒绝全市场扫描
                ranked_stocks = self.ranking_agent.rank_within_sector(sector, sector_stocks, date_str)
                current_day_sector_rankings[sector] = ranked_stocks

            # ---------------------------------------------------------
            # 阶段二：盘中交易执行与风控 (持仓处理)
            # ---------------------------------------------------------
            # 假设获取了当前持仓的最新真实价格
            held_codes = [p['code'] for p in self.portfolio]
            current_prices = self._get_simulated_prices(date_str, held_codes)

            # 1. 大盘风控检查 (例如市场极度恐慌时暂停买入)
            market_metrics = {"market_fever": 0.5} # 模拟指标
            halt_buying = self.risk_agent.check_market_risk(market_metrics)

            # 2. 个股风控检查 (止盈止损)
            sell_signals = self.risk_agent.check_portfolio_risk(self.portfolio, current_prices)
            
            # 执行卖出
            for signal in sell_signals:
                code_to_sell = signal['code']
                # 从持仓中移除并回笼资金 (伪代码简化计算)
                for p in self.portfolio:
                    if p['code'] == code_to_sell:
                        sell_val = p['shares'] * current_prices[code_to_sell] * (1 - self.commission)
                        self.capital += sell_val
                        self.portfolio.remove(p)
                        self.logger.info(f"📉 执行卖出: {code_to_sell}，原因: {signal['reason']}，当前资金: {self.capital:.2f}")

            # 3. 执行买入 (从昨天的观察池 watch_list 中挑选)
            if not halt_buying:
                available_slots = self.max_holdings - len(self.portfolio)
                if available_slots > 0 and self.watch_list:
                    # 取观察池头部标的买入
                    buy_candidates = self.watch_list[:available_slots]
                    target_prices = self._get_simulated_prices(date_str, buy_candidates)
                    
                    for code in buy_candidates:
                        # 简单等权仓位分配
                        cash_per_stock = self.capital / (available_slots + len(self.portfolio))
                        buy_price = target_prices.get(code, 10.0)
                        shares = int(cash_per_stock / buy_price / 100) * 100 # 按手买入
                        
                        if shares > 0:
                            cost = shares * buy_price * (1 + self.commission)
                            self.capital -= cost
                            self.portfolio.append({'code': code, 'buy_price': buy_price, 'shares': shares})
                            self.logger.info(f"📈 执行买入: {code}，价格: {buy_price}，数量: {shares}，剩余资金: {self.capital:.2f}")

            self.logger.info(f"💼 盘中执行完毕，当前持仓数量: {len(self.portfolio)} / {self.max_holdings}")

            # ---------------------------------------------------------
            # 阶段三：盘后决策阶段 (生成明日观察池) —— 🔥 BUG 修复核心区 🔥
            # ---------------------------------------------------------
            # 直接复用今日盘前算好的 current_day_sector_rankings，不再调用全局扫描
            if current_day_sector_rankings:
                new_watch_list = []
                for sector, ranked_stocks in current_day_sector_rankings.items():
                    # 提取每个板块的前 2 名作为明日重点观察
                    new_watch_list.extend(ranked_stocks[:2]) 
                    
                # 去重并更新观察池
                self.watch_list = list(dict.fromkeys(new_watch_list)) 
                self.logger.info(f"🌙 盘后决策：直接复用自上而下结果，生成明日观察池，共 {len(self.watch_list)} 只标的。")
            else:
                self.logger.warning("🌙 未获取到当日板块排序结果，维持原有观察池。")

        # 回测总结
        self.logger.info("\n✨ ================= 回测结束 =================")
        # 计算期末总资产
        final_held_codes = [p['code'] for p in self.portfolio]
        final_prices = self._get_simulated_prices(self.end_date, final_held_codes)
        portfolio_value = sum(p['shares'] * final_prices.get(p['code'], p['buy_price']) for p in self.portfolio)
        total_assets = self.capital + portfolio_value
        
        self.logger.info(f"🏁 初始资金: {STRATEGY_CONFIG['initial_capital']:.2f}")
        self.logger.info(f"🏁 期末总资产: {total_assets:.2f}")
        self.logger.info(f"🏁 策略总收益率: {(total_assets / STRATEGY_CONFIG['initial_capital'] - 1) * 100:.2f}%")