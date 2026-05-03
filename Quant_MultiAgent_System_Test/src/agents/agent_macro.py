import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from pathlib import Path
from configs.config import SYSTEM_CONFIG, AGENT_CONFIG
import warnings

# ================= 配置区 =================
DATA_DIR = Path(SYSTEM_CONFIG["data_dir"])
INDEX_300 = 'sh000300'
INDEX_1000 = 'sh000852'
# ==========================================

# 忽略计算过程中的警告
warnings.filterwarnings('ignore')

class MacroDynamicAgent:
    def __init__(self):
        self.data_dir = Path(SYSTEM_CONFIG["data_dir"])
        self.top_n = AGENT_CONFIG["macro_agent"]["top_n_sectors"]
        self.lookback = AGENT_CONFIG["macro_agent"]["lookback_days"]
        self.history_amounts = self._preload_history_amounts()

    def _preload_history_amounts(self):
        """
        核心修复：通过文件路径列表加载数据集，彻底避开 metadata 文件夹
        """
        print(f"📊 正在建立动态流动性参考坐标系 (扫描最近 {self.lookback} 日数据)...")
        
        # 1. 只搜寻 code= 开头的文件夹下的 parquet 文件
        # 这样会自动忽略 metadata、logs 等其他文件夹
        file_list = []
        for prefix in ['sh60', 'sh68', 'sz00', 'sz30']: # 只看 A 股
            file_list.extend(list(self.data_dir.glob(f'code={prefix}*/data.zstd.parquet')))
            
        if not file_list:
            raise FileNotFoundError(f"在 {self.data_dir} 下未找到 A 股 Parquet 数据文件。")

        # 2. 传入具体的【文件列表】，并指定 partitioning="hive"
        # PyArrow 会根据文件路径中的 code=xxx 自动提取 code 标签
        dataset = ds.dataset(file_list, format="parquet", partitioning="hive")
        
        # 3. 提取数据
        table = dataset.to_table(columns=['date', 'amount', 'code'])
        df_all = table.to_pandas()
        
        # 4. 聚合每日成交总额 (亿元)
        daily_amounts = df_all.groupby('date')['amount'].sum() / 1e8
        return daily_amounts.sort_index().tail(self.lookback)

    def get_dynamic_liquidity_score(self, today_vol):
        """基于分位数的动态流动性评分"""
        if self.history_amounts.empty: return 50, 50
        
        rank = (self.history_amounts < today_vol).sum()
        percentile = (rank / len(self.history_amounts)) * 100
        
        if percentile >= 90:   score = 100
        elif percentile >= 60: score = 80
        elif percentile >= 30: score = 50
        elif percentile >= 10: score = 30
        else:                  score = 10
        return score, percentile

    def get_market_snapshot(self, target_date):
        """修复版：精准获取当日个股快照，避开干扰文件"""
        file_list = []
        for prefix in ['sh60', 'sh68', 'sz00', 'sz30']:
            file_list.extend(list(self.data_dir.glob(f'code={prefix}*/data.zstd.parquet')))
            
        dataset = ds.dataset(file_list, format="parquet", partitioning="hive")
        table = dataset.to_table(
            filter=ds.field("date") == pd.Timestamp(target_date),
            columns=['amount', 'code']
        )
        return table.to_pandas()

    def get_index_trend(self, target_date):
        """计算宽基指数趋势"""
        scores = []
        for code in [INDEX_300, INDEX_1000]:
            path = self.data_dir / f"code={code}" / "data.zstd.parquet"
            if not path.exists(): continue
            df = pd.read_parquet(path)
            df = df[df['date'] <= pd.Timestamp(target_date)].tail(61)
            if len(df) < 60: continue
            ma60 = df['close'].rolling(60).mean().iloc[-1]
            current = df['close'].iloc[-1]
            scores.append(50 if current > ma60 else 0)
        return sum(scores) if scores else 0
    
    def get_strongest_sectors(self, target_date):
        """
        自上而下板块轮动：计算波动率调整后的动能，返回最强势的板块前缀
        """
        MOMENTUM_WINDOW = 20
        # 板块指数与 A股股票前缀的硬映射
        INDEX_MAPPING = {
            'sh000016': 'sh60',  # 上证50 -> 沪市大盘
            'sh000852': 'sz00',  # 中证1000 -> 深市中小盘
            'sz399006': 'sz30',  # 创业板指 -> 创业板
            'sh000688': 'sh68'   # 科创50 -> 科创板
        }
        
        strength_scores = {}
        for idx_code, prefix in INDEX_MAPPING.items():
            file_path = self.data_dir / f"code={idx_code}" / "data.zstd.parquet"
            if not file_path.exists(): continue
            
            # 读取目标日期前的数据
            df = pd.read_parquet(file_path, columns=['date', 'close'])
            df = df[df['date'] <= pd.Timestamp(target_date)].tail(MOMENTUM_WINDOW + 5)
            if len(df) < MOMENTUM_WINDOW: continue
            
            close = df['close']
            momentum = close.iloc[-1] / close.iloc[-MOMENTUM_WINDOW] - 1
            daily_ret = close.pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252)
            
            # 动能 / 波动率 得出强度分
            score = momentum / (volatility + 1e-6)
            
            # 如果有多个指数指向同一前缀，保留最高分
            if prefix in strength_scores:
                strength_scores[prefix] = max(strength_scores[prefix], score)
            else:
                strength_scores[prefix] = score
            
        if not strength_scores:
            # 如果没有找到任何指数数据，默认放行所有板块
            return ['sh60', 'sz00', 'sz30', 'sh68'] 
            
        # 找出得分最高的板块
        best_prefix = max(strength_scores, key=strength_scores.get)
        
        # 🚨 极寒风控：如果全市场最强的板块强度分都 < 0，说明覆巢之下无完卵
        if strength_scores[best_prefix] < 0:
            return [] # 返回空列表，告诉引擎彻底空仓
            
        return [best_prefix]

    def calculate_exposure(self, target_date):
        print(f"\n🕵️ 宏观智能体正在深度扫描 {target_date}...")
        
        # 1. 趋势维度 (40%)
        trend_score = self.get_index_trend(target_date)
        
        # 2. 动态流动性维度 (60%)
        today_df = self.get_market_snapshot(target_date)
        if today_df.empty:
            print("⚠️ 未找到当日市场快照，请检查日期或数据。")
            return 0.0
            
        today_vol = today_df['amount'].sum() / 1e8
        liq_score, pct = self.get_dynamic_liquidity_score(today_vol)
        median_val = today_df['amount'].median() / 1e4 
        
        drought_penalty = 20 if median_val < 1500 else 0
        raw_score = (trend_score * 0.4 + liq_score * 0.6) - drought_penalty
        final_exposure = max(0, min(100, raw_score)) / 100.0
        
        print("-" * 50)
        print(f"📊 宏观动态感知报告：")
        print(f"- 趋势状态：{'健康' if trend_score >= 50 else '风险(破位)'} ({trend_score}分)")
        print(f"- 成交总额：{today_vol:.0f} 亿 (处于年度 {pct:.1f}% 分位)")
        print(f"- 体感热度：中位数标的成交 {median_val:.0f} 万元")
        print(f"- 最终权重：{final_exposure:.1%}")
        
        if final_exposure >= 0.8:
            msg = "🚀 进攻：环境极佳，允许重仓。"
        elif final_exposure >= 0.4:
            msg = "🛡️ 均衡：环境一般，建议控制总位。"
        else:
            msg = "❄️ 防御：环境恶劣，建议空仓避险。"
        print(f"💡 决策建议：{msg}")
        print("-" * 50)
        
        return final_exposure
