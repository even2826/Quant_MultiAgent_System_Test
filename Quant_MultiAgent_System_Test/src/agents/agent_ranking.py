import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import math # 确保导入 math

# ================= 配置区 =================
DATA_DIR = Path(r'D:\QuantData')
SEQ_LEN = 15 #需要和训练时的序列长度保持一致         
FEATURE_CHANNELS = 9 # ⚠️ 特征同步升级为 9 维

# 限定训练数据范围，专注近期逻辑
TRAIN_START_DATE = '2020-01-01'  
TRAIN_END_DATE = '2024-12-31'

# 🚀 修复 1：加上了 rf 前缀，确保变量能被正确解析
DEFAULT_MODEL_PATH = Path(rf'D:\量化专用文件\2604\0_weights\ranking_transformer_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}.pth')
# ==========================================

# ================= Transformer 核心架构定义 =================
class PositionalEncoding(nn.Module):
    """位置编码器：让 Transformer 拥有时间顺序感"""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class StockTransformer(nn.Module):
    """专为 A 股量价数据设计的轻量级 Transformer"""
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.3):
        super(StockTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid() # 🚀 修复 4：推理时必须加回 Sigmoid，将输出压缩至 0~1 的概率区间
        )

    def forward(self, x):
        x = self.input_projection(x)          
        x = self.pos_encoder(x)               
        transformer_out = self.transformer_encoder(x) 
        last_step_out = transformer_out[:, -1, :]     
        return self.fc(last_step_out)                 

class RankingAgent:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        # 🚀 修复 2：先定义 device，再去加载 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StockTransformer().to(self.device)
        self.is_trained = False
        
        model_file = Path(model_path) if model_path else None
        
        if model_file and model_file.exists():
            # 兼容：如果保存的字典里多出了 Sigmoid 的 key，我们设置 strict=False 强行加载有效权重
            self.model.load_state_dict(torch.load(model_file, map_location=self.device), strict=False)
            self.is_trained = True
            print(f"✅ 成功加载预训练深度学习模型: {model_file.name}")
        else:
            print(f"⚠️ 警告: 未找到预训练权重 ({model_file})，模型输出为随机值。")
            
        self.model.eval()

    def _prepare_features(self, df_slice):
        """特征工程升维：拼接 9 个特征供 AI 推理"""
        base_price = df_slice['close'].iloc[0]
        if base_price == 0: return None
        
        # 1. 基础价量特征 (5维)
        p_features = df_slice[['open', 'high', 'low', 'close']] / base_price - 1
        v_mean = df_slice['volume'].mean()
        v_features = df_slice['volume'] / (v_mean + 1e-6)
        
        # 2. 趋势特征 (1维)
        # 🚀 修复 5：使用每天对应的 MA60 数组，而不是用最后一天作为全局除数
        trend_features = df_slice['close'] / df_slice['ma60'] - 1
        
        # 3. 提取在外层已算好的高阶特征 (3维)
        amp_features = df_slice['amp_feat']
        macd_features = df_slice['macd_feat']
        body_features = df_slice['body_feat']
        
        features = pd.concat([
            p_features, v_features, trend_features, 
            amp_features, macd_features, body_features
        ], axis=1).values
        
        if not np.isfinite(features).all():
            return None
            
        # 🚀 修复 3：将数据推送到对应的 device (GPU/CPU)
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def rank_market(self, target_date, stock_list):
        print(f"🧠 排序智能体正在分析 {len(stock_list)} 只标的...")
        scores = []
        
        for code in stock_list:
            try:
                path = DATA_DIR / f"code={code}" / "data.zstd.parquet"
                if not path.exists(): continue
                
                df = pd.read_parquet(path)
                df_hist = df[df['date'] <= pd.Timestamp(target_date)].tail(150).copy()
                
                if len(df_hist) < 100: continue
                
                df_hist['ma60'] = df_hist['close'].rolling(60).mean()
                df_hist['ema12'] = df_hist['close'].ewm(span=12, adjust=False).mean()
                df_hist['ema26'] = df_hist['close'].ewm(span=26, adjust=False).mean()
                df_hist['macd_feat'] = (df_hist['ema12'] - df_hist['ema26']) / df_hist['close']
                df_hist['body_feat'] = (df_hist['close'] - df_hist['open']) / df_hist['open']
                df_hist['amp_feat'] = df_hist['high'] / df_hist['low'] - 1
                
                # -----------------------------------------------------
                # 🛡️ 强规则趋势护栏
                # -----------------------------------------------------
                latest_close = df_hist['close'].iloc[-1]
                ma20 = df_hist['close'].rolling(20).mean().iloc[-1]
                ma60 = df_hist['ma60'].iloc[-1]
                
                if latest_close < ma20 or latest_close < ma60:
                    continue
                
                ma60_past = df_hist['ma60'].iloc[-5] 
                if ma60 <= ma60_past:
                    continue
                # -----------------------------------------------------

                df_slice = df_hist.tail(SEQ_LEN)
                # 修改点：_prepare_features 现在只需要传入 df_slice
                x = self._prepare_features(df_slice)
                
                if x is None:
                    continue
                
                with torch.no_grad():
                    alpha_score = self.model(x).item()
                
                if not self.is_trained:
                    momentum = latest_close / df_slice['close'].iloc[0] - 1
                    alpha_score = momentum 

                scores.append({'code': code, 'alpha': alpha_score})
                
            except Exception as e:
                continue
        
        if not scores:
            print("⚠️ 经过趋势护栏过滤后，没有股票符合多头条件。")
            return pd.DataFrame(columns=['code', 'alpha'])
            
        rank_df = pd.DataFrame(scores).sort_values('alpha', ascending=False)
        return rank_df