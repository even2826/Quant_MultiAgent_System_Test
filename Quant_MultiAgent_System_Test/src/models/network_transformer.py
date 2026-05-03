import torch
import torch.nn as nn
import math

class AlphaTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(AlphaTransformer, self).__init__()
        self.d_model = d_model
        
        # 1. 特征升维映射
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码 (Time-series 必备)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder 核心层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True  # 设定输入格式为 [batch, seq_len, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. 输出打分层 (用于排序或收益率预测)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # 输出单一的 Alpha Score
        )

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src)
        
        # 取序列最后一个时间步的特征进行预测
        final_step_feature = output[:, -1, :] 
        
        score = self.decoder(final_step_feature)
        return score.squeeze(-1)

# --- 辅助工具：位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)