import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path

# 引入统一配置
from configs.config import SYSTEM_CONFIG, MODEL_CONFIG
from src.models.network_transformer import AlphaTransformer

class ModelTrainer:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.save_dir = Path(SYSTEM_CONFIG["model_save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = AlphaTransformer(
            input_dim=self.config["input_dim"],
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_encoder_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)
        
        # 定义损失函数与优化器 (以预测收益率排序的 MSE Loss 为例)
        self.criterion = nn.MSELoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _get_dummy_dataloader(self):
        """此处模拟数据加载器 (实际使用中请替换为您封装好的 Dataset/DataLoader)"""
        # 假设：批量大小 = batch_size，时间步 = 20，特征数 = input_dim
        x_dummy = torch.randn(self.config["batch_size"] * 10, 20, self.config["input_dim"])
        y_dummy = torch.randn(self.config["batch_size"] * 10) # 模拟未来收益率打分
        
        dataset = torch.utils.data.TensorDataset(x_dummy, y_dummy)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)

    def train(self):
        self.logger.info(f"🧠 启动 Transformer 训练引擎... 使用设备: {self.device}")
        self.logger.info(f"⚙️ 模型超参: d_model={self.config['d_model']}, nhead={self.config['nhead']}, lr={self.config['learning_rate']}")
        
        train_loader = self._get_dummy_dataloader()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Epoch [{epoch+1}/{self.config['epochs']}] | Loss: {avg_loss:.6f}")
            
            # 早停与模型保存机制
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                save_path = self.save_dir / self.config["model_save_name"]
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f"✨ 发现更低 Loss，已保存最优模型至 {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    self.logger.warning(f"🛑 连续 {self.config['patience']} 轮未下降，触发早停机制！")
                    break
                    
        self.logger.info("🎉 训练流程全部结束。")