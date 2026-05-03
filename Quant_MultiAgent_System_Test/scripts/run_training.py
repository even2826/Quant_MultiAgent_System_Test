import sys
import os

# 将项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.trainer import ModelTrainer

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()