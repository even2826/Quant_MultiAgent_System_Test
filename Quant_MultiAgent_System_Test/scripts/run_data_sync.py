import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SYSTEM_CONFIG
# 假设重构后数据拉取模块为 src.data_pipeline.sync_data
from src.data_pipeline.sync_data import main_sync 

if __name__ == "__main__":
    print(f"开始同步数据到: {SYSTEM_CONFIG['data_dir']}")
    main_sync()