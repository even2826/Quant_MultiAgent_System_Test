# scripts/run_daily_update.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.incremental_sync import main_incremental_sync
# 这里如果您以后加了类似 "更新元数据" "更新复盘分析" 的模块，可以一起在这里调用

if __name__ == "__main__":
    print("====== 开始执行每日盘后数据作业 ======")
    main_incremental_sync()
    print("====== 每日盘后作业全部完成 ======")