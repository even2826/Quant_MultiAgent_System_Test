# src/data_pipeline/sync_data.py
import time
import pandas as pd
from mootdx.reader import Reader
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 引入统一配置
from configs.config import SYSTEM_CONFIG

# 从配置字典中读取全局变量
TDX_DIR = SYSTEM_CONFIG["tdx_dir"]
OUT_DIR = Path(SYSTEM_CONFIG["data_dir"])
MARKETS = SYSTEM_CONFIG["sync_markets"]
THREADS = SYSTEM_CONFIG["sync_threads"]

# 初始化 mootdx 读取器
reader = Reader.factory(market='std', tdxdir=TDX_DIR)

def sync_single_stock(market, file_path):
    """
    单个股票转换核心逻辑
    """
    try:
        full_code = file_path.stem  # 获取文件名如 sh600519
        
        # 使用 mootdx 读取
        df = reader.daily(symbol=full_code)
        if df is None or df.empty:
            return False
            
        # 数据清洗
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # 建立分区目录 (Hive-style)
        target_dir = OUT_DIR / f"code={full_code}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 写入 Parquet
        df.to_parquet(
            target_dir / "data.zstd.parquet",
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        return True
    except Exception:
        return False

def main_sync():
    """暴露给外部脚本调用的主函数"""
    start_time = time.time()
    print(f"🚀 启动全量行情同步 (线程数: {THREADS})...")
    
    all_files = []
    for mkt in MARKETS:
        src_path = Path(TDX_DIR) / 'vipdoc' / mkt / 'lday'
        if src_path.exists():
            files = list(src_path.glob("*.day"))
            # 给每个文件贴上市场标签
            all_files.extend([(mkt, f) for f in files])
    
    print(f"📦 待处理文件总数: {len(all_files)}")
    
    # 执行多线程并发
    success_count = 0
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(sync_single_stock, mkt, f) for mkt, f in all_files]
        
        for future in tqdm(futures, desc="同步进度"):
            if future.result():
                success_count += 1
                
    end_time = time.time()
    print(f"\n✨ 同步完成！")
    print(f"成功: {success_count} | 失败: {len(all_files) - success_count}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

# 测试时可以直接运行
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    main_sync()