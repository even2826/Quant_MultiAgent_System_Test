# src/data_pipeline/incremental_sync.py
import time
import pandas as pd
from mootdx.reader import Reader
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 引入统一配置
from configs.config import SYSTEM_CONFIG

TDX_DIR = SYSTEM_CONFIG["tdx_dir"]
OUT_DIR = Path(SYSTEM_CONFIG["data_dir"])
MARKETS = SYSTEM_CONFIG["sync_markets"]
THREADS = SYSTEM_CONFIG["sync_threads"]

reader = Reader.factory(market='std', tdxdir=TDX_DIR)

def get_task_list():
    """对比文件修改时间，筛选出需要更新的任务"""
    tasks = []
    print("🔍 正在扫描文件修改状态以确定增量范围...")
    
    for mkt in MARKETS:
        src_path = Path(TDX_DIR) / 'vipdoc' / mkt / 'lday'
        if not src_path.exists():
            continue
            
        files = list(src_path.glob("*.day"))
        for f in files:
            full_code = f.stem
            target_file = OUT_DIR / f"code={full_code}" / "data.zstd.parquet"
            
            # 增量判定：目标不存在，或通达信源文件比我们的 Parquet 更新
            if not target_file.exists() or f.stat().st_mtime > target_file.stat().st_mtime:
                tasks.append((mkt, f))
                
    return tasks

def sync_incremental_stock(mkt, file_path):
    """单个股票的增量转换与 pre_close 计算"""
    try:
        full_code = file_path.stem
        df = reader.daily(symbol=full_code)
        if df is None or df.empty:
            return False
        
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' not in df.columns:
            return False
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        if 'close' in df.columns:
            df['pre_close'] = df['close'].shift(1)
        else:
            return False
        
        target_dir = OUT_DIR / f"code={full_code}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            target_dir / "data.zstd.parquet",
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        return True
    except Exception:
        return False

def main_incremental_sync():
    """暴露给外部调用的增量同步主函数"""
    start_t = time.time()
    
    update_tasks = get_task_list()
    total_tasks = len(update_tasks)
    
    if total_tasks == 0:
        print("✨ 恭喜：所有数据已是最新，无需执行增量更新。")
        return

    print(f"📦 监测到 {total_tasks} 个文件有变动，准备启动 {THREADS} 线程增量更新...")

    success = 0
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(sync_incremental_stock, mkt, f) for mkt, f in update_tasks]
        for future in tqdm(futures, desc="增量同步进度"):
            if future.result():
                success += 1
                
    end_t = time.time()
    print(f"\n✅ 增量更新完成！成功: {success} | 跳过: {total_tasks - success} | 耗时: {end_t - start_t:.2f}s")

if __name__ == "__main__":
    main_incremental_sync()