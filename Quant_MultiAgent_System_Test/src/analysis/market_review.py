# src/analysis/market_review.py
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
import numpy as np

# 引入统一配置
from configs.config import SYSTEM_CONFIG

# 从配置中动态获取数据仓库路径
DATA_DIR = Path(SYSTEM_CONFIG["data_dir"])
META_FILE = DATA_DIR / 'metadata' / 'stock_info.csv'

def load_data_engine(target_date=None):
    """加载全市场某日行情（含 pre_close 列）"""
    file_list = list(DATA_DIR.glob('code=*/data.zstd.parquet'))
    if not file_list:
        return None
    dataset = ds.dataset(file_list, format="parquet", partitioning="hive")
    if target_date:
        table = dataset.to_table(filter=ds.field("date") == pd.Timestamp(target_date))
        df = table.to_pandas()
        df['code'] = df['code'].astype(str)
        return df
    return dataset

def get_ladder_height(code, target_date, depth=10):
    """使用 pre_close 计算真实涨停高度（支持10cm/20cm）"""
    try:
        p_file = DATA_DIR / f"code={code}" / "data.zstd.parquet"
        if not p_file.exists():
            return 0
        df = pd.read_parquet(p_file)
        df = df[df['date'] <= pd.Timestamp(target_date)].tail(depth + 1)
        if len(df) < 2:
            return 0

        if 'pre_close' in df.columns:
            df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close'] * 100
        else:
            df['pct_chg'] = df['close'].pct_change() * 100

        limit = 19.8 if code.startswith(('sz30', 'sh68')) else 9.8
        df['is_zt'] = df['pct_chg'] >= limit

        zt_series = df['is_zt'].values[::-1]
        height = 0
        for zt in zt_series:
            if zt:
                height += 1
            else:
                break
        return height
    except Exception as e:
        return 0

def generate_report(date_str):
    """生成市场话事人复盘报告"""
    print(f"🔍 正在深度解析 {date_str} 市场话事人...")

    ref_file = DATA_DIR / 'code=sh000001' / 'data.zstd.parquet'
    if not ref_file.exists():
        return "⚠️ 未找到基准指数数据，请先执行数据同步。"
        
    ref_df = pd.read_parquet(ref_file, columns=['date'])
    calendar = sorted(ref_df['date'].unique())
    target_dt = pd.Timestamp(date_str)

    if target_dt not in calendar:
        return f"⚠️ {date_str} 非交易日。"

    df = load_data_engine(target_dt)
    if df is None or df.empty:
        return f"⚠️ {date_str} 无行情数据。"

    # 关联元数据 (若存在)
    if META_FILE.exists():
        info = pd.read_csv(META_FILE, dtype={'code': str, 'full_code': str})
        df = pd.merge(df, info, left_on='code', right_on='full_code', how='left', suffixes=('', '_meta'))

    # 核心指标计算
    if 'pre_close' not in df.columns:
        prev_dt = calendar[calendar.index(target_dt) - 1]
        df_yesterday = load_data_engine(prev_dt)
        if df_yesterday is not None and 'close' in df_yesterday.columns:
            y_close = df_yesterday[['code', 'close']].rename(columns={'close': 'pre_close'})
            df = pd.merge(df, y_close, on='code', how='left')
        else:
            return "❌ 缺少 pre_close 数据，无法计算涨跌幅。"

    df['pct_chg'] = (df['close'] / df['pre_close'] - 1) * 100
    df['is_zt'] = np.where(
        df['code'].str.startswith(('sz30', 'sh68')),
        df['pct_chg'] >= 19.8,
        df['pct_chg'] >= 9.8
    )
    df['turnover_ratio'] = df['amount'] / (df['close'] * 1e6)

    # ... (维度分析逻辑不变，省略部分相同代码以保持简洁)
    # 国家队
    gjd_pool = df[df['indexes'].str.contains('上证50|沪深300', na=False)] if 'indexes' in df.columns else pd.DataFrame()
    gjd_strength = (gjd_pool['pct_chg'] > df['pct_chg'].mean()).mean() if not gjd_pool.empty else 0

    # 游资（连板高度）
    zt_pool = df[df['is_zt']]
    max_ladder = 0
    if not zt_pool.empty:
        top_zt_codes = zt_pool.sort_values('amount', ascending=False)['code'].head(15).tolist()
        heights = [get_ladder_height(c, date_str) for c in top_zt_codes]
        max_ladder = max(heights) if heights else 0

    # 话事人判定
    dominant = "国家队" if gjd_strength > 0.65 else ("游资" if max_ladder >= 3 else "量化/机构")

    report = f"""
【{date_str} 市场复盘报告】
-------------------------------------------
当日话事人：{dominant}
-------------------------------------------
1. 情绪维度（游资）：
   - 涨停 {len(zt_pool)} 家，最高连板 {max_ladder} 板。
2. 结构维度（机构/国家队）：
   - 国家队护盘强度：{(gjd_strength*100):.1f}%
-------------------------------------------
    """
    return report