# src/data_pipeline/build_metadata.py
import pandas as pd
from pathlib import Path

# 引入统一配置
from configs.config import SYSTEM_CONFIG

# 动态获取路径
TXT_DIR = Path(SYSTEM_CONFIG["raw_txt_dir"])
OUT_DIR = Path(SYSTEM_CONFIG["metadata_dir"])

def parse_block_file(file_path):
    """解析通达信板块导出格式：板块代码 板块名称 股票代码 股票名称"""
    data = []
    with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                block_name = parts[1]
                stock_code = parts[2]
                stock_name = "".join(parts[3:])
                
                # 标准化 A 股代码前缀
                full_code = ""
                if stock_code.startswith('6'): full_code = f"sh{stock_code}"
                elif stock_code.startswith(('0', '3')): full_code = f"sz{stock_code}"
                elif stock_code.startswith(('4', '8', '9')): full_code = f"bj{stock_code}"
                else: full_code = stock_code
                
                data.append({
                    'full_code': full_code,
                    'code': stock_code,
                    'name': stock_name,
                    'block_name': block_name
                })
    return pd.DataFrame(data)

def main_build_metadata():
    """构建多维映射表的主函数"""
    print("🚀 正在启动多维映射表构建引擎...")
    
    # 确保目录存在
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    category_map = {
        '行业板块.txt': 'industry',
        '地区板块.txt': 'region',
        '风格板块.txt': 'style',
        '概念板块.txt': 'concepts',
        '指数板块.txt': 'indexes'
    }
    
    dfs = {}
    all_stocks = pd.DataFrame(columns=['full_code', 'code', 'name'])
    
    for filename, col_name in category_map.items():
        file_path = TXT_DIR / filename
        if not file_path.exists():
            print(f"⚠️ 跳过缺失文件: {filename} (请确认已放入 {TXT_DIR})")
            continue
            
        print(f"📦 正在解析: {filename} -> {col_name}")
        df = parse_block_file(file_path)
        
        base_info = df[['full_code', 'code', 'name']].drop_duplicates('full_code')
        all_stocks = pd.concat([all_stocks, base_info]).drop_duplicates('full_code', keep='last')
        
        grouped = df.groupby('full_code')['block_name'].apply(lambda x: ','.join(x.unique())).reset_index()
        grouped.columns = ['full_code', col_name]
        dfs[col_name] = grouped

    if all_stocks.empty:
        print("❌ 未解析到任何股票数据，请检查 txt 文件格式。")
        return

    final_df = all_stocks
    for col_name in dfs.keys():
        final_df = pd.merge(final_df, dfs[col_name], on='full_code', how='left')
    
    final_df = final_df.fillna('无')
    
    save_path = OUT_DIR / "stock_info.csv"
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✨ 映射表构建成功！")
    print(f"📍 存储位置: {save_path}")
    print(f"📊 总计记录: {len(final_df)} 只股票")

if __name__ == "__main__":
    main_build_metadata()