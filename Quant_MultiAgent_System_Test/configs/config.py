import os

# 获取项目根目录，方便拼装绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. 系统与路径配置
SYSTEM_CONFIG = {
    # --- 基础路径 ---
    "data_dir": os.path.join(BASE_DIR, r"D:\\QuantData"),                 
    "tdx_dir": r"D:\htzq",                                      # 通达信本地安装目录
    "model_save_dir": os.path.join(BASE_DIR, "src", "models", "saved_models"),

    # 👇 新增：元数据与原始板块TXT的路径
    "raw_txt_dir": os.path.join(BASE_DIR, r"D:\\QuantData", "raw_txt"),   # 请把导出的板块TXT放这里
    "metadata_dir": os.path.join(BASE_DIR, r"D:\\QuantData", "metadata"), # 生成的 stock_info.csv 将存在这里
    
    # --- 数据同步参数 ---
    "sync_markets": ['sh', 'sz'],                               # 同步市场范围
    "sync_threads": 16,                                         # 并发线程数
    
    "log_level": "INFO",
}

# 2. 策略与回测配置
STRATEGY_CONFIG = {
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 1000000.0,    # 初始资金 100万
    "max_holdings": 3,               # 最大持仓（先锋标的数量）
    "stop_loss_threshold": -0.08,    # 止损线 -8%
    "take_profit_threshold": 0.15,   # 止盈线 15%
    "commission_rate": 0.0003        # 交易手续费率
}

# 3. 深度学习模型配置 (Transformer等)
MODEL_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_encoder_layers": 4,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 50
}

# 4. 多智能体配置 (Multi-Agent)
AGENT_CONFIG = {
    "macro_agent": {
        "check_frequency": "daily",
        "top_n_sectors": 3          # 顶层轮动选择的最强板块数
    },
    "ranking_agent": {
        "use_transformer": True,
        "target_pool_size": 100     # 粗筛后的股票池大小
    }
}