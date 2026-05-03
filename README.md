🌟 Quant_MultiAgent_System_Test 使用指南
欢迎使用 Quant_MultiAgent_System_Test！本项目是一个结合了深度学习（Transformer）、多智能体（Multi-Agent：宏观感知、深度排序、风控执行）和“自上而下”逻辑的高级量化交易与回测框架。

经过标准化重构，本系统已实现配置中心化、数据本地化、逻辑模块化，极其适合进行二次开发和实盘模拟。

📁 目录结构说明
Plaintext
Quant_MultiAgent_System_Test/
├── configs/
│   └── config.py                 # ⚙️ 全局统一配置中心（修改参数只在这里进行）
├── data/                         # 🗄️ 本地数据仓库
│   ├── raw_txt/                  # 存放通达信导出的板块 TXT 文件
│   ├── metadata/                 # 生成的股票标签映射表 (stock_info.csv)
│   └── code=*/                   # Parquet 格式的本地 K 线数据 (包含 pre_close)
├── src/
│   ├── data_pipeline/            # 🔄 数据同步与清洗模块
│   ├── analysis/                 # 📊 市场复盘与因子分析模块
│   ├── agents/                   # 🤖 多智能体模块 (宏观、排序、风控)
│   ├── engine/                   # ⚙️ 多智能体回测引擎
│   └── models/                   # 🧠 Transformer 深度学习模型与训练器
└── scripts/                      # 🚀 快捷执行入口（用户主要交互区）
🛠️ 环境准备与安装
Python 环境: 建议使用 Python 3.8+

依赖安装: 请确保安装了以下核心依赖包：

Bash
pip install pandas pyarrow tqdm mootdx torch
数据源准备:

本系统依赖通达信本地盘后数据。请确保本地已安装通达信，并下载了完整的日线数据。

在通达信软件中，导出相关板块数据（如行业板块.txt、概念板块.txt等）至项目的 data/raw_txt/ 目录下。

🚀 核心使用流程（Step-by-Step）
系统所有的核心操作都通过 scripts/ 目录下的快捷脚本完成。所有的参数修改都在 configs/config.py 中进行。

第一步：系统参数配置
在运行任何脚本前，请打开 configs/config.py：

配置通达信路径（tdx_dir）。

配置回测策略（如初始资金 initial_capital，最大持仓 max_holdings，止损止盈线）。

配置模型与智能体参数。

第二步：初始化数据底座
第一次使用系统时，需要构建股票元数据并拉取全量历史行情。

生成股票标签映射表:

Bash
python scripts/run_build_metadata.py
拉取全量历史数据（含 pre_close 前收盘价计算，该过程根据网络和硬盘速度可能需要几分钟）：

Bash
python scripts/run_data_sync.py
第三步：训练深度学习模型 (Transformer)
如果您开启了 Transformer 排序智能体，需要先训练出有效的 Alpha 打分模型。

执行以下脚本，系统将根据 config.py 中的 MODEL_CONFIG 读取特征并开始炼丹：

Bash
python scripts/run_training.py
训练好的最优模型会自动保存在 src/models/saved_models/ 目录下。

第四步：启动多智能体回测 (Backtest)
数据和模型就绪后，即可启动核心回测引擎。引擎严格遵循：盘前自上而下宏观选板块 -> 板块内 Transformer 深度排序 -> 盘中风控与交易执行 -> 盘后生成明日观察池 的逻辑闭环。

执行回测：

Bash
python scripts/run_backtest.py
回测引擎已彻底修复算力浪费问题，运行速度极快。控制台会打印每日的交易明细、资金变动以及触发的风控信号。

⏰ 日常实盘模拟工作流
如果您使用本系统进行每日的盘后选股与实盘模拟，日常工作流非常简单：

每天下午 15:30 收盘后：

开启通达信，下载当天的日线数据。

运行增量更新脚本（系统会自动比对文件修改时间，仅更新今天的新数据）：

Bash
python scripts/run_daily_update.py
（可选）如果想查看今天的市场话事人复盘报告：

Bash
# 直接运行 market_review.py 或为其编写一个 run_review.py 快捷脚本
python src/analysis/market_review.py 
运行回测/选股引擎，查看系统基于今日数据生成的“明日观察池”。

🧩 进阶开发建议
新增因子：在 src/analysis/ 中添加因子计算逻辑，并将结果拼接入 data/ 中的 parquet 文件。

修改网络结构：直接修改 src/models/network_transformer.py。

自定义风控：在 src/agents/agent_risk.py 中增加新的熔断、大盘避险或个股动态止盈逻辑，并在 config.py 中添加相应阈值参数。
