<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>

# FinRL: Financial Reinforcement Learning → FinRL-X

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="55%"/>
</div>

[![Downloads](https://static.pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://static.pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Join Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/trsr8SXpW5)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)
![](https://img.shields.io/github/issues-raw/AI4Finance-Foundation/finrl?label=Issues)
![](https://img.shields.io/github/issues-closed-raw/AI4Finance-Foundation/finrl?label=Closed+Issues)
![](https://img.shields.io/github/issues-pr-raw/AI4Finance-Foundation/finrl?label=Open+PRs)
![](https://img.shields.io/github/issues-pr-closed-raw/AI4Finance-Foundation/finrl?label=Closed+PRs)
[![X](https://img.shields.io/badge/X-Share-black?logo=x)](https://twitter.com/intent/tweet?text=FinRL-Financial-Deep-Reinforcement-Learning%20&url=https://github.com/AI4Finance-Foundation/FinRL&hashtags=DRL,AI) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL)


> [!IMPORTANT]
> **FinRL-X** is the next-generation evolution of FinRL, designed for AI-native, modular, and production-oriented quantitative trading.
>
> - **This repository (`FinRL`)** preserves the original end-to-end educational and research framework.
> - **For the latest architecture, live trading deployment, and production-focused development, please use [`FinRL-X / FinRL-Trading`](https://github.com/AI4Finance-Foundation/FinRL-Trading).**

**FinRL®** is widely recognized as the first open-source framework for financial reinforcement learning.
This repository contains the original FinRL library for education, benchmarking, and research prototyping.

For the next-generation AI-native and production-oriented trading stack, please visit **[FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)**.

## FinRL Ecosystem Roadmap

| Generation | Positioning | Target Users | Repository | Description |
|----|----|----|----|----|
| FinRL-Meta | Market Environments | Practitioners | [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta) | Gym-style financial market environments and benchmarks |
| FinRL | Classic End-to-End Framework | Learners, Developers, Researchers | [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Original train-test-trade pipeline for financial reinforcement learning |
| ElegantRL | Algorithm Layer | Researchers and Experts | [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | Lightweight and elegant DRL algorithms |
| **FinRL-X** | **Next Generation / Production** | **Professional traders, institutions, hedge funds** | [**FinRL-Trading**](https://github.com/AI4Finance-Foundation/FinRL-Trading) | **AI-native modular infrastructure for deployment-aware quantitative trading** |
> **Recommended for new users:** Start with **[FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)** if you are building modern or production-oriented trading systems.

### 🔄 FinRL-X vs. FinRL: What Changed

| Capability | FinRL (Stage 1.0) | FinRL-X (Stage 3.0) |
|---|---|---|
| **Paradigm** | Deep Reinforcement Learning | AI-Native (ML + DRL + LLM-ready) |
| **Architecture** | Three-layer coupled monolith | Fully decoupled modular layers |
| **Strategies** | DRL agents (A2C, DDPG, PPO, SAC, TD3) | ML selection + DRL timing + extensible base |
| **Data Layer** | 14 manually-wired processors | Auto-select: Yahoo Finance → FMP → WRDS |
| **Backtesting** | Custom hand-rolled evaluation loops | Professional `bt` library engine |
| **Live Trading** | Basic Alpaca support | Full multi-account integration + risk controls |
| **Configuration** | `config.py` + `config_tickers.py` | Type-safe Pydantic + `.env` multi-env |
| **Risk Management** | Gym environment constraints only | Order · portfolio · strategy-level controls |
| **Target Users** | Researchers & students | Quants, institutions, production deployments |
| **Paper** | [arXiv:2011.09607](https://arxiv.org/abs/2011.09607) | [arXiv:2603.21330](https://arxiv.org/abs/2603.21330) |

[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT): an open-source project for financial large language models, designed for research and real-world FinTech applications.

![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL&countColor=%23B17A)
[![Discord](https://dcbadge.limes.pink/api/server/trsr8SXpW5?v=20260320)](https://discord.gg/trsr8SXpW5)

## Outline

  - [Overview](#overview)
  - [File Structure](#file-structure)
  - [Supported Data Sources](#supported-data-sources)
  - [Installation](#installation)
  - [Status Update](#status-update)
  - [Tutorials](#tutorials)
  - [Publications](#publications)
  - [News](#news)
  - [Citing FinRL](#citing-finrl)
  - [Join and Contribute](#join-and-contribute)
    - [Contributors](#contributors)
    - [Sponsorship](#sponsorship)
  - [LICENSE](#license)

## Overview

FinRL is the original open-source framework for financial reinforcement learning, organized around three core layers:

- **Market Environments**
- **DRL Agents**
- **Financial Applications**

For a trading task, an agent interacts with a market environment and learns sequential decision-making policies.

This repository focuses on the **classic FinRL workflow** for education, experimentation, and research prototyping.

For the **next-generation production-oriented stack**, including modular deployment and AI-native trading infrastructure, please visit **[FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)**.

<div align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/FinRL_X_Framework.png" width="880"/>
</div>

<div align="center">
<img align="center" src=figs/finrl_framework.png>
</div>

Videos [FinRL](http://www.youtube.com/watch?v=ZSGJjtM-5jA) at [AI4Finance Youtube Channel](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g).

## FinRL Stock Trading 2026 Tutorial
This tutorial demonstrates the original FinRL workflow for educational and research purposes.
For the latest production-oriented pipeline, please use **[FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)**.
### Step 1: Clone the Repository

```bash
git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install FinRL

```bash
pip install -e .
```

### Step 4: Run the Scripts

**1. Data Download & Preprocessing**

```bash
python examples/FinRL_StockTrading_2026_1_data.py
```

This script downloads DOW 30 stock data from Yahoo Finance, adds technical indicators (MACD, RSI, etc.), VIX, and turbulence index, then splits the data into training set (2014–2025) and trading set (2026-01-01 to 2026-03-20), saving them as `train_data.csv` and `trade_data.csv`.

**2. Train DRL Agents**

```bash
python examples/FinRL_StockTrading_2026_2_train.py
```

This script trains 5 DRL agents (A2C, DDPG, PPO, TD3, SAC) using Stable Baselines 3 on the training data. Trained models are saved to the `trained_models/` directory.

**3. Backtest**

```bash
python examples/FinRL_StockTrading_2026_3_Backtest.py
```

This script loads the trained agents, runs them on the trading data, and compares their performance against two baselines: Mean Variance Optimization (MVO) and the DJIA index. Results are printed to the console and a plot is saved as `backtest_result.png`.


## File Structure

The main folder **finrl** has three subfolders **applications, agents, meta**. We employ a **train-test-trade** pipeline with three files: train.py, test.py, and trade.py.

```
FinRL
├── finrl (main folder)
│   ├── applications
│   	├── Stock_NeurIPS2018
│   	├── imitation_learning
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	└── stock_trading
│   ├── agents
│   	├── elegantrl
│   	├── rllib
│   	└── stablebaseline3
│   ├── meta
│   	├── data_processors
│   	├── env_cryptocurrency_trading
│   	├── env_portfolio_allocation
│   	├── env_stock_trading
│   	├── preprocessor
│   	├── data_processor.py
│       ├── meta_config_tickers.py
│   	└── meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
│
├── examples
├── unit_tests (unit tests to verify codes on env & data)
│   ├── environments
│   	└── test_env_cashpenalty.py
│   └── downloaders
│   	├── test_yahoodownload.py
│   	└── test_alpaca_downloader.py
├── setup.py
├── requirements.txt
└── README.md
```

## Supported Data Sources

|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Akshare](https://alpaca.markets/docs/introduction/)| CN Securities| 2015-now, 1day| Account-specific| OHLCV| Prices&Indicators|
|[Alpaca](https://docs.alpaca.markets/docs/getting-started)| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|[Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)| CN Securities| 1990-12-19-now, 5min| Account-specific| OHLCV| Prices&Indicators|
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggregated trades, OHLCV| Prices&Indicators|
|[CCXT](https://docs.ccxt.com/en/latest/manual.html)| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|[EODhistoricaldata](https://eodhistoricaldata.com/financial-apis/)| US Securities| Frequency-specific, 1min| API-specific | OHLCV | Prices&Indicators|
|[IEXCloud](https://iexcloud.io/docs/api/)| NMS US securities|1970-now, 1 day|100 per second per IP|OHLCV| Prices&Indicators|
|[JoinQuant](https://www.joinquant.com/)| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|[QuantConnect](https://www.quantconnect.com/docs/v2)| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|
|[RiceQuant](https://www.ricequant.com/doc/rqdata/python/)| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
[Sinopac](https://sinotrade.github.io/zh_TW/tutor/prepare/terms/) | Taiwan securities | 2023-04-13~now, 1min | Account-specific | OHLCV | Prices&Indicators|
|[Tushare](https://tushare.pro/document/1?doc_id=131)| CN Securities, A-share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[YahooFinance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|


<!-- |Data Source |Type |Max Frequency |Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |
|    AkShare |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Alpaca |  US Stocks, ETFs |  1 min |  OHLCV |  Prices, indicators |
|    Alpha Vantage | Stock, ETF, forex, crypto, technical indicators | 1 min |  OHLCV  & Prices, indicators |
|    Baostock |  CN Securities |  5 min |  OHLCV |  Prices, indicators |
|    Binance |  Cryptocurrency |  1 s |  OHLCV |  Prices, indicators |
|    CCXT |  Cryptocurrency |  1 min  |  OHLCV |  Prices, indicators |
|    currencyapi |  Exchange rate | 1 day |  Exchange rate | Exchange rate, indicators |
|    currencylayer |  Exchange rate | 1 day  |  Exchange rate | Exchange rate, indicators |
|    EOD Historical Data | US stocks, and ETFs |  1 day  |  OHLCV  | Prices, indicators |
|    Exchangerates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    findatapy |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Financial Modeling prep | US stocks, currencies, crypto |  1 min |  OHLCV  | Prices, indicators |
|    finnhub | US Stocks, currencies, crypto |   1 day |  OHLCV  | Prices, indicators |
|    Fixer |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    IEXCloud |  NMS US securities | 1 day  | OHLCV |  Prices, indicators |
|    JoinQuant |  CN Securities |  1 min  |  OHLCV |  Prices, indicators |
|    Marketstack | 50+ countries |  1 day  |  OHLCV | Prices, indicators |
|    Open Exchange Rates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    pandas\_datareader |  US Securities |  1 day |  OHLCV | Prices, indicators |
|    pandas-finance |  US Securities |  1 day  |  OHLCV  & Prices, indicators |
|    Polygon |  US Securities |  1 day  |  OHLCV  | Prices, indicators |
|    Quandl | 250+ sources |  1 day  |  OHLCV  | Prices, indicators |
|    QuantConnect |  US Securities |  1 s |  OHLCV |  Prices, indicators |
|    RiceQuant |  CN Securities |  1 ms  |  OHLCV |  Prices, indicators |
|    Sinopac   | Taiwan securities | 1min | OHLCV |  Prices, indicators |
|    Tiingo | Stocks, crypto |  1 day  |  OHLCV  | Prices, indicators |
|    Tushare |  CN Securities | 1 min  |  OHLCV |  Prices, indicators |
|    WRDS |  US Securities |  1 ms  |  Intraday Trades | Prices, indicators |
|    XE |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    Xignite |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    YahooFinance |  US Securities | 1 min  |  OHLCV  |  Prices, indicators |
|    ystockquote |  US Securities |  1 day  |  OHLCV | Prices, indicators | -->



OHLCV: open, high, low, and close prices; volume. adjusted_close: adjusted close price

Technical indicators: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'. Users also can add new features.


## Installation
+ [Install description for all operating systems (MAC OS, Ubuntu, Windows 10)](./docs/source/start/installation.rst)
+ [FinRL for Quantitative Finance: Install and Setup Tutorial for Beginners](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159)

## Status Update
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 2022-06-25
	0.3.5: Formal release of FinRL, neo_finrl is changed to FinRL-Meta with related files in directory: *meta*.
* 2021-08-25
	0.3.1: pytorch version with a three-layer architecture, apps (financial tasks), drl_agents (drl algorithms), neo_finrl (gym env)
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Removed TensorFlow 1.x support; TensorFlow 2.0 support was under development at the time.
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
</div>
</details>


## Tutorials

+ [Towards Data Science] [Deep Reinforcement Learning for Automated Stock Trading](https://medium.com/data-science/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)
