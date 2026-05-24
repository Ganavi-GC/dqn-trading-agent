import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
import os

st.set_page_config(page_title="DQN Trading Agent", page_icon="📈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #0d1a0d 50%, #0a0a0a 100%); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1a0d 0%, #0a120a 100%); border-right: 1px solid #1a3a1a; }
    .hero-title { font-size: 3rem; font-weight: 700; background: linear-gradient(90deg, #00ff88, #00cc6a, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; margin-bottom: 0.5rem; letter-spacing: -1px; }
    .hero-subtitle { text-align: center; color: #4a7a4a; font-size: 1rem; font-weight: 400; margin-bottom: 2rem; letter-spacing: 2px; text-transform: uppercase; }
    .metric-card { background: linear-gradient(135deg, #0d1a0d, #081208); border: 1px solid #1a4a1a; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; box-shadow: 0 0 30px rgba(0, 255, 136, 0.08); }
    .metric-label { color: #4a7a4a; font-size: 0.75rem; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem; }
    .metric-value { color: #00ff88; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
    .metric-delta-pos { color: #00ff88; font-size: 0.85rem; font-weight: 500; margin-top: 0.3rem; }
    .metric-delta-neg { color: #ff4444; font-size: 0.85rem; font-weight: 500; margin-top: 0.3rem; }
    .metric-neutral { color: #4a7a4a; font-size: 0.85rem; font-weight: 500; margin-top: 0.3rem; }
    .section-header { color: #00ff88; font-size: 1rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1a4a1a; }
    .status-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; }
    .badge-success { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid rgba(0, 255, 136, 0.3); }
    .badge-info { background: rgba(0, 150, 255, 0.1); color: #0096ff; border: 1px solid rgba(0, 150, 255, 0.3); }
    .info-box { background: rgba(0, 255, 136, 0.03); border: 1px solid rgba(0, 255, 136, 0.15); border-radius: 12px; padding: 1.5rem; text-align: center; color: #4a7a4a; font-size: 0.9rem; }
    div[data-testid="stButton"] button { background: linear-gradient(135deg, #00ff88, #00cc6a) !important; color: #000 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; letter-spacing: 1px !important; font-size: 0.85rem !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_capital):
        super().__init__()
        self.df = df
        self.initial_capital = initial_capital
        self.n_steps = len(df)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None):
        self.current_step = 0
        self.cash = self.initial_capital
        self.holdings = 0
        self.portfolio_value = self.cash
        self.portfolio_history = [self.cash]
        self.actions_taken = []
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        def val(x): return float(x.iloc[0]) if hasattr(x, 'iloc') else float(x)
        return np.array([
            val(row["Close"]) / 10000,
            val(row["MA10"])  / 10000,
            val(row["MA50"])  / 10000,
            val(row["Returns"]),
            val(row["Volatility"]),
            self.holdings
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        def val(x): return float(x.iloc[0]) if hasattr(x, 'iloc') else float(x)
        price = val(row["Close"])
        if action == 1 and self.cash >= price:
            self.holdings += 1
            self.cash -= price
        elif action == 2 and self.holdings > 0:
            self.holdings -= 1
            self.cash += price
        self.portfolio_value = self.cash + self.holdings * price
        self.portfolio_history.append(self.portfolio_value)
        self.actions_taken.append(action)
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        reward = (self.portfolio_value - self.initial_capital) / self.initial_capital
        return self._get_obs(), reward, done, False, {}

def get_data(start, end):
    df = yf.download("^NSEI", start=str(start), end=str(end), auto_adjust=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df["MA10"]       = df["Close"].rolling(10).mean()
    df["MA50"]       = df["Close"].rolling(50).mean()
    df["Returns"]    = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(10).std()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def make_chart(portfolio_history, initial_capital):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0d1a0d')
    ax.plot(portfolio_history, color='#00ff88', linewidth=1.5, label='Portfolio Value')
    ax.axhline(y=initial_capital, color='#ff4444', linestyle='--', linewidth=1, label='Initial Capital')
    ax.fill_between(range(len(portfolio_history)), initial_capital, portfolio_history,
                    where=[v > initial_capital for v in portfolio_history], alpha=0.1, color='#00ff88')
    ax.set_xlabel("Trading Days", color='#4a7a4a')
    ax.set_ylabel("Portfolio Value (₹)", color='#4a7a4a')
    ax.tick_params(colors='#4a7a4a')
    ax.spines['bottom'].set_color('#1a4a1a')
    ax.spines['left'].set_color('#1a4a1a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#0d1a0d', edgecolor='#1a4a1a', labelcolor='#4a7a4a')
    ax.grid(True, alpha=0.1, color='#1a4a1a')
    fig.tight_layout()
    return fig

def make_action_chart(df, actions):
    prices = df["Close"].values.flatten()[:len(actions)]
    buy_idx  = [i for i, a in enumerate(actions) if a == 1]
    sell_idx = [i for i, a in enumerate(actions) if a == 2]
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0d1a0d')
    ax.plot(prices, color='#0096ff', linewidth=1, alpha=0.8, label='NIFTY 50')
    ax.scatter(buy_idx,  [prices[i] for i in buy_idx],  color='#00ff88', marker='^', s=60, label='Buy',  zorder=5)
    ax.scatter(sell_idx, [prices[i] for i in sell_idx], color='#ff4444', marker='v', s=60, label='Sell', zorder=5)
    ax.set_xlabel("Trading Days", color='#4a7a4a')
    ax.set_ylabel("Price (₹)", color='#4a7a4a')
    ax.tick_params(colors='#4a7a4a')
    ax.spines['bottom'].set_color('#1a4a1a')
    ax.spines['left'].set_color('#1a4a1a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#0d1a0d', edgecolor='#1a4a1a', labelcolor='#4a7a4a')
    ax.grid(True, alpha=0.1, color='#1a4a1a')
    fig.tight_layout()
    return fig

with st.sidebar:
    st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)
    start_date      = st.date_input("Start Date",      value=pd.to_datetime("2023-01-01"))
    end_date        = st.date_input("End Date",        value=pd.to_datetime("2024-12-31"))
    timesteps       = st.slider("Training Timesteps", 5000, 30000, 15000, step=5000)
    initial_capital = st.number_input("Initial Capital (₹)", value=100000, step=10000)
    st.markdown('<p class="section-header">Model</p>', unsafe_allow_html=True)
    model_exists = os.path.exists("dqn_trading_model.zip")
    if model_exists:
        st.markdown('<span class="status-badge badge-success">✓ TRAINED MODEL READY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-info">○ NO MODEL YET</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run_button   = st.button("🚀 Train & Run Agent" if not model_exists else "▶ Run Agent")
    train_button = st.button("🔄 Retrain Model") if model_exists else False

st.markdown('<h1 class="hero-title">DQN Stock Trading Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Deep Reinforcement Learning · NIFTY 50 · Neural Network</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('''<div class="metric-card"><div class="metric-label">Algorithm</div><div class="metric-value" style="font-size:1.2rem">DQN</div><div class="metric-neutral">Deep Q-Network</div></div>''', unsafe_allow_html=True)
with col2:
    st.markdown('''<div class="metric-card"><div class="metric-label">Actions</div><div class="metric-value" style="font-size:1.2rem">3</div><div class="metric-neutral">Buy · Sell · Hold</div></div>''', unsafe_allow_html=True)
with col3:
    st.markdown('''<div class="metric-card"><div class="metric-label">State Space</div><div class="metric-value" style="font-size:1.2rem">6D</div><div class="metric-neutral">Price · MA · Vol</div></div>''', unsafe_allow_html=True)
with col4:
    st.markdown('''<div class="metric-card"><div class="metric-label">Framework</div><div class="metric-value" style="font-size:1.2rem">PyTorch</div><div class="metric-neutral">via SB3</div></div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

should_train = (run_button and not model_exists) or train_button
should_run   = run_button or train_button

if should_run:
    with st.spinner("Fetching NIFTY 50 market data..."):
        df = get_data(start_date, end_date)
    st.markdown(f'<span class="status-badge badge-success">✓ {len(df)} trading days loaded</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    env = StockTradingEnv(df, initial_capital)

    if should_train or not model_exists:
        st.markdown('<p class="section-header">Training</p>', unsafe_allow_html=True)
        progress = st.progress(0)
        status   = st.empty()
        for i in range(5):
            pct = (i + 1) * 20
            status.markdown(f'<p style="color:#4a7a4a;font-size:0.85rem;">Training neural network... {pct}%</p>', unsafe_allow_html=True)
            progress.progress(pct)
            if i == 0:
                model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-3, buffer_size=10000, batch_size=64)
                model.learn(total_timesteps=timesteps)
                model.save("dqn_trading_model")
        progress.progress(100)
        status.markdown('<span class="status-badge badge-success">✓ Training complete — model saved</span>', unsafe_allow_html=True)
    else:
        with st.spinner("Loading saved model..."):
            model = DQN.load("dqn_trading_model", env=env)
        st.markdown('<span class="status-badge badge-success">✓ Model loaded from disk</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))

    final_value = env.portfolio_value
    profit      = final_value - initial_capital
    returns     = (profit / initial_capital) * 100
    buy_count   = env.actions_taken.count(1)
    sell_count  = env.actions_taken.count(2)
    hold_count  = env.actions_taken.count(0)

    st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'''<div class="metric-card"><div class="metric-label">Initial Capital</div><div class="metric-value">₹{initial_capital:,.0f}</div></div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="metric-card"><div class="metric-label">Final Portfolio</div><div class="metric-value">₹{final_value:,.0f}</div></div>''', unsafe_allow_html=True)
    with c3:
        color = "metric-delta-pos" if profit > 0 else "metric-delta-neg"
        sign  = "+" if profit > 0 else ""
        st.markdown(f'''<div class="metric-card"><div class="metric-label">Profit / Loss</div><div class="metric-value">₹{abs(profit):,.0f}</div><div class="{color}">{sign}{returns:.1f}%</div></div>''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''<div class="metric-card"><div class="metric-label">Trading Days</div><div class="metric-value">{len(df)}</div></div>''', unsafe_allow_html=True)
    with c5:
        st.markdown(f'''<div class="metric-card"><div class="metric-label">Actions</div><div class="metric-value" style="font-size:1rem"><span style="color:#00ff88">▲{buy_count}</span> <span style="color:#ff4444">▼{sell_count}</span> <span style="color:#4a7a4a">—{hold_count}</span></div></div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Portfolio Performance</p>', unsafe_allow_html=True)
    st.pyplot(make_chart(env.portfolio_history, initial_capital))
    st.markdown('<p class="section-header">Agent Actions on NIFTY 50</p>', unsafe_allow_html=True)
    st.pyplot(make_action_chart(df, env.actions_taken))
    st.balloons()

else:
    st.markdown('''<div class="info-box">⚙️ Configure your parameters in the sidebar and click <strong style="color:#00ff88">Run Agent</strong> to start</div>''', unsafe_allow_html=True)