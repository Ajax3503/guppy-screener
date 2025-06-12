# guppy_screener_app.py

import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import datetime

# ——— App Header —————————————————————————————
st.set_page_config(page_title="Guppy EMA Screener", layout="wide")
st.title("📈 Guppy EMA Screener")

# ——— Sidebar Filters —————————————————————————
st.sidebar.header("Ticker Basket Filters")
min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=1e9)
min_price = st.sidebar.number_input("Min Price ($)", min_value=0.0, value=10.0)
exchange = st.sidebar.selectbox("Exchange", ["nasdaq", "nyse", "amex"])
if st.sidebar.button("Refresh Basket"):
    st.session_state.refresh = True

# ——— Fetch Ticker Basket —————————————————————
def fetch_tickers(exchange):
    url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange={exchange}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    data = r.json().get("data", {}).get("table", {}).get("rows", [])
    df = pd.DataFrame(data)
    # convert types
    df['marketCap'] = df['marketCap'].replace({'-':0}).str.replace('[\$,]', '', regex=True).astype(float)
    df['lastSale'] = df['lastSale'].replace({'-':0}).str.replace('[\$,]', '', regex=True).astype(float)
    return df[(df.marketCap >= min_mktcap) & (df.lastSale >= min_price)]['symbol'].tolist()

if 'tickers' not in st.session_state or st.session_state.get('refresh', False):
    tickers = fetch_tickers(exchange)
    st.session_state.tickers = tickers
    st.session_state.refresh = False
else:
    tickers = st.session_state.tickers

st.sidebar.write(f"📊 {len(tickers)} tickers in basket")

# ——— Date Selection —————————————————————————
start = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=90))
end = st.sidebar.date_input("End Date", datetime.date.today())

# ——— Core Logic ————————————————————————————
def process(ticker):
    df = yf.download(ticker, start=start, end=end)
    if df.empty: return None
    for p in [3,5,8,10,12,15]:
        df[f"EMA_{p}"] = df['Close'].ewm(span=p, adjust=False).mean()
    def cls(r):
        if r['EMA_3']>r['EMA_5']>r['EMA_8']>r['EMA_10']>r['EMA_12']>r['EMA_15']:
            return "Long"
        if r['EMA_15']>r['EMA_12']>r['EMA_10']>r['EMA_8']>r['EMA_5']>r['EMA_3']:
            return "Short"
        return "N/A"
    df['Signal'] = df.apply(cls, axis=1)
    df['Ticker'] = ticker
    return df.reset_index()

# ——— Run Analysis ————————————————————————————
if st.button("Run Guppy Screener"):
    all_data = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        res = process(t)
        if res is not None:
            all_data.append(res)
        progress.progress((i+1)/len(tickers))
    st.success("✅ Complete")
    if all_data:
        final = pd.concat(all_data)
        cols = ['Ticker','Date','Open','High','Low','Close'] + [f'EMA_{p}' for p in [3,5,8,10,12,15]] + ['Signal']
        final = final[cols].sort_values(['Ticker','Date'])
        st.dataframe(final)
        st.download_button("📥 Download CSV", final.to_csv(index=False), "guppy_signals.csv", "text/csv")
    else:
        st.warning("No data found for your filters.")

