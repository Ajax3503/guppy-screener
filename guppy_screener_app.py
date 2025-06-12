# guppy_screener_app.py
import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import gc  # Garbage collection

# ——— App Header —————————————————————————————
st.set_page_config(page_title="Guppy EMA Screener", layout="wide")
st.title("📈 Guppy EMA Screener")

# ——— Sidebar Filters —————————————————————————
st.sidebar.header("Ticker Basket Filters")

# Original default values maintained
min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=1000000000, step=1000000)
min_price = st.sidebar.number_input("Min Price ($)", min_value=0.0, value=10.0, step=0.1)

exchange = st.sidebar.selectbox("Exchange", ["nasdaq", "nyse", "amex"])

# Optimization 2: Batch size control
batch_size = st.sidebar.slider("Batch Size (fewer = less memory)", min_value=10, max_value=100, value=25)

# ——— Fetch Ticker Basket —————————————————————
@st.cache_data(ttl=3600, max_entries=3)  # Optimization 3: Limited cache entries
def fetch_tickers(exchange, min_mktcap, min_price):
    """Fetch and filter tickers from NASDAQ API"""
    try:
        url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange={exchange}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)  # Optimization 5: Shorter timeout
        r.raise_for_status()
        
        data = r.json().get("data", {}).get("table", {}).get("rows", [])
        if not data:
            st.error("No data received from API")
            return []
            
        # Optimization 6: Process data in chunks to reduce memory
        df = pd.DataFrame(data)
        
        # Optimization 7: Simplified conversion function
        def quick_convert(series, default=0.0):
            """Quick conversion with minimal memory usage"""
            return pd.to_numeric(
                series.astype(str).str.replace(r'[\$,-]', '', regex=True).replace('', '0'),
                errors='coerce'
            ).fillna(default)
        
        # Check for symbol column
        if 'symbol' not in df.columns:
            st.error("No 'symbol' column found")
            return []
        
        # Handle different column names efficiently
        price_col = next((col for col in ['lastSale', 'price', 'last', 'close'] if col in df.columns), None)
        mktcap_col = next((col for col in ['marketCap', 'marketcap', 'market_cap'] if col in df.columns), None)
        
        # Apply filters
        mask = (df['symbol'].notna()) & (df['symbol'] != '')
        
        if mktcap_col:
            df['mktcap_val'] = quick_convert(df[mktcap_col])
            mask &= (df['mktcap_val'] >= min_mktcap)
        
        if price_col:
            df['price_val'] = quick_convert(df[price_col])
            mask &= (df['price_val'] >= min_price)
        
        # Optimization 8: Return only symbols, drop the dataframe
        symbols = df.loc[mask, 'symbol'].tolist()
        del df  # Explicit cleanup
        gc.collect()  # Force garbage collection
        
        return symbols  # Return all filtered symbols
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return []

# Session state management - Fix Issue #2: Force refresh when needed
# Fix Issue #2: Add key to button and clear cache when refreshing
if st.sidebar.button("Refresh Basket", key="refresh_btn"):
    # Clear the cached function to force refresh
    fetch_tickers.clear()
    st.session_state.refresh = True

if 'tickers' not in st.session_state or st.session_state.get('refresh', False):
    with st.spinner("Fetching tickers..."):
        tickers = fetch_tickers(exchange, min_mktcap, min_price)
        st.session_state.tickers = tickers
        st.session_state.refresh = False
else:
    tickers = st.session_state.tickers

st.sidebar.write(f"📊 {len(tickers)} tickers in basket")

# ——— Date Selection —————————————————————————
# Original 90-day default maintained
start = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=90))
end = st.sidebar.date_input("End Date", datetime.date.today())

if start >= end:
    st.sidebar.error("Start date must be before end date")
    st.stop()

# ——— Optimized Core Logic ————————————————————————————
def process_ticker_lightweight(ticker, start_date, end_date):
    """Lightweight ticker processing with minimal memory usage"""
    try:
        # Optimization 11: Download only Close prices initially
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < 20:
            return None
        
        # Optimization 12: Work only with Close prices, drop other columns
        close_prices = df['Close'].dropna()
        if len(close_prices) < 20:
            return None
            
        # Optimization 13: Calculate EMAs efficiently using numpy
        ema_periods = [3, 5, 8, 10, 12, 15]
        emas = {}
        
        for period in ema_periods:
            alpha = 2.0 / (period + 1)
            ema_values = np.zeros(len(close_prices))
            ema_values[0] = close_prices.iloc[0]
            
            for i in range(1, len(close_prices)):
                ema_values[i] = alpha * close_prices.iloc[i] + (1 - alpha) * ema_values[i-1]
            
            emas[f'EMA_{period}'] = ema_values[-1]  # Only keep the latest value
        
        # Fix Issue #1: Updated signal calculation with new EMA logic
        latest_emas = {p: emas[f'EMA_{p}'] for p in ema_periods}
        
        # Check for valid EMAs
        if any(np.isnan(ema) for ema in latest_emas.values()):
            return None
        
        # New Signal logic for capturing transition points
        # Long signal: EMAs (3 > 5 > 8 > 10 > 12 < 15)
        long_condition = (
            latest_emas[3] > latest_emas[5] > 
            latest_emas[8] > latest_emas[10] > 
            latest_emas[12] and latest_emas[12] < latest_emas[15]
        )
        
        # Short signal: EMAs (15 < 12 > 10 > 8 > 5 > 3)
        short_condition = (
            latest_emas[15] < latest_emas[12] > 
            latest_emas[10] > latest_emas[8] > 
            latest_emas[5] > latest_emas[3]
        )
        
        if long_condition:
            signal = "Long"
        elif short_condition:
            signal = "Short"  
        else:
            signal = "N/A"
        
        # Optimization 15: Return only essential data
        result = {
            'Ticker': ticker,
            'Date': close_prices.index[-1].strftime('%Y-%m-%d'),
            'Close': close_prices.iloc[-1],
            'Signal': signal
        }
        
        # Add EMA values
        for period in ema_periods:
            result[f'EMA_{period}'] = round(emas[f'EMA_{period}'], 2)
        
        # Cleanup
        del df, close_prices, emas
        
        return result
        
    except Exception as e:
        return None

# ——— Run Analysis with Memory Management ————————————————————————————
if st.button("Run Guppy Screener"):
    if not tickers:
        st.warning("No tickers found with your current filters.")
        st.stop()
    
    # Optimization 16: Process in batches to manage memory
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = min(len(tickers), batch_size * 10)  # Limit total processing
    selected_tickers = tickers[:total_tickers]
    
    for i in range(0, len(selected_tickers), batch_size):
        batch = selected_tickers[i:i+batch_size]
        batch_results = []
        
        status_text.text(f"Processing batch {i//batch_size + 1} ({len(batch)} tickers)")
        
        for j, ticker in enumerate(batch):
            result = process_ticker_lightweight(ticker, start, end)
            if result:
                batch_results.append(result)
            
            # Update progress
            progress = (i + j + 1) / len(selected_tickers)
            progress_bar.progress(progress)
        
        # Add batch results and force garbage collection
        results.extend(batch_results)
        del batch_results
        gc.collect()
    
    status_text.text(f"✅ Complete! Processed {len(results)} tickers successfully.")
    
    if results:
        # Optimization 17: Create DataFrame from list of dicts (more efficient)
        final_df = pd.DataFrame(results)
        
        # Display summary first
        st.subheader("📊 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", len(final_df))
        with col2:
            long_count = len(final_df[final_df['Signal'] == 'Long'])
            st.metric("Long Signals", long_count)
        with col3:
            short_count = len(final_df[final_df['Signal'] == 'Short'])
            st.metric("Short Signals", short_count)
        with col4:
            na_count = len(final_df[final_df['Signal'] == 'N/A'])
            st.metric("No Signal", na_count)
        
        # Optimization 18: Show only signals, not N/A
        signals_only = final_df[final_df['Signal'] != 'N/A'].copy()
        
        if not signals_only.empty:
            st.subheader("📈 Active Signals")
            st.dataframe(signals_only, use_container_width=True)
            
            # Download button
            csv_data = signals_only.to_csv(index=False)
            st.download_button(
                label="📥 Download Signals CSV",
                data=csv_data,
                file_name=f"guppy_signals_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.info("No active signals found in the processed tickers.")
        
        # Cleanup
        del final_df, results
        gc.collect()
        
    else:
        st.warning("No data found. Try reducing filters or selecting different exchange.")

# Optimization 19: Memory usage info
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Memory Tips:**")
st.sidebar.markdown("- Use higher price/market cap filters")
st.sidebar.markdown("- Reduce batch size if app crashes")
st.sidebar.markdown("- Shorter date ranges use less memory")

# Add signal logic explanation
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Signal Logic:**")
st.sidebar.markdown("**Long:** 3>5>8>10>12<15 (transition)")
st.sidebar.markdown("**Short:** 15<12>10>8>5>3 (transition)")
