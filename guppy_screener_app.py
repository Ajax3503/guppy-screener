# guppy_screener_app.py
import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import datetime
import numpy as np

# ——— App Header —————————————————————————————
st.set_page_config(page_title="Guppy EMA Screener", layout="wide")
st.title("📈 Guppy EMA Screener")

# ——— Sidebar Filters —————————————————————————
st.sidebar.header("Ticker Basket Filters")

# Fix 1: Ensure consistent numeric types
min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=1000000000, step=1000000)
min_price = st.sidebar.number_input("Min Price ($)", min_value=0.0, value=10.0, step=0.1)

exchange = st.sidebar.selectbox("Exchange", ["nasdaq", "nyse", "amex"])

if st.sidebar.button("Refresh Basket"):
    st.session_state.refresh = True

# ——— Fetch Ticker Basket —————————————————————
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_tickers(exchange, min_mktcap, min_price):
    """Fetch and filter tickers from NASDAQ API"""
    try:
        url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange={exchange}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()  # Raise an exception for bad status codes
        
        data = r.json().get("data", {}).get("table", {}).get("rows", [])
        if not data:
            st.error("No data received from API")
            return []
            
        df = pd.DataFrame(data)
        
        # Fix 2: Better error handling for data conversion
        def safe_convert_to_float(series, default=0.0):
            """Safely convert series to float, handling various formats"""
            return (series
                    .replace({'-': default, '': default, None: default})
                    .astype(str)
                    .str.replace(r'[\$,]', '', regex=True)
                    .replace({'': default})
                    .astype(float, errors='coerce')
                    .fillna(default))
        
        # Check if required columns exist
        required_cols = ['marketCap', 'lastSale', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return []
        
        # Convert types safely
        df['marketCap'] = safe_convert_to_float(df['marketCap'])
        df['lastSale'] = safe_convert_to_float(df['lastSale'])
        
        # Filter based on criteria
        filtered_df = df[
            (df['marketCap'] >= min_mktcap) & 
            (df['lastSale'] >= min_price) &
            (df['symbol'].notna()) &
            (df['symbol'] != '')
        ]
        
        return filtered_df['symbol'].tolist()
        
    except requests.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return []

# Fix 3: Better session state handling
if 'tickers' not in st.session_state or st.session_state.get('refresh', False):
    with st.spinner("Fetching tickers..."):
        tickers = fetch_tickers(exchange, min_mktcap, min_price)
        st.session_state.tickers = tickers
        st.session_state.refresh = False
else:
    tickers = st.session_state.tickers

st.sidebar.write(f"📊 {len(tickers)} tickers in basket")

# ——— Date Selection —————————————————————————
start = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=90))
end = st.sidebar.date_input("End Date", datetime.date.today())

# Fix 4: Validate date inputs
if start >= end:
    st.sidebar.error("Start date must be before end date")
    st.stop()

# ——— Core Logic ————————————————————————————
@st.cache_data
def process_ticker(ticker, start_date, end_date):
    """Process a single ticker for Guppy EMA analysis"""
    try:
        # Fix 5: Better yfinance error handling
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, show_errors=False)
        
        if df.empty or len(df) < 15:  # Need at least 15 days for EMA calculation
            return None
            
        # Fix 6: Handle potential NaN values
        df = df.dropna()
        if df.empty:
            return None
            
        # Calculate EMAs
        ema_periods = [3, 5, 8, 10, 12, 15]
        for period in ema_periods:
            df[f"EMA_{period}"] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Fix 7: Vectorized signal calculation (more efficient)
        def calculate_signal(row):
            """Calculate Guppy signal for a single row"""
            emas = [row[f'EMA_{p}'] for p in ema_periods]
            
            # Check if all EMAs are valid numbers
            if any(pd.isna(ema) for ema in emas):
                return "N/A"
            
            # Long signal: EMAs in ascending order
            if all(emas[i] > emas[i+1] for i in range(len(emas)-1)):
                return "Long"
            # Short signal: EMAs in descending order
            elif all(emas[i] < emas[i+1] for i in range(len(emas)-1)):
                return "Short"
            else:
                return "N/A"
        
        df['Signal'] = df.apply(calculate_signal, axis=1)
        df['Ticker'] = ticker
        
        return df.reset_index()
        
    except Exception as e:
        st.warning(f"Error processing {ticker}: {str(e)}")
        return None

# ——— Run Analysis ————————————————————————————
if st.button("Run Guppy Screener"):
    if not tickers:
        st.warning("No tickers found with your current filters. Try adjusting the criteria.")
        st.stop()
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fix 8: Better progress tracking and error handling
    successful_tickers = 0
    failed_tickers = 0
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Processing {ticker} ({i+1}/{len(tickers)})")
        
        try:
            result = process_ticker(ticker, start, end)
            if result is not None:
                all_data.append(result)
                successful_tickers += 1
            else:
                failed_tickers += 1
        except Exception as e:
            failed_tickers += 1
            st.warning(f"Failed to process {ticker}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(tickers))
    
    status_text.text(f"✅ Complete! Processed {successful_tickers} tickers successfully, {failed_tickers} failed.")
    
    if all_data:
        # Fix 9: Better data concatenation and column handling
        try:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Ensure all required columns exist
            base_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close']
            ema_cols = [f'EMA_{p}' for p in [3, 5, 8, 10, 12, 15]]
            signal_cols = ['Signal']
            
            all_cols = base_cols + ema_cols + signal_cols
            existing_cols = [col for col in all_cols if col in final_df.columns]
            
            final_df = final_df[existing_cols].sort_values(['Ticker', 'Date'])
            
            # Display results
            st.subheader("📊 Results")
            st.dataframe(final_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Signals", len(final_df))
            with col2:
                long_signals = len(final_df[final_df['Signal'] == 'Long'])
                st.metric("Long Signals", long_signals)
            with col3:
                short_signals = len(final_df[final_df['Signal'] == 'Short'])
                st.metric("Short Signals", short_signals)
            
            # Download button
            csv_data = final_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"guppy_signals_{datetime.date.today()}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing results: {str(e)}")
    else:
        st.warning("No data found for your filters. Try:")
        st.write("- Reducing minimum market cap or price filters")
        st.write("- Selecting a different exchange")
        st.write("- Adjusting the date range")
