import pandas as pd
import numpy as np

# Carrega o parquet
df = pd.read_parquet('trade/data/SOLUSDT_15m_1825d.parquet')
if 'open_time' in df.columns:
    df = df.sort_values('open_time').reset_index(drop=True)

# Total de dados
oos_len = 6000

df_oos = df.iloc[-oos_len:].copy()
df_hist = df.iloc[:-oos_len].copy()

df_oos['ret'] = np.log(df_oos['close'] / df_oos['close'].shift(1))
df_hist['ret'] = np.log(df_hist['close'] / df_hist['close'].shift(1))

ret_oos = df_oos['ret'].dropna()
ret_hist = df_hist['ret'].dropna()

def stats(ret_series, name):
    cum_ret = ret_series.sum() * 100
    vol = ret_series.std() * np.sqrt(96 * 365) * 100
    up_pct = (ret_series > 0).mean() * 100
    down_pct = (ret_series < 0).mean() * 100
    autocorr = ret_series.autocorr(lag=1)
    
    print(f'[{name}]')
    print(f'  Retorno Cumulativo: {cum_ret:+.2f}%')
    print(f'  Volatilidade Anual: {vol:.2f}%')
    print(f'  Candles de Alta:    {up_pct:.1f}%')
    print(f'  Candles de Baixa:   {down_pct:.1f}%')
    print(f'  Autocorr (Lag 1):   {autocorr:+.4f}')
    print(f'  Assimetria (Skew):  {ret_series.skew():+.2f}')
    print('-'*40)

print(f'Datas OOS: {df_oos.iloc[0]["open_time"]} a {df_oos.iloc[-1]["open_time"]}')
stats(ret_hist, "HISTORICO (Todo o passado)")
stats(ret_oos, "OOS (Ultimos 6000 candles)")
