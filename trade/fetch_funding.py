import argparse
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

try:
    from binance.client import Client
except ImportError:
    raise ImportError("Please install python-binance: pip install python-binance")

def fetch_historical_funding(symbol: str, limit_days: int) -> pd.DataFrame:
    client = Client()
    print(f"\n[FUNDING] Iniciando download do Histórico de Funding para {symbol}...")
    
    # Binance permite até 1000 registros por chamada no funding rate
    # Funding acontece a cada 8 horas (3 por dia). 1000 registros = 333 dias.
    # Precisamos paginar usando o startTime
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (limit_days * 24 * 60 * 60 * 1000)
    
    all_funding = []
    current_start = start_time
    
    while current_start < end_time:
        print(f"[FUNDING]   Buscando a partir de: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')}")
        
        # O endpoint retorna no máximo 1000 itens a partir do startTime
        rates = client.futures_funding_rate(
            symbol=symbol,
            startTime=current_start,
            limit=1000
        )
        
        if not rates:
            break
            
        all_funding.extend(rates)
        
        # Pega o último fundingTime retornado e adiciona 1 ms para a próxima página
        last_time = rates[-1]['fundingTime']
        
        if last_time >= end_time or len(rates) < 1000:
            break
            
        current_start = last_time + 1
        time.sleep(0.5) # Respeitar rate limits
        
    if not all_funding:
        print("[FUNDING] Nenhum dado retornado.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_funding)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = pd.to_numeric(df['fundingRate'])
    df['markPrice'] = pd.to_numeric(df['markPrice'])
    
    df = df.sort_values('fundingTime').drop_duplicates(subset='fundingTime').reset_index(drop=True)
    print(f"[FUNDING] Download concluído! {len(df)} registros de Funding coletados.")
    return df

def save_funding(df: pd.DataFrame, save_dir: Path, symbol: str) -> Path:
    safe_symbol = symbol.upper()
    path = save_dir / f"{safe_symbol}_funding_rate.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[FUNDING] Salvo com sucesso em: {path}")
    return path

def main():
    parser = argparse.ArgumentParser(description="Puxa o Histórico de Funding Rates da Binance")
    parser.add_argument("--symbol", type=str, default="SOLUSDT")
    parser.add_argument("--days", type=int, default=1825, help="Dias retroativos (ex: 1825)")
    args = parser.parse_args()

    save_dir = Path(__file__).parent / "data"
    
    df_funding = fetch_historical_funding(args.symbol, args.days)
    if not df_funding.empty:
        save_funding(df_funding, save_dir, args.symbol)

if __name__ == "__main__":
    main()
