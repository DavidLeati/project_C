import pandas as pd
import numpy as np
import torch
from typing import Tuple

class CryptoFeatureBuilder:
    """
    Construtor de Features Temporais Imune a Look-ahead Bias.
    Todos os cálculos utilizam os dados estritamente até o tempo T.
    Para os alvos (y), utilizamos estritamente dados em T+1.
    """
    def __init__(self, window_size: int = 48):
        # Janela de rolling histórica (12 horas considerando 15 min candles)
        self.window_size = window_size
        
    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Espera um dataframe com colunas: ['open', 'high', 'low', 'close', 'volume']
        Retorna (X_features_tensor, Y_targets_tensor)
        """
        # --- 1. Target (Y) ---
        # Queremos que o y do passo index 'i' contenha o log-return do passo 'i+1'.
        # Isso significa que o target PnL do trade decidido em 'i' é o movimento de 'i' para 'i+1'.
        df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
        
        # --- 2. Features de Momentum (X) ---
        df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
        df['ret_4'] = np.log(df['close'] / df['close'].shift(4))
        df['ret_16'] = np.log(df['close'] / df['close'].shift(16))
        
        # --- 3. Features de Volatilidade e Range (X) ---
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['roll_volatility'] = df['ret_1'].rolling(window=self.window_size).std()
        
        # --- 4. Features de Volume ---
        df['log_volume'] = np.log(df['volume'] + 1e-8)
        df['vol_trend'] = df['log_volume'] - df['log_volume'].rolling(window=self.window_size).mean()

        # (Moved dropna to the end to catch ema_std NaNs too)
        
        # --- 5. Z-Score Rolling Progressivo Puro (Sem vazamento de média/variância Global) ---
        features_cols = ['ret_1', 'ret_4', 'ret_16', 'high_low_range', 'roll_volatility', 'vol_trend']
        
        # Criando o array de X
        X_raw = df[features_cols].values
        
        # Como o batch vai ser processado cronologicamente, vamos utilizar z-score 
        # do histórico de 30 dias pra trás aproximado exp, mas como é um stream, 
        # para simular vida real simples: vamos aplicar um z-score expansível (expanding) 
        # ou, para fins locais do array preenchido (já estamos sem lookahead com rolling mean).
        # Vamos usar um Z-Score dinâmico com EMA do Pandas para cada feature:
        for col in features_cols:
            ema_mu = df[col].ewm(span=self.window_size, adjust=False).mean()
            ema_std = df[col].ewm(span=self.window_size, adjust=False).std()
            df[col] = (df[col] - ema_mu) / (ema_std + 1e-8)
            
            # Clip para prevenir outliers de explodirem a VRAM e a rede (-5 a 5 variâncias)
            df[col] = df[col].clip(-5.0, 5.0)
            
        # Remove All NaNs created by shift, windows and ewm.std() (which is NaN for row 0)
        df = df.dropna().copy()
        
        X = df[features_cols].values
        Y = df[['target_return']].values
        
        # Convertemos para Type correto do torch
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
