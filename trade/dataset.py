import os
import pandas as pd
import torch
from typing import Tuple

from .features import CryptoFeatureBuilder

class CryptoDataLoader:
    """
    Carrega o arquivo Parquet real, invoca a criação de Features temporais
    e fragmenta os tensores em Train/Test Split cronológicos para prever Leak.
    """
    def __init__(self, data_dir: str = "trade/data"):
        self.data_dir = data_dir
        self.builder = CryptoFeatureBuilder(window_size=48) # 12 horas base 

    def load_asset(self, symbol_filename: str, train_ratio: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Carrega um arquivo (ex: 'BTCUSDT_15m_1825d.parquet') e garante split cronológico.
        Retornos: (X_train, Y_train, X_test, Y_test)
        """
        filepath = os.path.join(self.data_dir, symbol_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset não encontrado: {filepath}")

        # Loading raw data
        print(f"[{symbol_filename}] Baixando na memória...")
        df_raw = pd.read_parquet(filepath)
        
        # Garante a ordenação temporal correta por segurança
        if 'open_time' in df_raw.columns:
            df_raw = df_raw.sort_values(by='open_time').reset_index(drop=True)

        print(f"[{symbol_filename}] Processando Features Rigorosas (Z-Score Rolling)...")
        # Injeta Feature Engineering e corta fora Nulls/Lookaheads
        X, Y = self.builder.transform(df_raw)
        
        total_len = X.shape[0]
        train_len = int(total_len * train_ratio)
        
        # Split sem permutação nem shuffle (A temporalidade importa pro Stream Online da GRU)
        X_train, Y_train = X[:train_len], Y[:train_len]
        X_test, Y_test = X[train_len:], Y[train_len:]
        
        print(f"[{symbol_filename}] Split Finalizado -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        return X_train, Y_train, X_test, Y_test
