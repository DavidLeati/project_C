import os
import sys

# Guard para execução direta
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
from typing import Tuple, Dict

try:
    from .features import CryptoFeatureBuilder
except ImportError:
    from trade.features import CryptoFeatureBuilder  # type: ignore[no-redef]


class CryptoDataLoader:
    """
    Carrega arquivos Parquet reais, invoca a criação de Features temporais
    e fragmenta os tensores em Train/Test Split cronológicos para prever Leak.
    """
    def __init__(self, data_dir: str = "trade/data"):
        self.data_dir = data_dir
        # Builders por timeframe — window_size proporcional à granularidade
        self._builders: Dict[str, CryptoFeatureBuilder] = {
            "15m": CryptoFeatureBuilder(window_size=48),    # 12h em 15m
            "1h":  CryptoFeatureBuilder(window_size=24),    # 24h em 1h
            "4h":  CryptoFeatureBuilder(window_size=42),    # 1 semana em 4h
            "1d":  CryptoFeatureBuilder(window_size=30),    # 30 dias em 1d
        }

    # ------------------------------------------------------------------
    # API original: carrega um único ativo/timeframe
    # ------------------------------------------------------------------
    def load_asset(
        self,
        symbol_filename: str,
        train_ratio: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Carrega um arquivo (ex: 'BTCUSDT_15m_1825d.parquet') e garante split cronológico.
        Retorna: (X_train, Y_train, X_test, Y_test)
        """
        filepath = os.path.join(self.data_dir, symbol_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset não encontrado: {filepath}")

        print(f"[{symbol_filename}] Baixando na memória...")
        df_raw = pd.read_parquet(filepath)

        if 'open_time' in df_raw.columns:
            df_raw = df_raw.sort_values(by='open_time').reset_index(drop=True)

        print(f"[{symbol_filename}] Processando Features Rigorosas (Z-Score Rolling)...")
        X, Y = self._builders["15m"].transform(df_raw)

        total_len = X.shape[0]
        train_len = int(total_len * train_ratio)

        X_train, Y_train = X[:train_len], Y[:train_len]
        X_test, Y_test = X[train_len:], Y[train_len:]

        print(f"[{symbol_filename}] Split -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        return X_train, Y_train, X_test, Y_test

    # ------------------------------------------------------------------
    # Nova API: carrega os 4 timeframes da SOL alinhados ao 15m
    # ------------------------------------------------------------------
    def load_multi_timeframe_sol(
        self,
        train_ratio: float = 0.7,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Carrega os 4 timeframes da Solana (15m, 1h, 4h, 1d) e os alinha
        temporalmente ao timeframe de menor granularidade (15m) via forward-fill.

        O alinhamento é feito com .ffill() para garantir zero look-ahead bias:
        cada vela de timeframe maior é propagada para os passos de 15m
        *após* o seu fechamento, nunca antes.

        Retorna um dicionário com chaves "15m", "1h", "4h", "1d", onde cada
        valor é (X_train, Y_train, X_test, Y_test) com os índices cronológicos
        sincronizados ao índice 15m.
        """
        timeframes = {
            "15m": "SOLUSDT_15m_1825d.parquet",
            "1h":  "SOLUSDT_1h_1825d.parquet",
            "4h":  "SOLUSDT_4h_1825d.parquet",
            "1d":  "SOLUSDT_1d_1825d.parquet",
        }

        # 1. Carrega todos os DataFrames brutos
        raw_dfs: Dict[str, pd.DataFrame] = {}
        for tf, fname in timeframes.items():
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Dataset não encontrado: {fpath}")
            print(f"[SOL/{tf}] Carregando {fname}...")
            df = pd.read_parquet(fpath)
            if 'open_time' in df.columns:
                df = df.sort_values('open_time').reset_index(drop=True)
                df = df.set_index('open_time')
            else:
                raise ValueError(f"Coluna 'open_time' ausente em {fname}")
            raw_dfs[tf] = df

        # 2. Usa o índice 15m como âncora temporal de referência
        anchor_index = raw_dfs["15m"].index

        # 3. Para cada timeframe >15m, re-indexa ao grid de 15m usando ffill
        #    Isso garante que o dado de uma vela maior só "apareça" nos
        #    15m APÓS o fechamento daquela vela (sem look-ahead).
        aligned_dfs: Dict[str, pd.DataFrame] = {"15m": raw_dfs["15m"].copy()}
        for tf in ["1h", "4h", "1d"]:
            df_tf = raw_dfs[tf]
            # Reindex ao grid 15m: NaN onde não há vela naquele instante
            df_reindexed = df_tf.reindex(anchor_index)
            # Forward-fill: propaga o dado da última vela fechada
            df_reindexed = df_reindexed.ffill()
            # Backfill somente para preencher as primeiras linhas onde ffill não alcança
            df_reindexed = df_reindexed.bfill()
            aligned_dfs[tf] = df_reindexed

        # 4. Determina o índice de corte comum (após dropna de todas as features)
        #    Processamos features em cada timeframe separadamente
        feature_results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        min_length = None
        for tf, builder in self._builders.items():
            print(f"[SOL/{tf}] Processando Features (Z-Score Rolling EMA)...")
            df_aligned = aligned_dfs[tf].reset_index(drop=False)
            # Renomeia 'open_time' de volta para coluna se precisar
            if 'open_time' in df_aligned.columns:
                df_aligned = df_aligned.rename(columns={'open_time': 'open_time'})
            X, Y = builder.transform(df_aligned)
            feature_results[tf] = (X, Y)
            if min_length is None or X.shape[0] < min_length:
                min_length = X.shape[0]

        # 5. Trunca todos ao comprimento mínimo para garantir alinhamento perfeito
        #    (as janelas de rolling de diferentes tamanhos criam NaN nas primeiras linhas)
        results: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        train_len = int(min_length * train_ratio)

        for tf, (X, Y) in feature_results.items():
            # Pega os ÚLTIMOS min_length elementos (alinhados ao final da série)
            X_aligned = X[-min_length:]
            Y_aligned = Y[-min_length:]

            X_train, Y_train = X_aligned[:train_len], Y_aligned[:train_len]
            X_test, Y_test = X_aligned[train_len:], Y_aligned[train_len:]

            print(f"[SOL/{tf}] Split -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X.shape[1]}")
            results[tf] = (X_train, Y_train, X_test, Y_test)

        return results
