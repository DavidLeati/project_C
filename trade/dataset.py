import os
import sys

# Guard para execucao direta
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
from typing import Tuple, Dict, Optional

try:
    from .features import CryptoFeatureBuilder
except ImportError:
    from trade.features import CryptoFeatureBuilder  # type: ignore[no-redef]


class CryptoDataLoader:
    """
    Carrega arquivos Parquet reais, invoca a criacao de Features temporais
    e fragmenta os tensores em Train/Test Split cronologicos para prever Leak.

    Inter-mercado: para cada timeframe, tenta injetar BTC e ETH no mesmo
    granularity (ex: SOL 4h recebe BTC 4h e ETH 4h). Se o arquivo nao
    existir, o CryptoFeatureBuilder usa fallback zero para btc_ret_1/eth_ret_1.
    """

    def __init__(self, data_dir: str = "trade/data"):
        self.data_dir = data_dir
        # Builders por timeframe — window_size proporcional a granularidade
        self._builders: Dict[str, CryptoFeatureBuilder] = {
            "15m": CryptoFeatureBuilder(window_size=48),    # 12h em 15m
            "1h":  CryptoFeatureBuilder(window_size=24),    # 24h em 1h
            "4h":  CryptoFeatureBuilder(window_size=42),    # 1 semana em 4h
            "1d":  CryptoFeatureBuilder(window_size=30),    # 30 dias em 1d
        }

    # ------------------------------------------------------------------
    # Helper: carrega parquet de um simbolo/timeframe se existir
    # ------------------------------------------------------------------

    def _try_load(self, symbol: str, tf: str, days: int = 1825) -> Optional[pd.DataFrame]:
        """
        Tenta carregar '{symbol}_{tf}_{days}d.parquet' do data_dir.
        Retorna None silenciosamente se o arquivo nao existir.
        """
        fname = f"{symbol}_{tf}_{days}d.parquet"
        fpath = os.path.join(self.data_dir, fname)
        if not os.path.exists(fpath):
            return None
        df = pd.read_parquet(fpath)
        if "open_time" in df.columns:
            df = df.sort_values("open_time").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # API original: carrega um unico ativo/timeframe
    # ------------------------------------------------------------------

    def load_asset(
        self,
        symbol_filename: str,
        train_ratio: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Carrega um arquivo (ex: 'BTCUSDT_15m_1825d.parquet') e garante split cronologico.
        Retorna: (X_train, Y_train, X_test, Y_test)
        """
        filepath = os.path.join(self.data_dir, symbol_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset nao encontrado: {filepath}")

        print(f"[{symbol_filename}] Baixando na memoria...")
        df_raw = pd.read_parquet(filepath)

        if "open_time" in df_raw.columns:
            df_raw = df_raw.sort_values(by="open_time").reset_index(drop=True)

        print(f"[{symbol_filename}] Processando Features Rigorosas (Z-Score Rolling)...")
        X, Y = self._builders["15m"].transform(df_raw)

        total_len = X.shape[0]
        train_len = int(total_len * train_ratio)

        X_train, Y_train = X[:train_len], Y[:train_len]
        X_test, Y_test   = X[train_len:], Y[train_len:]

        print(f"[{symbol_filename}] Split -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        return X_train, Y_train, X_test, Y_test

    # ------------------------------------------------------------------
    # Nova API: carrega os 4 timeframes da SOL alinhados ao 15m
    # com inter-mercado BTC/ETH no timeframe correto de cada T0
    # ------------------------------------------------------------------

    def load_multi_timeframe_sol(
        self,
        train_ratio: float = 0.7,
        max_samples: int = 20_000,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Carrega os 4 timeframes da Solana (15m, 1h, 4h, 1d) e os alinha
        temporalmente ao timeframe de menor granularidade (15m) via forward-fill.

        Para cada timeframe, injeta BTC e ETH no mesmo granularity:
          - SOL 15m recebe BTC 15m e ETH 15m
          - SOL 1h  recebe BTC 1h  e ETH 1h
          - SOL 4h  recebe BTC 4h  e ETH 4h
          - SOL 1d  recebe BTC 1d  e ETH 1d

        Se o arquivo BTC/ETH nao existir para um timeframe, o builder
        usa fallback zero para btc_ret_1/eth_ret_1 (sem erro).

        Retorna dicionario com chaves "15m", "1h", "4h", "1d", onde cada
        valor e (X_train, Y_train, X_test, Y_test) com indices cronologicos
        sincronizados ao indice 15m.
        """
        sol_files = {
            "15m": "SOLUSDT_15m_1825d.parquet",
            "1h":  "SOLUSDT_1h_1825d.parquet",
            "4h":  "SOLUSDT_4h_1825d.parquet",
            "1d":  "SOLUSDT_1d_1825d.parquet",
        }

        # 1. Carrega todos os DataFrames brutos da SOL
        raw_dfs: Dict[str, pd.DataFrame] = {}
        for tf, fname in sol_files.items():
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Dataset nao encontrado: {fpath}")
            print(f"[SOL/{tf}] Carregando {fname}...")
            df = pd.read_parquet(fpath)
            if "open_time" in df.columns:
                df = df.sort_values("open_time").reset_index(drop=True)
                df = df.set_index("open_time")
            else:
                raise ValueError(f"Coluna 'open_time' ausente em {fname}")
            raw_dfs[tf] = df

        # 2. Usa o indice 15m como ancora temporal de referencia
        anchor_index = raw_dfs["15m"].index

        # 3. Para cada timeframe >15m, re-indexa ao grid de 15m usando ffill
        aligned_dfs: Dict[str, pd.DataFrame] = {"15m": raw_dfs["15m"].copy()}
        for tf in ["1h", "4h", "1d"]:
            df_tf       = raw_dfs[tf]
            df_reindexed = df_tf.reindex(anchor_index)
            df_reindexed = df_reindexed.ffill()
            df_reindexed = df_reindexed.bfill()
            aligned_dfs[tf] = df_reindexed

        # 4. Carrega BTC e ETH para cada timeframe e injeta como colunas extras
        for tf in ["15m", "1h", "4h", "1d"]:
            df_sol = aligned_dfs[tf]

            for symbol, col_name in [("BTCUSDT", "btc_close"), ("ETHUSDT", "eth_close")]:
                df_ext = self._try_load(symbol, tf)
                if df_ext is not None:
                    print(f"[{symbol}/{tf}] Inter-mercado carregado -> injetando {col_name}")
                    df_ext = df_ext.set_index("open_time")[["close"]].rename(
                        columns={"close": col_name}
                    )
                    if tf == "15m":
                        # Alinhamento direto por indice
                        df_sol = df_sol.join(df_ext, how="left")
                    else:
                        # Para TFs maiores: o df_sol ja esta no grid 15m,
                        # entao reindexamos o externo ao grid 15m via ffill
                        df_ext_aligned = df_ext.reindex(anchor_index).ffill().bfill()
                        df_sol = df_sol.join(df_ext_aligned, how="left")
                else:
                    print(f"[{symbol}/{tf}] Arquivo nao encontrado — usando fallback zero para {col_name}")

            aligned_dfs[tf] = df_sol

        # 5. Processa features em cada timeframe
        feature_results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        min_length = None
        for tf, builder in self._builders.items():
            print(f"[SOL/{tf}] Processando Features (Z-Score Rolling EMA)...")
            df_aligned = aligned_dfs[tf].reset_index(drop=False)
            # Garante que open_time e coluna (para sessao UTC)
            if "open_time" in df_aligned.columns:
                df_aligned = df_aligned.rename(columns={"open_time": "open_time"})
            X, Y = builder.transform(df_aligned)
            feature_results[tf] = (X, Y)
            if min_length is None or X.shape[0] < min_length:
                min_length = X.shape[0]

        # 6. Aplica max_samples: pega os ULTIMOS N apos alinhamento minimo
        effective_len = min(min_length, max_samples) if max_samples is not None else min_length
        train_len     = int(effective_len * train_ratio)
        test_len      = effective_len - train_len

        results: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for tf, (X, Y) in feature_results.items():
            X_tail = X[-effective_len:]
            Y_tail = Y[-effective_len:]

            X_train, Y_train = X_tail[:train_len], Y_tail[:train_len]
            X_test,  Y_test  = X_tail[train_len:], Y_tail[train_len:]

            print(
                f"[SOL/{tf}] Split -> Train: {X_train.shape[0]} | "
                f"Test: {X_test.shape[0]} | Features: {X.shape[1]}"
            )
            results[tf] = (X_train, Y_train, X_test, Y_test)

        print(
            f"\n[Dataset] Usando ultimos {effective_len} passos de 15m "
            f"(~{effective_len * 15 / 60 / 24:.0f} dias) "
            f"| Train: {train_len} | Test: {test_len}"
        )
        return results
