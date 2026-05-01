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

        # 2. Usa o known_time (close_time) do 15m como ancora temporal de referencia causal
        anchor_index = raw_dfs["15m"].index + pd.Timedelta(minutes=15)
        
        # Cria o alvo universal em 15m (comum para todas as cabeças do trader)
        # O target do 15m avalia o retorno do momento da decisão até o fechamento do candle seguinte
        target_df = pd.DataFrame(index=anchor_index)
        target_df["target_return"] = (
            raw_dfs["15m"]["close"].shift(-1).values / raw_dfs["15m"]["close"].values - 1.0
        )

        feature_results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        min_length = None
        
        offset_map = {
            "15m": pd.Timedelta(minutes=15), 
            "1h": pd.Timedelta(hours=1), 
            "4h": pd.Timedelta(hours=4), 
            "1d": pd.Timedelta(days=1)
        }

        # 3. Processa cada timeframe no seu ritmo nativo, depois reindexa
        for tf in ["15m", "1h", "4h", "1d"]:
            df_tf = raw_dfs[tf].copy()
            builder = self._builders[tf]

            # 4. Injeta BTC e ETH no timeframe nativo
            for symbol, col_name in [("BTCUSDT", "btc_close"), ("ETHUSDT", "eth_close")]:
                df_ext = self._try_load(symbol, tf)
                if df_ext is not None:
                    print(f"[{symbol}/{tf}] Inter-mercado carregado -> injetando {col_name}")
                    df_ext = df_ext.set_index("open_time")[["close"]].rename(
                        columns={"close": col_name}
                    )
                    df_tf = df_tf.join(df_ext, how="left")
                else:
                    print(f"[{symbol}/{tf}] Arquivo nao encontrado — usando fallback zero para {col_name}")

            # 4.5. Injeta Funding Rate
            funding_path = os.path.join(self.data_dir, "SOLUSDT_funding_rate.parquet")
            if os.path.exists(funding_path):
                print(f"[SOL/{tf}] Injetando Histórico de Funding Rate")
                df_fund = pd.read_parquet(funding_path)
                df_fund = df_fund.set_index("fundingTime")[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
                
                df_tf = pd.merge_asof(
                    df_tf.sort_index(),
                    df_fund.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                )
                df_tf["funding_rate"] = df_tf["funding_rate"].fillna(0.0)
            else:
                df_tf["funding_rate"] = 0.0

            # 5. Processa features no ritmo nativo (Garante target e momentum corretos)
            print(f"[SOL/{tf}] Processando Features nativas (Z-Score Rolling EMA)...")
            df_tf = df_tf.reset_index(drop=False)
            df_feats = builder.build_features_df(df_tf)

            # 6. Alinha (Forward-fill) as features prontas para o grid de 15m (Lag Causal)
            if tf != "15m":
                print(f"[SOL/{tf}] Projetando matriz de features para o grid 15m...")
            
            # Desloca o index para o momento em que a feature realmente fica disponivel (close do candle)
            df_feats["known_time"] = df_feats["open_time"] + offset_map[tf]
            df_feats = df_feats.set_index("known_time").sort_index()
            
            anchor_df = pd.DataFrame(index=anchor_index).sort_index()
            
            df_aligned = pd.merge_asof(
                anchor_df,
                df_feats.drop(columns=["target_return"], errors="ignore"),
                left_index=True,
                right_index=True,
                direction="backward",
                allow_exact_matches=True,
            )
            
            # Adiciona o alvo universal do grid de 15m
            df_aligned["target_return"] = target_df["target_return"]
            df_aligned = df_aligned.dropna(subset=builder.FEATURE_COLS + ["target_return"])

            X = df_aligned[builder.FEATURE_COLS].values
            Y = df_aligned[["target_return"]].values
            
            X_t = torch.tensor(X, dtype=torch.float32)
            Y_t = torch.tensor(Y, dtype=torch.float32)

            feature_results[tf] = (X_t, Y_t)
            if min_length is None or X_t.shape[0] < min_length:
                min_length = X_t.shape[0]

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
