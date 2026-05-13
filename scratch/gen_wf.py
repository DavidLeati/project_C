import sys

with open('trade/engine_monolithic.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Substituir o início do run_backtest
target = '''        # 1. Carrega dados
        data = self.loader.load_multi_timeframe_sol(
            train_ratio=0.7,
            max_samples=self.max_samples,
        )

        data_dev: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for tf in self.TIMEFRAMES:
            Xtr, Ytr, Xte, Yte = data[tf]
            data_dev[tf] = (
                Xtr.to(self.device),
                Ytr.to(self.device),
                Xte.to(self.device),
                Yte.to(self.device),
            )

        X_train_15m = data_dev["15m"][0]
        Y_train_15m = data_dev["15m"][1]
        X_test_15m  = data_dev["15m"][2]
        Y_test_15m  = data_dev["15m"][3]

        train_len = X_train_15m.shape[0]
        test_len  = X_test_15m.shape[0]'''

replacement = '''        # 1. Carrega dados
        data = self.loader.load_multi_timeframe_sol(
            train_ratio=1.0,  # Carrega tudo em Xtr
            max_samples=self.max_samples,
        )

        data_all = {}
        for tf in self.TIMEFRAMES:
            Xtr, Ytr, _, _ = data[tf]
            data_all[tf] = (Xtr.to(self.device), Ytr.to(self.device))

        total_samples = data_all["15m"][0].shape[0]
        
        # --- PARÂMETROS DO WALK-FORWARD ---
        WALK_TRAIN_LEN = 30000
        WALK_TEST_LEN  = 10000
        N_WALKS = (total_samples - WALK_TRAIN_LEN) // WALK_TEST_LEN
        
        print(f"\\n[Walk-Forward] Total samples: {total_samples}")
        print(f"[Walk-Forward] Config: Treino={WALK_TRAIN_LEN}, Teste={WALK_TEST_LEN}, Walks={N_WALKS}")
        
        # Armazenar resultados combinados OOS
        wf_oos_agent_net = []
        wf_oos_bh_ret    = []
        wf_oos_pos       = []
        wf_oos_gross     = []
        wf_oos_cost      = []
        wf_oos_ret       = []'''

code = code.replace(target, replacement)

# Criar import TemporalHistoryBuffer
if 'from src.runtime.temporal import TemporalHistoryBuffer' not in code:
    code = code.replace('from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig',
                        'from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig\\nfrom src.runtime.temporal import TemporalHistoryBuffer')

# Identar do `print(f"\\n Timeframes:")` até `tmp_ckpt_path` dentro do `for w in range(N_WALKS):`
# O OOS original precisa ser trocado tbm

print("DONE")
