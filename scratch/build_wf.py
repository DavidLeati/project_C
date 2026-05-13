import re

with open('trade/engine_monolithic.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
in_loop = False
in_data_load = False

for i, line in enumerate(lines):
    # Rename class
    if line.startswith('class MonolithicEngine:'):
        out.append('class WalkForwardEngine:\n')
        continue
    
    # Inject import
    if 'from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig' in line:
        out.append(line)
        out.append('from src.runtime.temporal import TemporalHistoryBuffer\n')
        continue

    # Data loading
    if '# 1. Carrega dados' in line:
        in_data_load = True
        out.append(line)
        out.append('''        data = self.loader.load_multi_timeframe_sol(train_ratio=1.0, max_samples=self.max_samples)
        data_all = {}
        for tf in self.TIMEFRAMES:
            Xtr, Ytr, _, _ = data[tf]
            data_all[tf] = (Xtr.to(self.device), Ytr.to(self.device))
        total_samples = data_all["15m"][0].shape[0]
        WALK_TRAIN_LEN = 30000
        WALK_TEST_LEN  = 10000
        N_WALKS = (total_samples - WALK_TRAIN_LEN) // WALK_TEST_LEN
        print(f"\\n[Walk-Forward] Total samples: {total_samples}")
        print(f"[Walk-Forward] Config: Treino={WALK_TRAIN_LEN}, Teste={WALK_TEST_LEN}, Walks={N_WALKS}")
        wf_net_stream = []
        wf_pos_np = []
        wf_gross_np = []
        wf_cost_np = []
        wf_ret_np = []
        wf_bh_ret = []\n''')
        continue
    
    if in_data_load:
        if '# 2. Instancia modelos' in line:
            in_data_load = False
            out.append(line)
        else:
            continue
    
    if not in_data_load and not in_loop:
        if '# FASE 1-A: Warm-up exclusivo dos previsores' in line:
            in_loop = True
            # Add loop start
            out.append('''        for w in range(N_WALKS):
            train_start = w * WALK_TEST_LEN
            train_end   = train_start + WALK_TRAIN_LEN
            test_start  = train_end
            test_end    = test_start + WALK_TEST_LEN
            
            print(f"\\n{'='*75}")
            print(f" [WALK {w+1}/{N_WALKS}] Treino: {train_start} -> {train_end} | Teste: {test_start} -> {test_end}")
            print(f"{'='*75}")
            
            data_dev = {}
            for tf in self.TIMEFRAMES:
                X_full, Y_full = data_all[tf]
                data_dev[tf] = (
                    X_full[train_start:train_end], Y_full[train_start:train_end],
                    X_full[test_start:test_end],   Y_full[test_start:test_end]
                )
                
            X_train_15m, Y_train_15m, X_test_15m, Y_test_15m = data_dev["15m"]
            train_len = X_train_15m.shape[0]
            test_len  = X_test_15m.shape[0]
            
            for tf in self.TIMEFRAMES:
                rt = predictors[tf]
                rt.history_buffer = TemporalHistoryBuffer(max_events=rt.multi_scale_builder.required_history())
                rt.global_step = 0
            trader.history_buffer = TemporalHistoryBuffer(max_events=trader.multi_scale_builder.required_history())
            trader.global_step = 0\n\n''')
            out.append('    ' + line)
            continue
        out.append(line)
        continue
    
    if in_loop:
        if 'posicoes_teste = position_from_logits(' in line:
            in_loop = False
            # Wait, posicoes_teste block is the end of OOS.
            # I must indent posicoes_teste block too!
            # Let's just indent everything until `# Gráfico`!
            pass
        
        if '# Gráfico' in line:
            in_loop = False
            # Before plotting, concatenate the Walk-Forward aggregators!
            out.append('''        # --- FIM DOS WALKS ---
        net_stream = np.concatenate(wf_net_stream)
        pos_np     = np.concatenate(wf_pos_np)
        gross_np   = np.concatenate(wf_gross_np)
        cost_np    = np.concatenate(wf_cost_np)
        ret_np     = np.concatenate(wf_ret_np)
        
        bh_equity = np.cumsum(ret_np)
        agent_equity = np.cumsum(net_stream)
        test_len = len(agent_equity)
        
''')
            out.append(line)
            continue
        
        # OOS collection
        if 'net_stream = net_pnl_oos.cpu().numpy().flatten()' in line:
            out.append('    ' + line)
            out.append('            wf_net_stream.append(net_stream)\n')
            continue
        if 'pos_np = posicoes_teste.cpu().numpy().flatten()' in line:
            out.append('    ' + line)
            out.append('            wf_pos_np.append(pos_np)\n')
            continue
        if 'gross_np = gross_pnl_oos.cpu().numpy().flatten()' in line:
            out.append('    ' + line)
            out.append('            wf_gross_np.append(gross_np)\n')
            continue
        if 'cost_np = costs_oos.cpu().numpy().flatten()' in line:
            out.append('    ' + line)
            out.append('            wf_cost_np.append(cost_np)\n')
            continue
        if 'ret_np = Y_test_15m.cpu().numpy().flatten()' in line:
            out.append('    ' + line)
            out.append('            wf_ret_np.append(ret_np)\n')
            continue
        
        # Remove original equity curve cumsum inside the loop
        if 'bh_equity = np.cumsum(ret_np)' in line or 'agent_equity = np.cumsum(net_stream)' in line:
            continue
            
        if 'tmp_ckpt_path = f"models/monolithic_temp_pre_oos.pt"' in line:
            out.append(line.replace('monolithic_temp_pre_oos.pt', 'monolithic_temp_pre_oos_walk_{w}.pt'))
            continue

        if 'engine = MonolithicEngine()' in line:
            out.append(line.replace('MonolithicEngine', 'WalkForwardEngine'))
            continue
            
        out.append('    ' + line)

with open('trade/engine_walkforward.py', 'w', encoding='utf-8') as f:
    f.writelines(out)
