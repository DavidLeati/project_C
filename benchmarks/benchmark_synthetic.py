import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost não está instalado. Rode: pip install xgboost scikit-learn")
    import sys; sys.exit(1)

from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.replay import PrioritizedSurprisalBuffer, FixedSurprisalThreshold, SurprisalBufferConfig, AdaptiveQuantileThreshold
from src.runtime.surprisal import EMASurprisalEstimator
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec

def generate_synthetic_data(samples=1500, features=8):
    X = np.random.randn(samples, features)
    time_index = np.linspace(0, 10 * np.pi, samples)
    
    # Non-linear mixture sequence over time
    y = np.sin(X[:, 0] * time_index) + (0.5 * X[:, 1]**2) + np.random.normal(0, 0.1, samples)
    
    split = int(0.8 * samples)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    return (
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )

def evaluate_test(model, X_test, y_test, *, runtime=None):
    model.eval()
    with torch.no_grad():
        if runtime is not None:
            preds = runtime.predict(X_test, raw_output=True)
        else:
            preds, _ = model(X_test, raw_output=True)  # saída linear: y tem valores negativos
        mse = F.mse_loss(preds, y_test).item()
    model.train()
    return mse, preds.numpy()

def main():
    print("--------------------------------------------------")
    print("BENCHMARK: BASE VS REPLAY VS MEGAZORD (RUNTIME)")
    print("--------------------------------------------------")
    X_train, y_train, X_test, y_test = generate_synthetic_data(samples=4000, features=8)
    n_train = len(X_train)
    batch_size = 16
    epochs = 50

    print(f"[Dataset] Train: {n_train} amostras | Test: {len(X_test)} amostras")

    # ================= 1. XGBoost =================
    xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    t0 = time.time()
    xgb_model.fit(X_train.numpy(), y_train.numpy().ravel())
    xgb_preds = xgb_model.predict(X_test.numpy())
    xgb_mse = np.mean((xgb_preds - y_test.numpy().ravel())**2)
    print(f"[XGBoost Baseline] Treinado em {(time.time()-t0):.3f}s | MSE Teste: {xgb_mse:.4f}")

    # ================= 2. Instâncias HSAMA =================
    lr_base = 2e-3
    
    # A) Base
    model_a = HSAMA(in_features=8, out_features=1, num_nodes=10, k=4, max_hops=2)
    opt_a = optim.Adam(model_a.parameters(), lr=lr_base)
    
    # B) Replay Manual (O que usávamos antes)
    model_b = HSAMA(in_features=8, out_features=1, num_nodes=10, k=4, max_hops=2)
    opt_b = optim.Adam(model_b.parameters(), lr=lr_base)
    buffer_b = PrioritizedSurprisalBuffer(
        capacity=1500, threshold=FixedSurprisalThreshold(tau=0.15), sample_with_replacement=True
    )
    
    # C) MEGAZORD (HSAMAOnlineRuntime englobando tudo!)
    model_c = HSAMA(in_features=8, out_features=1, num_nodes=10, k=4, max_hops=2)
    megazord_config = HSAMARuntimeConfig(
        replay_config=SurprisalBufferConfig(
            capacity=1500, 
            threshold=AdaptiveQuantileThreshold(window_size=128, quantile=0.75), # Adaptável!
            replay_ratio=0.5,
            importance_weighting=True
        ),
        multi_scale_t0_config=MultiScaleT0Config(
            scales=(T0ScaleSpec(name="curto", stride=1, history_length=5), T0ScaleSpec(name="longo", stride=4, history_length=4)),
            aggregation="concat", append_scale_surprisal=True
        ),
        use_exploration=True,
        observe_mode="batch",   # 1 gradient step por batch (igual a A e B)
        optimizer_t1_kwargs={"lr": lr_base},
        optimizer_t0_model_kwargs={"lr": lr_base}
    )
    runtime_megazord = HSAMAOnlineRuntime(
        model=model_c, config=megazord_config, surprisal_estimator=EMASurprisalEstimator()
    )

    history_test  = {'A_Base': [], 'B_Replay': [], 'C_Megazord': []}
    criterion = nn.MSELoss()

    print("\n[Iniciando Treinamento CRONOLÓGICO para respeitar o Módulo Temporal do Megazord]")
    t_start = time.time()
    for ep in range(epochs):
        epoch_losses_c = []
        
        # Mantendo puramente cronológico (sem Shuffle) para o HistoryBuffer fazer sentido geográfico temporal
        for i in range(0, n_train, batch_size):
            bx, by = X_train[i:min(i + batch_size, n_train)], y_train[i:min(i + batch_size, n_train)]
            
            # --- Modelo A ---
            opt_a.zero_grad()
            out_a, _ = model_a(bx, raw_output=True, use_exploration=False)
            loss_a = criterion(out_a, by)
            loss_a.backward()
            opt_a.step()
            
            # --- Modelo B ---
            opt_b.zero_grad()
            out_b, _ = model_b(bx, raw_output=True, use_exploration=False)
            loss_b = criterion(out_b, by)
            for evt_idx in range(bx.size(0)):
                buffer_b.insert(x=bx[evt_idx:evt_idx+1], y=by[evt_idx:evt_idx+1], priority_surprisal=torch.abs(out_b[evt_idx] - by[evt_idx]).item(), raw_loss=loss_b.item(), event_id=i+evt_idx)
            loss_b.backward()
            opt_b.step()
            if len(buffer_b) > 16:
                sd = buffer_b.sample(8, importance_weighting=True)
                if sd is not None:
                    opt_b.zero_grad()
                    s_out, _ = model_b(sd["x"], raw_output=True)  # consistente com treino live
                    r_loss = (F.mse_loss(s_out, sd["y"], reduction='none') * sd["sample_weight"].unsqueeze(1)).mean()
                    r_loss.backward()
                    opt_b.step()

            # --- Modelo C (MEGAZORD) ---
            # Uma única chamada faz backprop em 16 items individualmente, atualiza históricos e injeta replays!
            results = runtime_megazord.observe(bx, by)
            if isinstance(results, list):
                epoch_losses_c.append(np.mean([r.total_loss.item() for r in results]))
            else:
                epoch_losses_c.append(results.total_loss.item())

        # Eval Test
        mse_a, _ = evaluate_test(model_a, X_test, y_test)
        mse_b, _ = evaluate_test(model_b, X_test, y_test)
        mse_c, _ = evaluate_test(model_c, X_test, y_test, runtime=runtime_megazord)
        history_test['A_Base'].append(mse_a)
        history_test['B_Replay'].append(mse_b)
        history_test['C_Megazord'].append(mse_c)
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f" Época {ep+1:02d} | Test MSE -> Base: {mse_a:.4f} | Replay: {mse_b:.4f} | MEGAZORD: {mse_c:.4f}")

    print(f"\n[Fim] Treino Cronológico Finalizado em {time.time()-t_start:.2f}s")
    print(f"Tamanho do Buffer interno do Megazord: {runtime_megazord.replay_buffer.capacity}")

    # ================= GERANDO VISUALIZAÇÕES =================
    _, preds_a = evaluate_test(model_a, X_test, y_test)
    _, preds_c = evaluate_test(model_c, X_test, y_test, runtime=runtime_megazord)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(history_test['A_Base'], marker='', label="HSAMA Base (Amnésia Temporal)")
    ax1.plot(history_test['B_Replay'], marker='', label="HSAMA Replay Manual")
    ax1.plot(history_test['C_Megazord'], marker='', label="HSAMA MEGAZORD Completo", linewidth=2, color='green')
    ax1.axhline(y=xgb_mse, color='r', linestyle='--', label=f"XGBoost Offline ({xgb_mse:.4f})")
    ax1.set_title("Corrida de Convergência Online (Cronológica vs XGBoost Offline)")
    ax1.set_ylabel("Test MSE"); ax1.legend(); ax1.grid(True, alpha=0.5)
    
    lim = 120
    ax2.plot(y_test.numpy()[:lim], color='black', linewidth=3, label="Valor Real (Ground Truth)")
    ax2.plot(xgb_preds[:lim], color='red', linestyle='--', label="XGBoost Previsão")
    ax2.plot(preds_c[:lim], color='green', alpha=0.8, label="HSAMA Megazord Previsão")
    ax2.set_title("Previsão Visual do Megazord - MultiScale + Replay"); ax2.legend(); ax2.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)

if __name__ == "__main__":
    main()
