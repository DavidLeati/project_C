import pandas as pd
import numpy as np

df = pd.read_csv('artifacts/SOL_walkforward_trades_oos.csv')
pos = df.position.values

# Delta de posição entre candles
delta = np.abs(np.diff(pos))

print("=== DELTA DE POSIÇÃO (|pos[t] - pos[t-1]|) ===")
print(f"Delta médio:    {delta.mean():.4f}")
print(f"Delta mediano:  {np.median(delta):.4f}")
print(f"Delta p90:      {np.percentile(delta, 90):.4f}")
print(f"Delta p99:      {np.percentile(delta, 99):.4f}")
print(f"Delta max:      {delta.max():.4f}")
print()

# Distribuição dos deltas
bins = [0, 0.5, 1, 2, 5, 10, 20, 40]
print("=== DISTRIBUIÇÃO DOS DELTAS ===")
for i in range(len(bins)-1):
    mask = (delta >= bins[i]) & (delta < bins[i+1])
    pct = mask.mean() * 100
    cost_share = delta[mask].sum() / delta.sum() * 100 if delta.sum() > 0 else 0
    print(f"  [{bins[i]:5.1f}, {bins[i+1]:5.1f}):  {pct:5.1f}% dos candles  |  {cost_share:5.1f}% do custo total")
print()

# Custo gerado por tipo de mudança
# Mudança de direção vs ajuste de tamanho
direction = np.sign(pos)
dir_change = direction[1:] != direction[:-1]
same_dir = ~dir_change

cost_flip = delta[dir_change].sum()
cost_resize = delta[same_dir].sum()
cost_total = delta.sum()

print("=== ORIGEM DO CUSTO ===")
print(f"Flips de direção:     {dir_change.sum():5d} eventos | Custo: {cost_flip:.2f} ({cost_flip/cost_total*100:.1f}%)")
print(f"Ajuste de tamanho:    {same_dir.sum():5d} eventos | Custo: {cost_resize:.2f} ({cost_resize/cost_total*100:.1f}%)")
print()

# Dentro dos ajustes de tamanho (mesma direção), qual o delta típico?
print("=== DELTAS SEM FLIP (mesma direção) ===")
resize_deltas = delta[same_dir]
print(f"Delta médio (resize): {resize_deltas.mean():.4f}")
print(f"Delta mediano (resize): {np.median(resize_deltas):.4f}")
print(f"Resize > 5:  {(resize_deltas > 5).mean()*100:.1f}%")
print(f"Resize > 10: {(resize_deltas > 10).mean()*100:.1f}%")
print(f"Resize > 1:  {(resize_deltas > 1).mean()*100:.1f}%")
print()

# Exemplo: 20 candles consecutivos
print("=== AMOSTRA DE 20 CANDLES (step 5000-5020) ===")
sample = df.iloc[5000:5020]
for _, row in sample.iterrows():
    print(f"  step={row.step:5.0f}  pos={row.position:+8.4f}  action={row.action:5s}  net={row.net_pnl:+.6f}")
