import torch
import pytest

from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.replay import SurprisalBufferConfig, AdaptiveQuantileThreshold
from src.runtime.surprisal import EMASurprisalEstimator
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec

def test_ultimate_runtime_megazord():
    """
    Este teste prova a capacidade de acoplar simultaneamente TODAS
    as peças do ecossistema Runtime (MultiScale, Replay, Surprisal, Temporal)
    na classe orquestradora única (HSAMAOnlineRuntime).
    """
    
    # 1. Instanciamos a casca fundamental de matemática do grafo vazio:
    base_model = HSAMA(
        in_features=8, out_features=1, num_nodes=10, k=4, max_hops=3
    )

    # 2. Configurações da Memória (Replay e Threshold Flexível de Surpresa)
    config_replay = SurprisalBufferConfig(
        capacity=1000,
        threshold=AdaptiveQuantileThreshold(window_size=128, quantile=0.8), # Adapta a surpresa ao ruído natural!
        replay_ratio=0.5, # Ex: puxar 50% de eventos passados a cada amostra nova
        importance_weighting=True
    )

    # 3. Configurações da Visão Multi-Escala "T0" e do Relógio Histórico
    config_multiscale = MultiScaleT0Config(
        scales=(
            # Escala curta/agressiva: olha 5 frames seguidos
            T0ScaleSpec(name="curto", stride=1, history_length=5),
            # Escala macro/tendência: olha blocos pulando a cada 4 frames
            T0ScaleSpec(name="tendencia", stride=4, history_length=4)
        ),
        aggregation="concat",
        append_scale_surprisal=True
    )

    # 4. A Grande Configuração do Runtime acoplando tudo...
    megazord_config = HSAMARuntimeConfig(
        replay_config=config_replay,
        multi_scale_t0_config=config_multiscale,
        use_exploration=True, # Ativa viés estocástico para quebra de simetria ("Exploration")
        optimizer_t1_kwargs={"lr": 0.01},
        optimizer_t0_model_kwargs={"lr": 0.005}
    )

    # 5. E finalmente instanciamos o ORQUESTRADOR GERAL passando as ferramentas!
    runtime = HSAMAOnlineRuntime(
        model=base_model,
        config=megazord_config,
        surprisal_estimator=EMASurprisalEstimator(beta_mu=0.9, beta_var=0.9)
    )

    # --- SIMULANDO O STREAMING ONLINE (O TESTE) ---
    # Geramos uma sequência cronológica de eventos (20 eventos no tempo, com 8 features)
    features_chronological = torch.randn(20, 8)
    targets_chronological = torch.randn(20, 1)

    # Em vez de ter que chamar manualmente o Replay, História temporal, etc,
    # O runtime cuida disso tudo por si só nas "observações"
    for i in range(20):
        # Apenas alimente-o com 1 frame de cada vez simulando Tick de mercado real
        bx = features_chronological[i:i+1]
        by = targets_chronological[i:i+1]
        
        # .observe() cuida magicamente de TODAS as 5 funcionalidades vistas!
        # Ele avalia a EMA_Surprisal, alimenta histórico, ajusta escalas temporais,
        # faz backprop em múltiplos otimizadores, aciona replay... e devolve estatísticas.
        resultado = runtime.observe(bx, by)

    # Assertivo: se o cdigo aguentou os frames cronolgicos e o histrico tem pelo menos o limite mximo calculado mvel de 13 eventos
    assert runtime.global_step == 20
    assert runtime.history_buffer is not None
    required_len = runtime.multi_scale_builder.required_history()
    assert len(runtime.history_buffer) == required_len
    assert runtime.replay_buffer.capacity == 1000

    print("\n[OK] Modelo Megazord executou o fluxo multi-escala temporal com maestria!")
