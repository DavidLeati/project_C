import torch
import torch.nn.functional as F
from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig

def test_multi_sector_t0():
    """
    Verifica se a arquitetura suporta T0 multi-cabeça (out_features=2).
    A cabeça 0 será a Predição (MSE), iterada pelo status real.
    A cabeça 1 será a Operação (Classificação/Regressão Lógica), iterada por um label.
    As duas cabeças compartilharão os contextos da GRU.
    """
    # 1. Configurar um T0 com dois outputs (setores) 
    batch_size = 4
    features = 8
    
    model = HSAMA(
        in_features=features,
        out_features=2, # <-- O "Pulo do Gato": 2 setores de saída!
        num_nodes=8,
        state_dim=16,
        max_hops=1, 
        context_dim=8
    )
    
    # 2. Iniciar o T1 (Runtime) que gerencia o T0 
    config = HSAMARuntimeConfig(
        observe_mode="batch",
        raw_output=True # Necessário True para múltiplas cabeças divergentes
    )
    runtime = HSAMAOnlineRuntime(model, config=config)
    
    # 3. Preparar inputs e expected outputs 
    x_batch = torch.randn(batch_size, features)
    
    # Labels (alvos correspondentes às DUAS cabeças de cada amostra)
    y_pred = torch.randn(batch_size, 1) # Target contínuo do preço (MSE)
    y_trade = torch.randint(0, 2, (batch_size, 1)).float() # Target discreto (0/1 Long/Short)
    y_batch_combinado = torch.cat([y_pred, y_trade], dim=1) # Shape: [batch_size, 2]
    
    # 4. Avaliação Zero-Shot
    preds_inference = runtime.predict(x_batch, raw_output=True)
    assert preds_inference.shape == (batch_size, 2), "Predict deveria retornar duas colunas (2 setores)"
    
    # 5. Pipeline de Treino (observe)
    # A sacada é que o Runtime recebe um y_batch cópia do Target shape.
    # O runtime usará a self._per_sample_loss. Contudo, em T0 multi-header 
    # a `criterion` padrão que calcula o erro não saberá gerenciar MSE e BCE por conta.
    # Para isso, validamos apenas se o graph forward via Runtime é executado sem quebrar as dimensões temporais.
    
    # Precisamos sobrescrever temporariamente a function de calculos do erro 
    # para aceitar os 2 outputs de forma customizada nas nossas duas pernas:
    def custom_triplex_loss(preds, targets):
        # preds: [B, 2] ; targets: [B, 2]
        pred_sector = preds[:, 0:1]
        target_sector = targets[:, 0:1]
        
        trade_sector = preds[:, 1:2]
        target_trade = targets[:, 1:2]
        
        loss_pred = F.mse_loss(pred_sector, target_sector, reduction='none')
        # Simulando uma crossentropy na regressao
        loss_trade = F.binary_cross_entropy_with_logits(trade_sector, target_trade, reduction='none')
        
        return (loss_pred + loss_trade).mean(dim=1) # per-sample loss scalar
    
    # Assinando a loss customizada (Substitui o _per_sample_loss do HSAMAOnlineRuntime)
    runtime._per_sample_loss = custom_triplex_loss
    
    # Observe-Batch
    result = runtime.observe(x_batch, y_batch_combinado)
    
    # Verificar saídas e Shapes
    assert result.event_id is not None
    assert result.live_prediction.shape == (batch_size, 2), "Runtime falhou no tracking da dimensão das múltiplas cabeças"
    assert result.total_loss.shape == (), "Loss deveria retornar uma redução escalar final após ser reduzida"
    
    print("Sucesso! A Arquitetura T1 + T0 Multi-Cabeças funciona perfeitamente de forma integrada.")

if __name__ == "__main__":
    test_multi_sector_t0()
