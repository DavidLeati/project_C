import torch

from trade.dataset import CryptoDataLoader
from trade.loss import TriplexTradingLoss, position_from_logits


def test_multi_timeframe_targets_are_native_not_copied_15m():
    loader = CryptoDataLoader(data_dir="trade/data")
    data = loader.load_multi_timeframe_sol(train_ratio=0.7, max_samples=1600)

    y_15m = torch.cat([data["15m"][1], data["15m"][3]], dim=0)
    for tf in ("1h", "4h", "1d"):
        y_tf = torch.cat([data[tf][1], data[tf][3]], dim=0)
        assert y_tf.shape == y_15m.shape
        assert not torch.allclose(y_tf, y_15m)


def test_triplex_loss_charges_batch_boundary_cost_from_previous_position():
    loss = TriplexTradingLoss(
        cost_bps=0.01,
        trade_weight=1.0,
        return_scale=1.0,
        bias_weight=0.0,
        stagnation_weight=0.0,
        position_deadzone=0.0,
    )
    preds = torch.tensor([[0.0], [0.0]])
    targets = torch.zeros(2, 1)

    loss.set_previous_position(0.0)
    no_boundary_cost = loss(preds, targets)

    loss.set_previous_position(1.0)
    with_boundary_cost = loss(preds, targets)

    assert with_boundary_cost[0] > no_boundary_cost[0]
    assert torch.isclose(with_boundary_cost[1], no_boundary_cost[1])


def test_position_from_logits_deadzone_zeroes_small_positions():
    logits = torch.tensor([[-0.02], [0.0], [0.02], [2.0]])
    positions = position_from_logits(logits, deadzone=0.05)

    assert positions[0].item() == 0.0
    assert positions[1].item() == 0.0
    assert positions[2].item() == 0.0
    assert positions[3].item() > 0.0
