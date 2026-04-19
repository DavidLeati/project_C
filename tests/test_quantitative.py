import time
import math
import torch
import pytest

from src.models.hsama import HSAMA
from src.runtime.replay import PrioritizedSurprisalBuffer, FixedSurprisalThreshold

@pytest.fixture
def base_hsama():
    """Initializes a base HSAMA model for quantitative testing."""
    torch.manual_seed(42)
    return HSAMA(
        in_features=8,
        out_features=1,
        num_nodes=10,
        k=4,
        state_dim=4,
        context_dim=4,
        max_hops=3,
        num_sectors=2
    )

def test_comparative_hierarchical_surprisal_response(base_hsama):
    """
    Quantitatively tests how HSAMA's meta-architecture (T0 policy) 
    changes the edge DNA configuration dynamically depending on surprisal values.
    """
    x = torch.randn(4, 8) # Batch=4, features=8
    
    # 1. Low Surprisal Baseline
    policy_low = base_hsama.build_policy(x, surprisal=0.0, use_exploration=False)
    
    # 2. High Surprisal Response
    policy_high = base_hsama.build_policy(x, surprisal=10.0, use_exploration=False)
    
    # Quantitative checks
    dna_low_std = policy_low.edge_dnas.std().item()
    dna_high_std = policy_high.edge_dnas.std().item()
    
    # Under high surprisal, scale shifts based on temperature
    assert policy_low.temperature.mean().item() < policy_high.temperature.mean().item(), \
        "Temperature should strictly increase under high surprisal."
    assert policy_low.scale.mean().item() < policy_high.scale.mean().item(), \
        "DNA Edge Scale should increase under high surprisal."
        
    # Execute and ensure structural changes yield distinctly scaled graph messages
    out_low, proj_low = base_hsama.execute_policy(x, policy=policy_low, raw_output=True)
    out_high, proj_high = base_hsama.execute_policy(x, policy=policy_high, raw_output=True)
    
    low_norm = torch.norm(out_low).item()
    high_norm = torch.norm(out_high).item()
    
    # Results should be distinct because the hypernetwork rewires DNA scales
    assert low_norm != high_norm, "Execution must differ dynamically via meta-weights."

def test_comparative_runtime_hops_reduction(base_hsama):
    """
    Compares the execution time and structural outputs between 
    a fully deep run (max_hops) vs an analytically reduced graph execution
    using `surprisal_variance` cutoff.
    """
    x = torch.randn(32, 8)
    
    # Full execution
    t0 = time.perf_counter()
    out_full, _ = base_hsama(x, surprisal_variance=10.0, surprisal=0.0) # > delta=1e-3, full hops
    t_full = time.perf_counter() - t0
    
    # Reduced topology (cold variance triggers 1 hop instead of max_hops)
    t1 = time.perf_counter()
    out_reduced, _ = base_hsama(x, surprisal_variance=0.0001, surprisal=0.0) # < delta=1e-3, reduced
    t_reduced = time.perf_counter() - t1
    
    # Reduced should theoretically take less time and produce different representations
    diff_norm = torch.norm(out_full - out_reduced).item()
    assert diff_norm > 1e-4, "Full hops vs Reduced hops must yield quantifiably distinct graphs."
    
def test_quantitative_replay_buffer_weighting():
    """
    Tests and compares the Priority Surprisal Buffer quantitatively:
    Checking importance weighting variance distribution vs uniform.
    """
    buffer = PrioritizedSurprisalBuffer(
        capacity=100,
        threshold=FixedSurprisalThreshold(tau=0.5), # Only accept surprisal > 0.5
        sample_with_replacement=True
    )
    
    # Insert multiple simulated training events
    for i in range(50):
        buffer.insert(
            x=torch.ones(1, 4),
            y=torch.zeros(1),
            priority_surprisal=0.6 + (i * 0.1), # Increasing priority
            raw_loss=0.1,
            event_id=i
        )
        
    # Ensure threshold correctly blocked elements below tau (if any were inserted)
    buffer.insert(
        x=torch.ones(1, 4),
        y=torch.zeros(1),
        priority_surprisal=0.1, # Below 0.5 tau cutoff
        raw_loss=0.1,
        event_id=999
    )
    
    assert len(buffer) == 50, "Buffer should have rejected the low surprisal."
    
    # Compare Sample weighting logic
    sample_uniform = buffer.sample(32, importance_weighting=False)
    sample_weighted = buffer.sample(32, importance_weighting=True)
    
    std_uniform = sample_uniform["sample_weight"].std().item()
    std_weighted = sample_weighted["sample_weight"].std().item()
    
    # Quantitatively: Importance weighting introduces variance across sample probabilities
    assert math.isclose(std_uniform, 0.0, abs_tol=1e-5), "Uniform weights should have zero variance."
    assert std_weighted > 0.0, "Priority-weighted samples must distribute variance structurally."
