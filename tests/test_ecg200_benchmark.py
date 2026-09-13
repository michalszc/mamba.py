import os

import pytest
import torch
from torch import nn

from benchmarks.ecg200 import (
    ECGClassifier, MambaBlock, audit_gradients, classification_metrics,
    fit_normalization, stratified_split,
)
from mambapy.mamba import MambaBlock as MambaMixer


def test_residual_wrapper_uses_exactly_one_bare_mixer():
    block = MambaBlock(16, d_state=64, dropout=0., backend='torch')
    assert isinstance(block.mamba, MambaMixer)
    assert isinstance(block.norm, nn.LayerNorm)
    block.mamba = nn.Identity()
    x = torch.randn(2, 7, 16)
    torch.testing.assert_close(block(x), x + block.norm(x))


def test_stratification_and_normalization_use_only_training_partition():
    labels = torch.tensor([0] * 31 + [1] * 69)
    training, validation = stratified_split(labels)
    assert len(training) == 80 and len(validation) == 20
    assert set(training.tolist()).isdisjoint(validation.tolist())
    assert sorted(torch.cat((training, validation)).tolist()) == list(range(100))
    assert labels[validation].bincount().tolist() == [6, 14]
    assert torch.equal(training, stratified_split(labels)[0])
    x = torch.randn(100, 96, 1)
    mean, std = fit_normalization(x[training])
    x[validation] = 1e6
    mean2, std2 = fit_normalization(x[training])
    assert torch.equal(mean, mean2) and torch.equal(std, std2)
    normalized = (x[training] - mean) / std
    torch.testing.assert_close(normalized.mean(), torch.tensor(0.), atol=1e-6, rtol=0.)
    torch.testing.assert_close(normalized.std(unbiased=False), torch.tensor(1.))


def test_metrics_include_class_imbalance_and_confusion_matrix():
    logits = torch.tensor([[2., 0.]] * 4)
    metrics = classification_metrics(logits, torch.tensor([0, 0, 0, 1]))
    assert metrics['accuracy'] == .75
    assert metrics['balanced_accuracy'] == .5
    assert metrics['confusion_matrix_true_rows'] == [[3, 0], [1, 0]]
    assert metrics['macro_f1'] == pytest.approx(3 / 7)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_classifier_all_gradients_and_audit_restores_modes(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        if os.environ.get('MAMBAPY_REQUIRE_MPS') == '1':
            pytest.fail('Real MPS required')
        pytest.skip('MPS unavailable')
    torch.manual_seed(11)
    model = ECGClassifier(16, d_state=64, num_layers=2, dropout=.1,
                          backend='torch' if device == 'cpu' else 'mps').to(device).train()
    model.head.eval()  # Mixed module modes must survive the dropout-free audit.
    modes = [module.training for module in model.modules()]
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.randn(2, 96, 1, device=device)
    y = torch.tensor([0, 1], device=device)
    assert model(x).shape == (2, 2)
    result = audit_gradients(model, x, y)
    assert result['pass'], result
    assert all('missing' not in row for row in result['gradients'].values())
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
