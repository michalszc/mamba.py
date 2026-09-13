import copy

import pytest
import torch

from benchmarks.real_text import (
    ASSETS, audit, corpus_splits, difference, make_batches, prepare_assets, snapshot,
)
from mambapy.lm import LM
from mambapy.mamba import MambaConfig


def test_text_batches_are_shifted_reproducible_and_stay_in_partition():
    text = 'abcde' * 90 + 'VWXYZ' * 10
    train, validation, alphabet = corpus_splits(text, 7)
    assert len(train) == 450 and len(validation) == 50
    assert 0 not in train and 0 not in validation  # padding never consumes a real character
    assert not set(train.tolist()).intersection(validation.tolist())
    x, y, starts = make_batches(validation, 8, 3, 7, 41)
    x2, y2, starts2 = make_batches(validation, 8, 3, 7, 41)
    assert torch.equal(x, x2) and torch.equal(y, y2) and torch.equal(starts, starts2)
    assert torch.equal(x[..., 1:], y[..., :-1])
    assert torch.equal(y[..., -1], validation[starts + 7])
    assert starts.max() + 7 < len(validation)
    assert set(alphabet) == set(text)


def test_assets_require_explicit_download_and_reject_modified_source(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        prepare_assets(tmp_path)
    (tmp_path / next(iter(ASSETS))).write_text('changed upstream source')
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare_assets(tmp_path)


def test_audit_catches_missing_and_wrong_gradients_despite_identical_logits():
    torch.manual_seed(14)
    reference = LM(MambaConfig(16, 1, backend='torch'), 12)
    initial = snapshot(reference)
    models = {name: copy.deepcopy(reference) for name in ('correct', 'missing', 'wrong')}
    models['missing'].mamba.layers[0].mixer.D.requires_grad_(False)
    models['wrong'].lm_head.weight.register_hook(lambda grad: grad * 100)
    x = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
    y = torch.tensor([[2, 3, 4, 5], [4, 5, 6, 7]])
    results = audit(models, reference, initial, x, y, .002)
    assert results['correct']['pass']
    assert all(row['logits']['pass'] for row in results.values())
    assert not results['missing']['pass'] and not results['wrong']['pass']
    assert results['missing']['missing_gradients'] == ['mamba.layers.0.mixer.D']
    assert not results['wrong']['gradients']['lm_head.weight']['pass']
    assert all(check['pass'] for check in results['wrong']['adamw_same_gradient_updates'].values())
    # A correctness probe must not secretly advance subsequent training weights.
    for model in models.values():
        assert all(torch.equal(value, initial[key]) for key, value in snapshot(model).items())
    assert all(torch.equal(value, initial[key]) for key, value in snapshot(reference).items())


def test_nonfinite_values_cannot_pass_correctness():
    result = difference(torch.tensor([float('nan')]), torch.ones(1), 'gradients')
    assert not result['pass'] and not result['finite']
    assert result['max_abs_error'] is None
