"""From-scratch ECG200 classification with this fork's Mamba-1 backend only."""
import argparse
import copy
import io
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Optional
import urllib.request
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.real_text import difference, sha256, snapshot, write_report
from mambapy.mamba import MambaBlock as MambaMixer, MambaConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_URL = 'https://timeseriesclassification.com/aeon-toolkit/ECG200.zip'
ZIP_SHA256 = '755937ea37849346ea20033666b19d17345bb039971e8fa33458b16b62c09380'
MEMBER_HASHES = {
    'ECG200_TRAIN.txt': 'a81521c4fdcded0bbb47334136bc4a6bf7dd8cdf8c8d5877b36d207146f55151',
    'ECG200_TEST.txt': '907c243f6e92234fc3a249448240e95db292835217457a2bc4cbe3d73a02902d',
}


class MambaBlock(nn.Module):
    """User's LayerNorm -> Mamba -> Dropout -> residual, [B,T,D] -> [B,T,D]."""
    def __init__(self, hidden_dim=256, d_state=64, d_conv=4, expand=2,
                 dropout=.1, device: Optional[torch.device] = None, backend='mps'):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # MambaMixer is the bare mixer in this fork. Using the top-level Mamba
        # would add another normalization and residual connection internally.
        self.mamba = MambaMixer(MambaConfig(hidden_dim, 1, d_state=d_state,
                                           d_conv=d_conv, expand_factor=expand, backend=backend))
        self.dropout = nn.Dropout(dropout)
        if device is not None:
            self.to(device)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class ECGClassifier(nn.Module):
    """Scalar input projection, stacked pure Mamba blocks, mean pool, MLP head."""
    def __init__(self, hidden_dim=256, d_state=64, d_conv=4, expand=2, dropout=.1,
                 num_layers=2, device: Optional[torch.device] = None, backend='mps'):
        super().__init__()
        self.input_projection = nn.Linear(1, hidden_dim)
        self.mamba_blocks = nn.Sequential(*[
            MambaBlock(hidden_dim, d_state, d_conv, expand, dropout, backend=backend)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 2))
        if device is not None:
            self.to(device)

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != 1 or x.size(1) == 0:
            raise ValueError('Expected a nonempty sequence shaped [batch, time, 1]')
        x = self.input_projection(x)
        x = self.mamba_blocks(x)
        return self.head(x.mean(dim=1))


def read_split(data):
    rows = np.loadtxt(io.BytesIO(data), dtype=np.float32)
    if rows.shape != (100, 97) or not np.isfinite(rows).all():
        raise ValueError('ECG200 must contain 100 finite rows with one label and 96 values')
    labels = rows[:, 0]
    if set(labels.tolist()) != {-1., 1.}:
        raise ValueError('Expected ECG200 class labels -1 and 1')
    return torch.from_numpy(rows[:, 1:].copy()).unsqueeze(-1), torch.from_numpy((labels > 0).astype(np.int64))


def load_ecg200(cache, download=False):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'ECG200.zip'
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'{path} missing; use --download once')
        data = urllib.request.urlopen(DATA_URL, timeout=60).read()
        if sha256(data) != ZIP_SHA256:
            raise ValueError('ECG200 archive checksum mismatch')
        path.write_bytes(data)
    data = path.read_bytes()
    if sha256(data) != ZIP_SHA256:
        raise ValueError('Cached ECG200 archive checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        splits = []
        for name, digest in MEMBER_HASHES.items():
            member = archive.read(name)  # Read named members; never extract arbitrary paths.
            if sha256(member) != digest:
                raise ValueError(f'ECG200 member checksum mismatch: {name}')
            splits.append(read_split(member))
    return splits


def stratified_split(labels, seed=42, validation_fraction=.2):
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = max(1, round(len(indices) * validation_fraction))
        if count >= len(indices):
            raise ValueError('Each class needs examples in both training and validation')
        validation.append(indices[:count])
        train.append(indices[count:])
    return torch.cat(train), torch.cat(validation)


def fit_normalization(training):
    return training.mean(dim=(0, 1), keepdim=True), training.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)


def classification_metrics(logits, targets):
    logits, targets = logits.detach().cpu(), targets.detach().cpu()
    predicted = logits.argmax(-1)
    confusion = torch.bincount(targets * 2 + predicted, minlength=4).reshape(2, 2)
    recall = confusion.diag() / confusion.sum(1).clamp_min(1)
    f1 = 2 * confusion.diag() / (confusion.sum(0) + confusion.sum(1)).clamp_min(1)
    return {'loss': float(F.cross_entropy(logits, targets)),
            'accuracy': float((predicted == targets).float().mean()),
            'balanced_accuracy': float(recall.mean()), 'macro_f1': float(f1.mean()),
            'confusion_matrix_true_rows': confusion.tolist()}


@torch.no_grad()
def evaluate(model, x, y, batch_size):
    model.eval()
    logits = torch.cat([model(batch) for batch in x.split(batch_size)])
    return classification_metrics(logits, y), logits.detach().cpu()


def audit_gradients(model, x, y):
    """Same classifier/weights, dropout off; this fork's CPU path is the oracle."""
    reference = copy.deepcopy(model).cpu().eval()
    for block in reference.mamba_blocks:
        block.mamba.config.backend = 'torch'
    modes = [(module, module.training) for module in model.modules()]
    outputs = []
    try:
        for candidate, device in ((reference, 'cpu'), (model, x.device)):
            candidate.eval()
            candidate.zero_grad(set_to_none=True)
            inputs = x.detach().to(device).requires_grad_(True)
            logits = candidate(inputs)
            F.cross_entropy(logits, y.to(device)).backward()
            grads = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                     for name, p in candidate.named_parameters()}
            grads['<input>'] = None if inputs.grad is None else inputs.grad.detach().cpu()
            outputs.append((logits.detach().cpu(), grads))
        expected, actual = outputs
        forward = difference(actual[0], expected[0], 'logits')
        gradients = {name: difference(gradient, expected[1][name], 'gradients')
                     for name, gradient in actual[1].items()}
        return {'pass': forward['pass'] and all(item['pass'] for item in gradients.values()),
                'logits': forward, 'gradients': gradients}
    finally:
        model.zero_grad(set_to_none=True)
        for module, training in modes:
            module.training = training


def update(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.detach()


def check_gradients(model):
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    if missing:
        raise RuntimeError(f'Missing training gradients: {missing}')
    finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all()
    if not bool(finite):
        raise RuntimeError('Nonfinite training gradients')


def run_seed(args, seed, training, validation, test, checkpoints):
    torch.manual_seed(seed)
    model = ECGClassifier(args.hidden_dim, args.state, args.conv, args.expand,
                          args.dropout, args.layers, device=torch.device('mps'))
    initial = snapshot(model)
    train_x, train_y = training
    val_x, val_y = validation
    row = {'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
           'epochs': [], 'samples_update_ms': []}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    torch.mps.synchronize()
    start = time.perf_counter()
    update(model, optimizer, train_x[:args.batch], train_y[:args.batch])
    torch.mps.synchronize()
    row['first_update_ms'] = (time.perf_counter() - start) * 1000
    for _ in range(3):
        update(model, optimizer, train_x[:args.batch], train_y[:args.batch])
    torch.mps.synchronize()
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    # Warmup must not advance either training weights, AdamW state, or dropout RNG.
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    row['initial_audit'] = audit_gradients(model, train_x[:2], train_y[:2])
    if not row['initial_audit']['pass']:
        return {**row, 'status': 'initial_correctness_failure'}
    row['initial_validation'], _ = evaluate(model, val_x, val_y, args.batch)
    generator = torch.Generator().manual_seed(seed + 1000)
    best_loss, best_epoch, best_weights = row['initial_validation']['loss'], 0, initial
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_x), generator=generator).to('mps')
        weighted_loss, epoch_times = 0., []
        for indices in order.split(args.batch):
            x, y = train_x[indices], train_y[indices]
            torch.mps.synchronize()
            start = time.perf_counter()
            loss = update(model, optimizer, x, y)
            torch.mps.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            # Numerical checks and host logging are outside measured update time.
            check_gradients(model)
            loss_value = float(loss.cpu())
            if not math.isfinite(loss_value):
                raise RuntimeError('Nonfinite training loss')
            weighted_loss += loss_value * len(indices)
            epoch_times.append(elapsed)
        metrics, _ = evaluate(model, val_x, val_y, args.batch)
        row['samples_update_ms'].extend(epoch_times)
        row['epochs'].append({'epoch': epoch, 'training_loss': weighted_loss / len(train_x),
                              'validation': metrics, 'training_update_total_ms': sum(epoch_times)})
        if metrics['loss'] < best_loss:
            best_loss, best_epoch, best_weights = metrics['loss'], epoch, snapshot(model)
        if epoch % 10 == 0 or epoch == 1:
            print(f'seed={seed} epoch={epoch}: train loss={weighted_loss / len(train_x):.4f}, '
                  f'validation loss={metrics["loss"]:.4f}, accuracy={metrics["accuracy"]:.1%}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_weights)
    row['best_epoch'] = best_epoch
    row['final_audit'] = audit_gradients(model, val_x[:2], val_y[:2])
    row['training'], _ = evaluate(model, train_x, train_y, args.batch)
    row['validation'], _ = evaluate(model, val_x, val_y, args.batch)
    # The official test labels never enter optimization, early stopping or audits.
    row['test'], logits = evaluate(model, *test, args.batch)
    row['test_predictions'] = logits.argmax(-1).tolist()
    row['test_probabilities'] = logits.softmax(-1).tolist()
    row['unchanged_parameters'] = [name for name, value in best_weights.items() if torch.equal(value, initial[name])]
    times = row['samples_update_ms']
    row['median_update_ms'] = statistics.median(times)
    row['p95_update_ms'] = sorted(times)[min(len(times) - 1, int(.95 * len(times)))]
    row['median_epoch_training_ms'] = statistics.median(e['training_update_total_ms'] for e in row['epochs'])
    row['status'] = 'pass' if row['final_audit']['pass'] and not row['unchanged_parameters'] else 'correctness_failure'
    checkpoint = checkpoints / f'ecg200-seed-{seed}.pt'
    torch.save({'state_dict': best_weights, 'seed': seed, 'best_epoch': best_epoch,
                'model_config': {'hidden_dim': args.hidden_dim, 'd_state': args.state, 'd_conv': args.conv,
                                 'expand': args.expand, 'dropout': args.dropout, 'num_layers': args.layers},
                'normalization': args.normalization, 'label_mapping': {-1: 0, 1: 1}}, checkpoint)
    # Validate serialization on the exact classifier and official test inputs.
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(loaded['state_dict'], strict=True)
    _, restored_logits = evaluate(model, *test, args.batch)
    torch.testing.assert_close(restored_logits, logits, rtol=1e-4, atol=1e-5)
    row['checkpoint'] = str(checkpoint)
    print(f'seed={seed}: selected epoch={best_epoch}, test accuracy={row["test"]["accuracy"]:.1%}, '
          f'balanced accuracy={row["test"]["balanced_accuracy"]:.1%}, '
          f'update={row["median_update_ms"]:.2f} ms, correctness={row["status"]}', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--cache', type=Path, default=ROOT / 'benchmark-results/ecg200-data')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-results/ecg200.json')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--state', type=int, default=64)
    parser.add_argument('--conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    args = parser.parse_args()
    for key in ('hidden_dim', 'state', 'conv', 'expand', 'layers', 'batch', 'epochs', 'patience'):
        if getattr(args, key) <= 0:
            parser.error(f'{key} must be positive')
    if args.hidden_dim < 2 or args.state > 128 or not 0 <= args.dropout < 1:
        parser.error('Require hidden_dim >=2, state <=128, and dropout in [0,1)')
    if not math.isfinite(args.lr) or args.lr <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Learning rate must be positive/finite and seeds unique')
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        parser.error('Real MPS with PYTORCH_ENABLE_MPS_FALLBACK=0 is required')
    torch.mps.set_per_process_memory_fraction(min(1., 16 * 2**30 / torch.mps.recommended_max_memory()))
    (x, y), (test_x, test_y) = load_ecg200(args.cache, args.download)
    train_indices, val_indices = stratified_split(y)
    mean, std = fit_normalization(x[train_indices])
    args.normalization = {'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()}
    splits = [((x[train_indices] - mean) / std, y[train_indices]),
              ((x[val_indices] - mean) / std, y[val_indices]), ((test_x - mean) / std, test_y)]
    splits = [tuple(t.to('mps') for t in split) for split in splits]
    checkpoints = args.output.parent / (args.output.stem + '-checkpoints')
    checkpoints.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'environment': {'torch': torch.__version__, 'python': platform.python_version(),
                              'platform': platform.platform(), 'dtype': 'float32', 'device': 'mps', 'cpu_fallback': False},
              'source': {'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                         'benchmark_sha256': sha256(Path(__file__).read_bytes()),
                         'backend_hashes': {str(p.relative_to(ROOT)): sha256(p.read_bytes())
                              for p in sorted((ROOT / 'mambapy').rglob('*')) if p.suffix in ('.py', '.metal')}},
              'dataset': {'url': DATA_URL, 'archive_sha256': ZIP_SHA256, 'member_sha256': MEMBER_HASHES,
                          'official_train': 100, 'official_test': 100, 'length': 96, 'label_mapping': {-1: 0, 1: 1},
                          'training_indices': train_indices.tolist(), 'validation_indices': val_indices.tolist(),
                          'split_seed': 42, 'test_labels': test_y.tolist(),
                          'training_majority_class_test_accuracy': float((test_y == y[train_indices].bincount().argmax()).float().mean())},
              'protocol': 'From scratch; user residual block, mean pooling, MLP head. Fixed training-only validation split and normalization. Checkpoint by validation CE with early stopping; official test evaluated after selection. CPU reference audit is this fork only, dropout disabled. Timings include synchronized forward/loss/backward/AdamW, exclude data indexing, metrics, gradient checks, audits, warmup, and validation.',
              'tolerances': {'logits': {'rtol': 1e-4, 'atol': 1e-5}, 'gradients': {'rtol': 1e-3, 'atol': 1e-7}},
              'runs': []}
    write_report(args.output, report)
    for seed in args.seeds:
        row = run_seed(args, seed, *splits, checkpoints)
        report['runs'].append(row)
        write_report(args.output, report)
        if row['status'] != 'pass':
            report['status'] = 'correctness_failure'
            write_report(args.output, report)
            return 2
    accuracies = [row['test']['accuracy'] for row in report['runs']]
    report['summary'] = {'mean_test_accuracy': statistics.mean(accuracies),
                         'sample_std_test_accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else None,
                         'mean_balanced_accuracy': statistics.mean(row['test']['balanced_accuracy'] for row in report['runs'])}
    report['status'] = 'pass'
    write_report(args.output, report)
    print(f'All checks passed. Mean test accuracy={report["summary"]["mean_test_accuracy"]:.1%}. Report: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
