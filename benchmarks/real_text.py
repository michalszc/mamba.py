"""Train matched Mamba-1 LMs on Tiny Shakespeare; audit gradients before timing.

Run with ``python -m benchmarks.real_text --help``. Upstream's CUDA backend
cannot run on a Mac: its unchanged Mamba class uses its own reference scan here.
No packages are installed. Downloaded source/data have pinned SHA-256 hashes.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Optional
import urllib.request

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange, repeat

from benchmarks.blocks import load_pr78
from mambapy.lm import LM
from mambapy.mamba import MambaConfig


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REV = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
DATA_REV = '6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e'
PR_REV = '94c9dc285e104f2fceab0119d6d3198c091b6e01'
UPSTREAM_URL = f'https://raw.githubusercontent.com/state-spaces/mamba/{UPSTREAM_REV}'
ASSETS = {
    'mamba_simple.py': (f'{UPSTREAM_URL}/mamba_ssm/modules/mamba_simple.py',
        'a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3'),
    'selective_scan_interface.py': (f'{UPSTREAM_URL}/mamba_ssm/ops/selective_scan_interface.py',
        'a570f4f15a2eebabf62024a9ffb0d1dde04280300405e1632f43809f448bfb1a'),
    'LICENSE': (f'{UPSTREAM_URL}/LICENSE',
        '760939b000194d04548ede6a857bbe735d1695d8422ec85955c8e2bd7f4b95c5'),
    'input.txt': (f'https://raw.githubusercontent.com/karpathy/char-rnn/{DATA_REV}/data/tinyshakespeare/input.txt',
        '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'),
}
# Fixed before measuring. Learning curves are diagnostic, not a numerical oracle.
TOLERANCES = {'logits': (1e-4, 1e-5), 'gradients': (1e-3, 1e-7),
              'updates': (1e-3, 1e-6)}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare_assets(cache, download=False):
    cache.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path} is missing; use --download once to fetch pinned source/data')
            data = urllib.request.urlopen(url, timeout=60).read()
            if sha256(data) != expected:
                raise ValueError(f'Unexpected download hash: {url}')
            path.write_bytes(data)
        if sha256(path.read_bytes()) != expected:
            raise ValueError(f'Asset hash mismatch: {path}; refusing changed source/data')


def load_upstream_reference(cache):
    """Execute original definition bodies; bypass CUDA-only module imports.

    This is an explicit dependency adapter, not a claim of native upstream MPS
    support. The downloaded definitions are unchanged and hash verified.
    """
    namespace = dict(torch=torch, nn=nn, F=F, Tensor=Tensor, Optional=Optional,
                     math=math, rearrange=rearrange, repeat=repeat,
                     causal_conv1d_fn=None, causal_conv1d_update=None,
                     selective_state_update=None, mamba_inner_fn=None)
    for filename, name in [('selective_scan_interface.py', 'selective_scan_ref'),
                           ('mamba_simple.py', 'Mamba')]:
        path = cache / filename
        if sha256(path.read_bytes()) != ASSETS[filename][1]:
            raise ValueError(f'Upstream source hash mismatch: {path}')
        tree = ast.parse(path.read_text())
        definition, = [node for node in tree.body if getattr(node, 'name', None) == name]
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
        if name == 'selective_scan_ref':
            namespace['selective_scan_fn'] = namespace[name]
    return namespace['Mamba']


def corpus_splits(text, length):
    # Reserve zero for LM's padding_idx; every corpus character can be trained.
    alphabet = sorted(set(text))
    encode = {char: i + 1 for i, char in enumerate(alphabet)}
    tokens = torch.tensor([encode[char] for char in text], dtype=torch.long)
    boundary = int(len(tokens) * .9)
    train, validation = tokens[:boundary], tokens[boundary:]
    if min(len(train), len(validation)) <= length:
        raise ValueError('Corpus partitions must each contain more than sequence_length characters')
    return train, validation, alphabet


def make_batches(tokens, count, batch, length, seed):
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(len(tokens) - length, (count, batch), generator=generator)
    offsets = torch.arange(length)
    x = tokens[starts[..., None] + offsets]
    y = tokens[starts[..., None] + offsets + 1]
    return x, y, starts


def with_mixers(base, factory):
    model = copy.deepcopy(base)
    for layer in model.mamba.layers:
        layer.mixer = factory()
    model.load_state_dict(base.state_dict(), strict=True)
    return model


def difference(actual, expected, kind):
    if actual is None or expected is None:
        return {'pass': False, 'missing': 'actual' if actual is None else 'reference'}
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error = actual - expected
    rtol, atol = TOLERANCES[kind]
    return {'pass': finite and torch.allclose(actual, expected, rtol=rtol, atol=atol),
            'finite': finite,
            'max_abs_error': float(error.abs().max()) if finite else None,
            'relative_l2_error': float(error.norm() / expected.norm().clamp_min(1e-30)) if finite else None}


def optimizer_for(model, lr):
    return adamw(model.parameters(), lr)


def adamw(parameters, lr):
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=.01,
                             betas=(.9, .95), foreach=False)


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def probe(model, x, y, lr):
    model.train()
    model.zero_grad(set_to_none=True)
    embedded = []
    def retain_embedding(module, args, output):
        output.retain_grad()
        embedded.append(output)
    hook = model.embedding.register_forward_hook(retain_embedding)
    try:
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
    finally:
        hook.remove()
    grads = {name: None if p.grad is None else p.grad.detach().cpu().clone()
             for name, p in model.named_parameters()}
    grads['<embedding_output>'] = (None if embedded[0].grad is None
                                 else embedded[0].grad.detach().cpu().clone())
    before = snapshot(model)
    # Isolate optimizer correctness from its sensitivity to roundoff in gradients.
    # AdamW's first step is proportional to g/(abs(g)+eps); near-zero gradients
    # can amplify a tiny, valid CPU/MPS derivative difference into a larger update.
    cpu_parameters = {name: nn.Parameter(before[name].clone()) for name, _ in model.named_parameters()}
    for name, parameter in cpu_parameters.items():
        parameter.grad = grads[name]
    adamw(cpu_parameters.values(), lr).step()
    cpu_updates = {name: p.detach() - before[name] for name, p in cpu_parameters.items()}
    optimizer_for(model, lr).step()
    updates = {name: value - before[name] for name, value in snapshot(model).items()}
    model.load_state_dict(before)
    model.zero_grad(set_to_none=True)
    return {'logits': logits.detach().cpu(), 'loss': float(loss.detach().cpu()),
            'gradients': grads, 'updates': updates, 'cpu_updates_from_same_gradients': cpu_updates}


def audit(models, reference, weights, x, y, lr):
    reference.load_state_dict(weights)
    expected = probe(reference, x.cpu(), y.cpu(), lr)
    results = {}
    for name, model in models.items():
        saved = snapshot(model)
        model.load_state_dict(weights)
        actual = probe(model, x, y, lr)
        model.load_state_dict(saved)
        grads = {key: difference(value, expected['gradients'][key], 'gradients')
                 for key, value in actual['gradients'].items()}
        updates = {key: difference(value, actual['cpu_updates_from_same_gradients'][key], 'updates')
                   for key, value in actual['updates'].items()}
        trajectory = {key: difference(value, expected['updates'][key], 'updates')
                      for key, value in actual['updates'].items()}
        forward = difference(actual['logits'], expected['logits'], 'logits')
        results[name] = {'pass': forward['pass'] and all(v['pass'] for v in grads.values())
                         and all(v['pass'] for v in updates.values()),
                         'logits': forward, 'loss': actual['loss'], 'reference_loss': expected['loss'],
                         'missing_gradients': [key for key, value in actual['gradients'].items() if value is None],
                         'gradients': grads, 'adamw_same_gradient_updates': updates,
                         'cross_device_update_differences_diagnostic_only': trajectory}
    return results


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    losses, correct = [], []
    x, y = batches
    for inputs, targets in zip(x, y):
        logits = model(inputs)
        losses.append(F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
        correct.append((logits.argmax(-1) == targets).float().mean())
    loss = float(torch.stack(losses).mean().cpu())
    return {'cross_entropy': loss, 'perplexity': math.exp(loss) if loss < 700 else None,
            'next_character_accuracy': float(torch.stack(correct).mean().cpu())}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch pinned source and Tiny Shakespeare if absent')
    parser.add_argument('--cache', type=Path, default=ROOT / 'benchmark-results/real-text-assets')
    parser.add_argument('--pr-root', type=Path, required=True,
                        help='Audit root with pr78-review/ and compiled pr78-build/metal_pscan/')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-results/real-text.json')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--state', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--eval-batches', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--lr', type=float, default=.002)
    args = parser.parse_args()
    for name in ('steps', 'length', 'batch', 'width', 'layers', 'state', 'warmup', 'eval_every', 'eval_batches'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.state > 32:
        parser.error('PR #78 uses unchecked fixed state arrays; this comparison requires --state <=32')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        parser.error('Run with PYTORCH_ENABLE_MPS_FALLBACK=0; CPU fallback must be disabled')
    if not torch.backends.mps.is_available():
        parser.error('A real MPS GPU is required; this benchmark cannot silently skip it')
    torch.mps.set_per_process_memory_fraction(min(1., 16 * 2**30 / torch.mps.recommended_max_memory()))
    prepare_assets(args.cache, args.download)
    upstream = load_upstream_reference(args.cache)
    pr_head = subprocess.check_output(['git', '-C', str(args.pr_root / 'pr78-review'),
                                       'rev-parse', 'HEAD'], text=True).strip()
    if pr_head != PR_REV:
        parser.error(f'Expected audited PR revision {PR_REV}, got {pr_head}')
    pr_dirty = subprocess.check_output(['git', '-C', str(args.pr_root / 'pr78-review'),
                                        'diff', 'HEAD', '--', 'mambapy', 'metal_pscan'], text=True)
    if pr_dirty:
        parser.error('PR source has modifications; restore the audited checkout before comparison')
    pr = load_pr78(args.pr_root)
    train, validation, alphabet = corpus_splits((args.cache / 'input.txt').read_text(), args.length)
    train_x, train_y, starts = make_batches(train, args.steps, args.batch, args.length, args.seed + 1)
    val_x, val_y, val_starts = make_batches(validation, args.eval_batches, args.batch, args.length, args.seed + 2)
    # Transfers and data preparation are excluded from update timing.
    train_x, train_y, val_x, val_y = [t.to('mps') for t in (train_x, train_y, val_x, val_y)]
    torch.manual_seed(args.seed)
    config = MambaConfig(args.width, args.layers, d_state=args.state, backend='mps')
    base = LM(config, len(alphabet) + 1)
    original = with_mixers(base, lambda: upstream(args.width, d_state=args.state, use_fast_path=False))
    reference = copy.deepcopy(original)  # Independent upstream CPU numerical oracle.
    old = with_mixers(base, lambda: pr.MambaBlock(pr.MambaConfig(args.width, args.layers, d_state=args.state)))
    models = {'state_spaces_reference_mps': original.to('mps'),
              'ours_mps': copy.deepcopy(base).to('mps'), 'pr78_mps': old.to('mps')}
    initial = snapshot(base)
    names = list(models)
    report = {'status': 'running', 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'environment': {'torch': torch.__version__, 'python': platform.python_version(),
                              'platform': platform.platform(), 'dtype': 'float32', 'device': 'mps',
                              'cpu_fallback': False, 'allocator_budget_gib': 16},
              'provenance': {'fork_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                             'benchmark_sha256': sha256(Path(__file__).read_bytes()),
                             'fork_source_hashes': {str(p.relative_to(ROOT)): sha256(p.read_bytes())
                                 for p in sorted((ROOT / 'mambapy').rglob('*'))
                                 if p.suffix in ('.py', '.metal')},
                             'upstream_revision': UPSTREAM_REV, 'pr_revision': pr_head,
                             'assets': ASSETS,
                             'pr_build_hashes': {str(p.relative_to(args.pr_root)): sha256(p.read_bytes())
                                 for p in sorted((args.pr_root / 'pr78-build').rglob('*'))
                                 if p.suffix in ('.so', '.py')}},
              'implementations': {'state_spaces_reference_mps': 'Unchanged upstream Mamba definition + upstream selective_scan_ref; CUDA imports bypassed, use_fast_path=False. NOT original CUDA performance.',
                                  'ours_mps': 'Fork MambaBlock, backend=mps',
                                  'pr78_mps': 'Audited PR MambaBlock with its original fused Metal dispatch; no autograd repair'},
              'model': {'layers': args.layers, 'width': args.width, 'state': args.state,
                        'parameters': sum(p.numel() for p in base.parameters()),
                        'shared_wrapper': 'Identical LM embedding, residual/RMSNorm, head; only Mamba mixers differ'},
              'data': {'name': 'Tiny Shakespeare', 'task': 'next-character prediction', 'vocabulary': alphabet,
                       'padding_id': 0, 'train_characters': len(train), 'validation_characters': len(validation),
                       'split': 'first 90% train, last 10% validation; no windows cross split',
                       'train_starts': starts.tolist(), 'validation_starts': val_starts.tolist()},
              'tolerances_rtol_atol': TOLERANCES,
              'audit_protocol': 'Logits and all gradients versus original upstream CPU FP32 reference, at identical initial and learned weights. AdamW CPU/MPS updates compared using identical supplied gradients. Updates using independently computed CPU/MPS gradients are reported separately as diagnostics, because AdamW amplifies near-zero derivative roundoff.',
              'timing_protocol': 'Synchronized wall-clock full AdamW update; batches resident on MPS. First update and warmups excluded. Weights and optimizer reset after warmup. Backend order rotates each step. Validation, audits and CPU metrics excluded. p95 is empirical.',
              'results': {name: {'samples_ms': [], 'train_losses': [], 'validation': []} for name in names}}
    print('Original state-spaces CUDA kernels are unavailable on MPS; timing its upstream reference scan.', flush=True)
    write_report(args.output, report)
    # Measure each implementation's first training call before the numerical audit.
    for name, model in models.items():
        optimizer = optimizer_for(model, args.lr)
        torch.mps.synchronize()
        start = time.perf_counter()
        train_step(model, optimizer, train_x[0], train_y[0])
        torch.mps.synchronize()
        report['results'][name]['first_update_ms'] = (time.perf_counter() - start) * 1000
        for _ in range(args.warmup):
            train_step(model, optimizer, train_x[0], train_y[0])
        torch.mps.synchronize()
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
    report['initial_audit'] = audit(models, reference, initial, train_x[0], train_y[0], args.lr)
    for name, checks in report['initial_audit'].items():
        print(f'{name}: initial correctness={checks["pass"]}, missing gradients={checks["missing_gradients"]}', flush=True)
    optimizers = {name: optimizer_for(model, args.lr) for name, model in models.items()}
    def validation_checkpoint(step):
        for name, model in models.items():
            metrics = evaluate(model, (val_x, val_y))
            report['results'][name]['validation'].append({'step': step, **metrics})
            print(f'step {step:4d} {name}: validation CE={metrics["cross_entropy"]:.5f}, perplexity={metrics["perplexity"]:.3f}', flush=True)
        write_report(args.output, report)
    validation_checkpoint(0)
    for step in range(args.steps):
        order = names[step % len(names):] + names[:step % len(names)]
        for name in order:
            torch.mps.synchronize()
            start = time.perf_counter()
            loss = train_step(models[name], optimizers[name], train_x[step], train_y[step])
            torch.mps.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            loss_value = float(loss.cpu())
            if not math.isfinite(loss_value):
                report['status'] = f'nonfinite_loss:{name}:step{step + 1}'
                write_report(args.output, report)
                raise RuntimeError(report['status'])
            report['results'][name]['samples_ms'].append(elapsed)
            report['results'][name]['train_losses'].append(loss_value)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            validation_checkpoint(step + 1)
    # Recheck at learned weights, always identical across implementations.
    report['trained_weight_audit'] = audit(models, reference, snapshot(models['ours_mps']),
                                            val_x[0], val_y[0], args.lr)
    for name, row in report['results'].items():
        values = row['samples_ms']
        checks = [report[stage][name]['pass'] for stage in ('initial_audit', 'trained_weight_audit')]
        row.update(median_update_ms=statistics.median(values),
                   p95_update_ms=sorted(values)[min(len(values) - 1, int(.95 * len(values)))],
                   total_update_ms=sum(values), tokens_per_second=args.batch * args.length * 1000 / statistics.median(values),
                   correctness_pass=all(checks), valid_training_timing=all(checks),
                   unchanged_parameters=[key for key, value in snapshot(models[name]).items()
                                         if torch.equal(value, initial[key])])
        print(f'{name}: median={row["median_update_ms"]:.3f} ms, p95={row["p95_update_ms"]:.3f} ms, '
              f'correct={row["correctness_pass"]}; validation CE={row["validation"][-1]["cross_entropy"]:.5f}', flush=True)
    report['status'] = 'complete' if all(row['correctness_pass'] for row in report['results'].values()) else 'complete_with_correctness_failures'
    write_report(args.output, report)
    print(f'Report: {args.output}', flush=True)
    # A broken PR run remains visible but is never a successful training benchmark.
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
