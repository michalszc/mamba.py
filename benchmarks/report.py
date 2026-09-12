"""Collect reproducible results and evaluate the agreed numerical speed gates."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('benchmark-results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {p.stem: json.loads(p.read_text()) for p in args.results_dir.glob('*.json')
               if p.stem.startswith(('small-', '130m-', 'controlled-blocks'))}

    def metric(name, length, operation):
        return next(r for r in results[name]['rows'] if r['length'] == length)[operation]['median_ms']

    gates = {}
    for size in ('small', '130m'):
        for backend in ('mps', 'transformer'):
            for mode, lengths in (('speed', {512, 2048}), ('generation', {128, 512, 2048})):
                run = results[f'{size}-{backend}-{mode}']
                assert run['args']['dtype'] == 'float32'
                assert {r['length'] for r in run['rows']} == lengths
                assert all('out_of_budget' not in r for r in run['rows'])
                if mode == 'generation':
                    assert run['args']['tokens'] == 256
                    assert all('decode' in r for r in run['rows'])
        native_count = results[f'{size}-mps-speed']['parameters']
        transformer_count = results[f'{size}-transformer-speed']['parameters']
        assert abs(transformer_count/native_count - 1) < .02
    for operation in ('training', 'prefill'):
        speedups = [metric('small-torch-speed', length, operation) /
                    metric('small-mps-speed', length, operation) for length in (512, 2048)]
        mean = statistics.geometric_mean(speedups)
        gates[operation] = dict(speedups=speedups, geometric_mean=mean, target=2, passed=mean >= 2)
    for length in (512, 2048):
        ratio = metric('small-torch-uncached', length, 'generation') / metric('small-mps-generation', length, 'generation')
        gates[f'generation_{length}'] = dict(speedup=ratio, target=2, passed=ratio >= 2)
    blocks = results['controlled-blocks']['rows']
    assert {(r['batch'],r['length'],r['width']) for r in blocks} == {
        (2,64,128), (2,512,128), (2,2048,128), (2,512,384), (1,2048,384)}
    gates['pr78_forward'] = dict(ratios=[r['pr_ratio'] for r in blocks],
                               maximum_ratio=1.1, passed=all(r['pr_gate_pass'] for r in blocks))
    if 'controlled-blocks-repeat' in results:
        repeated = results['controlled-blocks-repeat']['rows']
        assert len(repeated) == 5
        gates['pr78_forward_repeat'] = dict(ratios=[r['pr_ratio'] for r in repeated],
                                            maximum_ratio=1.1, passed=all(r['pr_gate_pass'] for r in repeated))
    extra = {}
    for name in ('tuning', 'compilation', 'convolution', 'groups', 'long-sequence'):
        path = args.results_dir / (name + '.jsonl')
        if path.exists():
            extra[name] = [json.loads(line) for line in path.read_text().splitlines()]
    sources = list(Path('mambapy').rglob('*.py')) + list(Path('mambapy').rglob('*.metal')) + list(Path('benchmarks').glob('*.py'))
    report = dict(hardware='Apple M4 Pro, 24 GiB', gates=gates, runs=results, diagnostics=extra,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(gates, indent=2))
    if not all(g['passed'] for g in gates.values()):
        raise SystemExit('Performance gates remain open')


if __name__ == '__main__':
    main()
