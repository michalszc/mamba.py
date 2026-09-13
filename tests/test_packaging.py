"""Build with existing setuptools; exercise imports from the unpacked wheel."""
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def distribution(tmp_path_factory):
    target = tmp_path_factory.mktemp('mamba-distribution')
    subprocess.run([sys.executable, 'setup.py', 'sdist', '--dist-dir', str(target),
                    'bdist_wheel', '--dist-dir', str(target)], cwd=ROOT,
                   check=True, capture_output=True, text=True)
    wheel = next(target.glob('*.whl'))
    source = next(target.glob('*.tar.gz'))
    expected = {'mambapy/mps/scan.metal', 'mambapy/mps/conv.metal'}
    unpacked = target / 'unpacked'
    with zipfile.ZipFile(wheel) as archive:
        assert expected <= set(archive.namelist())
        archive.extractall(unpacked)
    with tarfile.open(source) as archive:
        names = {'/'.join(name.split('/')[1:]) for name in archive.getnames()}
        assert expected <= names
        assert {'docs/MPS.md', 'docs/MPS_BENCHMARKS.md', 'benchmarks/acceptance.py'} <= names
    return unpacked


def run_packaged(unpacked, source):
    env = {**os.environ, 'PYTHONPATH': str(unpacked), 'PYTORCH_ENABLE_MPS_FALLBACK': '0'}
    subprocess.run([sys.executable, '-c', source], cwd=unpacked.parent,
                   env=env, check=True, capture_output=True, text=True)


def test_packaged_cpu_does_not_load_metal(distribution):
    run_packaged(distribution, '''
import sys
from pathlib import Path
from unittest.mock import patch
import torch
import mambapy
from mambapy.lm import LM
from mambapy.mamba import MambaConfig
assert 'unpacked' in Path(mambapy.__file__).parts
with patch('torch.mps.compile_shader', side_effect=AssertionError('Unexpected Metal compilation')):
    model = LM(MambaConfig(16, 1), 64)
    tokens = torch.randint(0, 64, (1, 7))
    model(tokens).square().mean().backward()
    assert model.generate(tokens, 3, sample=False).shape == (1, 3)
assert 'mambapy.mps.scan' not in sys.modules
assert 'mambapy.mps.conv' not in sys.modules
''')


def test_packaged_mps_training_and_generation(distribution):
    if not torch.backends.mps.is_available():
        if os.environ.get('MAMBAPY_REQUIRE_MPS') == '1':
            pytest.fail('Real MPS execution is required for release validation')
        pytest.skip('Real MPS required')
    run_packaged(distribution, '''
from pathlib import Path
import torch
import mambapy
from mambapy.lm import LM
from mambapy.mamba import MambaConfig
assert 'unpacked' in Path(mambapy.__file__).parts
assert torch.backends.mps.is_available()
model = LM(MambaConfig(16, 1, backend='mps'), 64).to('mps')
tokens = torch.randint(0, 64, (2, 65), device='mps')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
with torch.autocast('mps', dtype=torch.bfloat16):
    model(tokens).float().square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer.step()
assert model.generate(tokens[:, :3], 4, sample=False).shape == (2, 4)
torch.mps.synchronize()
''')
