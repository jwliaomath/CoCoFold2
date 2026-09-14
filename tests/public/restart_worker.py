"""Isolated-process test driver with an analytic decoder; no real model weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import pytest
import torch

from test_b2b_randomness import sampler
from test_epoch_restart import install


if __name__ == '__main__':
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        sample = sampler.__wrapped__(patch)
        install(patch, sample)
        import train
        train.main(train.build_parser().parse_args(sys.argv[1:]))
