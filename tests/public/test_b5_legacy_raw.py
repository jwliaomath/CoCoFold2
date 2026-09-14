"""Missing placement fields permit explicit raw export, never implicit alignment."""
from types import SimpleNamespace

import pytest
import torch

from test_foundations import make_cache, write_fixture, fake_protenix
from structure_io import read_template


@pytest.mark.parametrize('missing', [('rotation',), ('translation',), ('rotation', 'translation')])
def test_legacy_raw_without_placement(tmp_path, monkeypatch, fake_protenix, missing):
    import get_pdb
    cache = make_cache()
    cache.update(p_lm=None, c_l=None, N_sample=1, inplace_safe=False,
                 rotation=torch.eye(3), translation=torch.tensor([20., 30., 40.]))
    for key in missing:
        del cache[key]
    raw = torch.arange(18).reshape(1, 6, 3).float() / 10
    def sample(**kw):
        torch.testing.assert_close(kw['pair_z'], cache['pair_z'] * cache['z_mul'] + cache['z_bias'])
        assert kw['enable_efficient_fusion'] is True
        return raw
    monkeypatch.setattr(get_pdb, '_sample_diffusion', sample)
    path = tmp_path / 'old.pth'
    torch.save(cache, path)
    before = path.read_bytes()
    get_pdb.main(SimpleNamespace(device='cpu', diffusion_data_dir=str(path), out_dir=str(tmp_path / 'out'),
                  cif_path=str(write_fixture(tmp_path)), pdbid='old', output_format='cif', coordinate_frame='raw'))
    torch.testing.assert_close(read_template(tmp_path / 'out/old_refined_prediction.cif')[0], raw[0], rtol=0, atol=1e-6)
    assert path.read_bytes() == before


def test_missing_placement_rejected_before_reference_decode(tmp_path, monkeypatch):
    import get_pdb
    cache = make_cache()
    torch.save(cache, tmp_path / 'old.pth')
    monkeypatch.setattr(get_pdb, '_decode_aligned_global', lambda *a, **k: pytest.fail('Model decode must not start'))
    with pytest.raises(ValueError, match='rotation/translation'):
        get_pdb.main(SimpleNamespace(device='cpu', diffusion_data_dir=str(tmp_path / 'old.pth'),
                      out_dir=str(tmp_path / 'out'), cif_path=str(write_fixture(tmp_path)), pdbid='old', output_format='cif'))
