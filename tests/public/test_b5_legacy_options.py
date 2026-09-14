"""Historical optional fusion flag: saved fields/configs precede Protenix inference default True."""
import json
from types import SimpleNamespace

import pytest
import torch

from test_foundations import make_cache, write_fixture, fake_protenix
from structure_io import read_template


@pytest.mark.parametrize('option,config_option,expected_flag,source', [
    ('missing', 'missing', True, 'protenix_inference_default_true'),
    ('missing', False, False, 'checkpoint_configs'),
    ('missing', True, True, 'checkpoint_configs'),
    (False, 'missing', False, 'checkpoint'),
    (True, 'missing', True, 'checkpoint'),
    (False, True, False, 'checkpoint'),
    (True, False, True, 'checkpoint'),
])
def test_export_legacy_optional_fusion(tmp_path, monkeypatch, fake_protenix, option, config_option, expected_flag, source):
    import get_pdb
    cache = make_cache()
    cache.update(pred_dict={}, p_lm=None, c_l=None, N_sample=1, inplace_safe=False,
                 rotation=torch.eye(3), translation=torch.tensor([1., 2., 3.]))
    if option != 'missing':
        cache['enable_efficient_fusion'] = option
    if config_option != 'missing':
        cache['configs'].enable_efficient_fusion = config_option
    raw = torch.arange(18).reshape(1, 6, 3).float() / 10
    calls = []
    def sample(**kw):
        assert kw['enable_efficient_fusion'] is expected_flag
        torch.testing.assert_close(kw['pair_z'], cache['pair_z'] * cache['z_mul'] + cache['z_bias'])
        calls.append(kw['enable_efficient_fusion'])
        return raw
    monkeypatch.setattr(get_pdb, '_sample_diffusion', sample)
    path = tmp_path / 'legacy.pth'
    torch.save(cache, path)
    before = path.read_bytes()
    get_pdb.main(SimpleNamespace(device='cpu', diffusion_data_dir=str(path), out_dir=str(tmp_path / 'export'),
                 cif_path=str(write_fixture(tmp_path)), pdbid='old', output_format='cif'))
    torch.testing.assert_close(read_template(tmp_path / 'export/old_refined_prediction.cif')[0],
                               raw[0] + cache['translation'], atol=1e-6, rtol=0)
    assert calls == [expected_flag] and path.read_bytes() == before

    record = next((tmp_path / 'export/records').iterdir())
    stages = json.loads((record / 'resolved_config.json').read_text(encoding='utf-8'))['stages']
    values = next(s['values'] for s in stages if s['stage'] == 'decode_configuration')
    assert values['enable_efficient_fusion'] is expected_flag
    assert values['efficient_fusion_source'] == source


@pytest.mark.parametrize('flag', [False, True])
def test_mapping_configs_preserve_explicit_fusion(flag):
    from get_pdb import _resolve_efficient_fusion
    assert _resolve_efficient_fusion({'configs': {'enable_efficient_fusion': flag}}) == (flag, 'checkpoint_configs')
