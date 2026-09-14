"""Peak-2D recomputation parity, including >1000 atoms and optimizer replay."""
import argparse
import copy
import unittest
import warnings
from unittest.mock import patch

import torch
from test_gmm import GaussianProjector, OLD, image_objective, scene
from gmm import add_gmm_arguments, gmm_from_arguments
from pts2img import sum_of_gaussians_2d_torch_checkpointed


class Peak2DCheckpointTests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings('ignore', message='.*grid_sample.*')
        warnings.filterwarnings('ignore', message='.*meshgrid.*')

    def test_multiple_crops_partial_chunk_raw_image_and_all_gradients(self):
        # Deliberately use distinct block supports: this detects loop-closure
        # replay bugs that a single-block or identical-crop test would miss.
        gen = torch.Generator().manual_seed(573)
        centers = torch.rand(2, 2003, 2, generator=gen, dtype=torch.float64)*3
        centers[:, :1000] += 6
        centers[:, 1000:2000] += 15
        centers[:, 2000:] += 24
        weights = torch.rand(2003, generator=gen, dtype=torch.float64)*3 + .5
        widths = torch.rand(2003, 2, generator=gen, dtype=torch.float64)*.7 + .09
        results=[]
        for fn in (OLD.sum_of_gaussians_2d_torch, sum_of_gaussians_2d_torch_checkpointed):
            tensors = [v.clone().requires_grad_() for v in (centers,weights,widths)]
            image=fn(*tensors, 5, torch.zeros(2,32,32,dtype=torch.float64))
            image_objective(image).backward()
            results.append((image, [v.grad for v in tensors]))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
        for a,b in zip(results[0][1],results[1][1]):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def check_training(self, device, dtype, cpu_weights=False):
        gen=torch.Generator().manual_seed(456)
        xyz=(torch.rand(1003,3,generator=gen)*5-2.5).to(device=device,dtype=dtype)
        _,rot,trans,center=scene(dtype=dtype,device=device)
        weights=(torch.rand(1003,generator=gen)*22+.3).to(dtype=dtype)
        if not cpu_weights:
            weights=weights.to(device)
        for mode in ('legacy','isotropic'):
            with self.subTest(device=device,dtype=dtype,mode=mode,cpu_weights=cpu_weights):
                old=GaussianProjector(weights,mode,amplitude_convention='peak_2d',shape_device=device)
                with torch.no_grad():
                    if mode=='legacy':
                        old.sdevs[::7,0]=.09; old.sdevs[::11,1]=.87
                    else:
                        old.raw_sigma[::7,0]=-3.; old.raw_sigma[::11,0]=.7
                new=copy.deepcopy(old); new.checkpoint_peak2d=True
                states=[]
                for model in (old,new):
                    inputs=[t.clone().requires_grad_() for t in (xyz,rot,trans)]
                    opt=torch.optim.AdamW([
                        {'params':inputs,'lr':.001},
                        {'params':model.amplitude_parameters(),'lr':.01},
                        {'params':model.shape_parameters(),'lr':.005}])
                    states.append((model,inputs,opt))
                max_grad=0.; max_param=0.; max_image=0.
                # Multiple optimizer steps and repeated backward accumulation.
                for step in range(4):
                    snapshots=[]
                    for model,inputs,opt in states:
                        opt.zero_grad()
                        for repeat in range(2):
                            image=model(inputs[0],inputs[1],inputs[2],2.7,center,box_size=24,apix=1.073)
                            loss=image_objective(image)+model.regularization()
                            loss.backward()
                        params=inputs+list(model.parameters())
                        snapshots.append((image.detach().clone(),[p.grad.clone() for p in params]))
                        opt.step()
                    max_image=max(max_image,(snapshots[0][0]-snapshots[1][0]).abs().max().item())
                    for a,b in zip(snapshots[0][1],snapshots[1][1]):
                        max_grad=max(max_grad,(a-b).abs().max().item())
                        torch.testing.assert_close(a,b,rtol=2e-5 if device=='cuda' else 0,atol=2e-7 if device=='cuda' else 0)
                    for a,b in zip(states[0][1]+list(old.parameters()),states[1][1]+list(new.parameters())):
                        max_param=max(max_param,(a-b).abs().max().item())
                        torch.testing.assert_close(a,b,rtol=2e-5 if device=='cuda' else 0,atol=2e-7 if device=='cuda' else 0)
                    torch.testing.assert_close(snapshots[0][0],snapshots[1][0],rtol=2e-5 if device=='cuda' else 0,atol=2e-7 if device=='cuda' else 0)
                print(f'PARITY {device} {dtype} {mode} cpu_weights={cpu_weights}: image={max_image:.3g} gradient={max_grad:.3g} parameter={max_param:.3g}')

    def test_cpu_multiple_adamw_steps(self):
        self.check_training('cpu',torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_multiple_adamw_steps_and_cpu_amplitudes(self):
        self.check_training('cuda',torch.float32)
        self.check_training('cuda',torch.float32,cpu_weights=True)

    def test_raw_component_path_and_replay_is_actually_used(self):
        import torch.utils.checkpoint as cp
        xyz,rot,trans,center=scene()
        for mode in ('legacy','isotropic'):
            model=GaussianProjector(torch.tensor([2.,5.,9.],dtype=xyz.dtype),mode,amplitude_convention='peak_2d')
            results=[]
            for on in (False,True):
                model.checkpoint_peak2d=on
                projected=model.project_coordinates(xyz,rot,1.073)
                origin=projected.amin(1,keepdim=True)
                with patch.object(cp,'checkpoint',wraps=cp.checkpoint) as mocked:
                    raw=model.render_raw(projected,rot,origin,2.7,24)
                    grads=torch.autograd.grad(image_objective(raw),list(model.parameters()))
                    self.assertEqual(mocked.call_count,1 if on else 0)
                results.append((raw,grads))
            torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
            for a,b in zip(results[0][1],results[1][1]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_cli_save_load_and_no_grad(self):
        parser=argparse.ArgumentParser(); add_gmm_arguments(parser)
        for mode in ('legacy','isotropic'):
            args=parser.parse_args(['--gmm-kernel',mode,'--gmm-amplitude','peak_2d'])
            self.assertFalse(gmm_from_arguments(torch.ones(3),args).checkpoint_peak2d)
            args=parser.parse_args(['--gmm-kernel',mode,'--gmm-amplitude','peak_2d','--gmm-checkpoint-peak2d'])
            model=gmm_from_arguments(torch.ones(3,dtype=torch.float64),args)
            loaded=GaussianProjector.from_checkpoint(model.export_checkpoint())
            self.assertTrue(loaded.checkpoint_peak2d)
            old_payload=model.export_checkpoint(); old_payload['config'].pop('checkpoint_peak2d')
            self.assertFalse(GaussianProjector.from_checkpoint(old_payload).checkpoint_peak2d)
            xyz,rot,trans,center=scene()
            with torch.no_grad():
                expected=model(xyz,rot,trans,3.,center,box_size=24)
                loaded.checkpoint_peak2d=False
                actual=loaded(xyz,rot,trans,3.,center,box_size=24)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
