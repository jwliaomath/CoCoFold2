"""Fixed coordinate transforms shared by training and structure export.

Row vectors, Angstrom: Y[a] = X[a] @ rotations[body[a]] + translations[body[a]].
The existing global branches remain unchanged unless block mode is requested.
"""
import json
import copy
import hashlib
from pathlib import Path
import gemmi
import numpy as np
import torch
from torch import nn


def add_coordinate_arguments(parser):
    parser.add_argument('--coordinate-mode', choices=('global', 'blocks'), default='global',
                        help='Original global alignment by default; saved block checkpoints retain their mode')
    parser.add_argument('--block-alignment', help='Block manifest JSON for FIRST initialization only')
    parser.add_argument('--alignment-sampler', choices=('auto', 'global', 'block'), default='auto',
                        help='auto: new per-chain manifests use original train sampler; old block checkpoints keep their saved decoder. Explicit override only for new initialization.')
    parser.add_argument('--coordinate-handoff-tolerance', type=float, default=.05,
                        help='Maximum raw-coordinate RMS difference in Angstrom at block checkpoint handoff')
    parser.add_argument('--block-update-trace-threshold', type=float, default=2.5,
                        help='Shared threshold for optional per-body updates: trace(R_new @ R_old.T) below this value triggers R/t replacement; default 2.5, valid (-1,3)')


def template_atom_keys(path):
    """Atom identities in the SAME row order/filter as cif_to_tensor/read_template.

Use author IDs to match PDB split annotations, not label_seq_id (which can differ).
No sorting, chain guessing, added-H filtering or atom-count-only matching.
"""
    path = Path(path)
    keys = []
    if path.suffix.lower() in ('.cif', '.mmcif'):
        block = gemmi.cif.read_file(str(path)).sole_block()
        tags = ['label_alt_id', 'auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code',
                'label_atom_id', 'type_symbol']
        table = block.find('_atom_site.', tags)
        if not table:
            raise ValueError('Block template requires author chain/residue IDs and atom identities')
        for row in table:
            if row[0] not in ('.', ''):
                continue
            ins = '' if row[3] in ('.', '?') else str(row[3])
            keys.append([str(row[1]), int(row[2]), ins, str(row[4]), str(row[5]).upper()])
    else:
        structure = gemmi.read_structure(str(path))
        if len(structure) != 1:
            raise ValueError('Block template must contain one model')
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:
                    if atom.altloc not in ('\x00', ' '):
                        continue
                    keys.append([chain.name, residue.seqid.num, residue.seqid.icode.strip(),
                                 atom.name, atom.element.name.upper()])
    if not keys or len({tuple(k) for k in keys}) != len(keys):
        raise ValueError('Empty/duplicate template atom identities; check models and alternate locations')
    return keys


def fit_rigid_row(x, y):
    if len(x) < 3 or torch.linalg.matrix_rank(x-x.mean(0)) < 2:
        raise ValueError('Each fitting core needs >=3 noncollinear atoms')
    if torch.linalg.matrix_rank(y-y.mean(0)) < 2:
        raise ValueError('Target fitting core is collinear')
    u, _, vh = torch.linalg.svd((x-x.mean(0)).T @ (y-y.mean(0)))
    d = torch.ones(3, dtype=x.dtype, device=x.device)
    d[-1] = torch.linalg.det(u @ vh)
    rotation = (u*d) @ vh
    return rotation, y.mean(0)-x.mean(0) @ rotation


def validate_alignment_manifest(data, keys):
    """Validate identities, complete body assignment and fitting cores on CPU."""
    if not isinstance(data, dict) or data.get('format') != 'cocofold2-block-manifest-v1':
        raise ValueError('Unsupported block manifest')
    entries, names = data.get('atoms'), data.get('body_names')
    if (not isinstance(names, list) or not names or
            any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names)):
        raise ValueError('Empty/duplicate block names')
    if not isinstance(entries, list) or not entries or any(not isinstance(a, dict) for a in entries):
        raise ValueError('Invalid block atoms')
    try:
        by_key = {tuple(a['key']): a for a in entries}
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid manifest atom identities') from exc
    if len(by_key) != len(entries) or set(by_key) != {tuple(k) for k in keys}:
        raise ValueError('Manifest/template atom identities mismatch; exact coverage required')
    ordered = [by_key[tuple(k)] for k in keys]
    if (any(type(a.get('body_id')) is not int for a in ordered) or
            {a['body_id'] for a in ordered} != set(range(len(names)))):
        raise ValueError('Invalid block assignments')
    if any(type(a.get('fit_core')) is not bool for a in ordered):
        raise ValueError('Block fit_core must be boolean')
    try:
        target = torch.tensor([a.get('target') for a in ordered], dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid block target coordinates') from exc
    if target.shape != (len(keys), 3) or not torch.isfinite(target).all():
        raise ValueError('Invalid block target coordinates')
    if data.get('grouping') == 'auth_asym_id':
        if any(a['key'][0] != names[a['body_id']] for a in ordered):
            raise ValueError('Chain manifest must assign each author chain to its own body')
    for i, name in enumerate(names):
        selected = [j for j, a in enumerate(ordered) if a['body_id'] == i and a['fit_core']]
        if len(selected) < 3:
            raise ValueError(f'Block {name}: requires at least three fitting atoms')
        core = target[selected]
        if torch.linalg.matrix_rank(core - core.mean(0)) < 2:
            raise ValueError(f'Block {name}: target fitting core is collinear')
    return ordered


def read_alignment_manifest(path, template):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    ordered = validate_alignment_manifest(data, template_atom_keys(template))
    return data, ordered


def subset_alignment_manifest(data, atom_keys):
    """Select complete bodies by explicit global atom identity, never local chain guesses.

    The result contains reference targets, not fitted transforms: a future component
    must fit against its OWN decoded coordinates. No GPU/rank assignment is inferred.
    """
    validate_alignment_manifest(data, [a['key'] for a in data['atoms']])
    wanted = {tuple(k) for k in atom_keys}
    if len(wanted) != len(atom_keys) or not wanted:
        raise ValueError('Subset requires nonempty unique global atom identities')
    by_key = {tuple(a['key']): a for a in data['atoms']}
    if not wanted <= by_key.keys():
        raise ValueError('Subset contains unknown global atom identities')
    selected = [by_key[tuple(k)] for k in atom_keys]
    bodies = sorted({a['body_id'] for a in selected})
    if any(tuple(a['key']) not in wanted for a in data['atoms'] if a['body_id'] in bodies):
        raise ValueError('Subset must retain every atom of each selected alignment body')
    result = copy.deepcopy(data)
    result['body_names'] = [data['body_names'][i] for i in bodies]
    result['atoms'] = [dict(copy.deepcopy(a), body_id=bodies.index(a['body_id'])) for a in selected]
    validate_alignment_manifest(result, atom_keys)
    return result


class CoordinateTransform(nn.Module):
    VERSION = 'cocofold2-block-rigid-v1'

    def __init__(self, rotations, translations, atom_body_id, atom_keys, body_names,
                 initial_raw, report=None, handoff_raw=None, fit_targets=None, fit_core=None):
        super().__init__()
        r = torch.as_tensor(rotations).detach().clone()
        if not r.is_floating_point(): r = r.float()
        t = torch.as_tensor(translations, dtype=r.dtype, device=r.device).detach().clone()
        ids = torch.as_tensor(atom_body_id, device=r.device)
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('atom_body_id must contain integers')
        ids = ids.long().detach().clone()
        raw = torch.as_tensor(initial_raw, dtype=r.dtype, device=r.device).detach().clone()
        n, k = len(atom_keys), len(body_names)
        if r.shape != (k,3,3) or t.shape != (k,3) or ids.shape != (n,) or raw.shape != (n,3):
            raise ValueError('Invalid block transform dimensions')
        if not k or not n or len(set(body_names)) != k or len({tuple(a) for a in atom_keys}) != n:
            raise ValueError('Empty/duplicate blocks or atom identities')
        if set(ids.cpu().tolist()) != set(range(k)):
            raise ValueError('Every block must be nonempty; block indices must be contiguous')
        if not all(torch.isfinite(v).all() for v in (r,t,raw)):
            raise ValueError('Nonfinite block transform')
        # medium/high float32 matmul can use reduced precision on CUDA. Check
        # geometry in float64 so TF32/BF16 roundoff cannot reject a valid R.
        # Keep the original storage dtype, tolerance and strict SO(3) test.
        checked_r = r.double()
        gram = checked_r.transpose(-1,-2) @ checked_r
        determinants = torch.linalg.det(checked_r)
        eye = torch.eye(3, dtype=checked_r.dtype, device=r.device).expand(k,-1,-1)
        if not torch.allclose(gram, eye, atol=1e-5, rtol=1e-5) or not torch.allclose(determinants, checked_r.new_ones(k), atol=1e-5, rtol=1e-5):
            orth_error = (gram-eye).abs().max().item()
            det_error = (determinants-1).abs().max().item()
            raise ValueError('Only proper rigid rotations are allowed, not reflection/scale/shear; '
                             f'float64 check: max |R^T R-I|={orth_error:.6g}, '
                             f'max |det(R)-1|={det_error:.6g}, stored dtype={r.dtype}')
        self.register_buffer('rotations',r)
        self.register_buffer('translations',t)
        self.register_buffer('atom_body_id',ids)
        self.register_buffer('initial_raw',raw)
        handoff = raw if handoff_raw is None else torch.as_tensor(handoff_raw, dtype=r.dtype, device=r.device)
        if handoff.shape != raw.shape or not torch.isfinite(handoff).all():
            raise ValueError('Invalid handoff coordinates')
        self.register_buffer('handoff_raw',handoff.detach().clone())
        self.atom_keys = [list(a) for a in atom_keys]
        self.body_names = list(body_names)
        self.report = report or {}
        if (fit_targets is None) != (fit_core is None):
            raise ValueError('Saved fitting targets and core must be provided together')
        targets = None if fit_targets is None else torch.as_tensor(fit_targets, device=r.device).detach().clone().double()
        core = None if fit_core is None else torch.as_tensor(fit_core, device=r.device).detach().clone()
        if targets is not None:
            if targets.shape != (n, 3) or not torch.isfinite(targets).all() or core.shape != (n,) or core.dtype != torch.bool:
                raise ValueError('Invalid saved fitting targets/core')
            for i in range(k):
                selected = targets[(ids == i) & core]
                if len(selected) < 3 or torch.linalg.matrix_rank(selected-selected.mean(0)) < 2:
                    raise ValueError(f'Block {body_names[i]}: invalid saved fitting core')
        self.register_buffer('fit_targets', targets)
        self.register_buffer('fit_core', core)

    def forward(self, coords):
        if coords.ndim not in (2,3) or coords.shape[-2:] != (len(self.atom_keys),3):
            raise ValueError('Coordinates must be [N,3] or [B,N,3] in verified atom order')
        if not coords.is_floating_point(): raise ValueError('Coordinates must be floating point')
        rot = self.rotations[self.atom_body_id].to(coords)
        tr = self.translations[self.atom_body_id].to(coords)
        return torch.einsum('...ni,nij->...nj',coords,rot)+tr

    def export_checkpoint(self, handoff_raw=None):
        state = dict(format=self.VERSION,mode='blocks',convention='row_vectors_angstrom',
                    rotations=self.rotations.detach().cpu(),translations=self.translations.detach().cpu(),
                    atom_body_id=self.atom_body_id.cpu(),atom_keys=self.atom_keys,body_names=self.body_names,
                    initial_raw=self.initial_raw.cpu(),report=self.report,
                    handoff_raw=(self.handoff_raw if handoff_raw is None else handoff_raw).detach().cpu())
        if self.fit_targets is not None:
            state.update(fit_targets=self.fit_targets.cpu(), fit_core=self.fit_core.cpu())
        return copy.deepcopy(state)

    @classmethod
    def from_checkpoint(cls, state, template, device):
        if state.get('format') != cls.VERSION or state.get('convention') != 'row_vectors_angstrom':
            raise ValueError('Unsupported coordinate transform checkpoint')
        if template_atom_keys(template) != state['atom_keys']:
            raise ValueError('Block checkpoint/template atom identity or ORDER changed')
        obj = cls(**{k:state[k] for k in ('rotations','translations','atom_body_id','atom_keys',
                                         'body_names','initial_raw','report','handoff_raw')},
                  fit_targets=state.get('fit_targets'), fit_core=state.get('fit_core'))
        return obj.to(device)

    @torch.no_grad()
    def update_from_coordinates(self, raw, threshold=2.5):
        """Independently replace triggered R/t pairs, using one shared threshold.

        Fits are detached. Replacing buffers (rather than mutating them) preserves
        transforms retained by an earlier autograd graph.
        """
        validate_update_threshold(threshold)
        if self.fit_targets is None:
            raise ValueError('Old block checkpoint lacks fitting targets/core; use fixed transforms or initialize a new alignment run')
        if raw.shape != self.initial_raw.shape or not torch.isfinite(raw).all():
            raise ValueError('Invalid coordinates for block alignment update')
        candidates, shifts, events = [], [], []
        for i, name in enumerate(self.body_names):
            selected = (self.atom_body_id == i) & self.fit_core
            r, t = fit_rigid_row(raw[selected].double(), self.fit_targets[selected])
            trace = torch.trace(r @ self.rotations[i].double().T).item()
            triggered = trace < threshold
            candidates.append(r.to(self.rotations) if triggered else self.rotations[i])
            shifts.append(t.to(self.translations) if triggered else self.translations[i])
            if triggered:
                events.append(dict(body=name, body_id=i, rotation_trace=trace, threshold=threshold,
                                   rotation=r.tolist(), translation=t.tolist()))
        if events:
            self.rotations = torch.stack(candidates)
            self.translations = torch.stack(shifts)
            counts = self.report.setdefault('update_count_by_body', {})
            for event in events:
                counts[event['body']] = counts.get(event['body'], 0) + 1
            self.report['last_updates'] = events
        self.report['update_trace_threshold'] = threshold
        return events

    @torch.no_grad()
    def check_handoff(self, raw, tolerance=.05):
        if tolerance <= 0 or not np.isfinite(tolerance): raise ValueError('Handoff tolerance must be positive/finite')
        if raw.shape != self.handoff_raw.shape or not torch.isfinite(raw).all():
            raise ValueError('Invalid handoff decoder coordinates')
        rms = (raw-self.handoff_raw.to(raw)).square().sum(-1).mean().sqrt().item()
        if rms > tolerance:
            raise ValueError(f'Block decoder handoff differs by {rms:.6g} A (limit {tolerance:g}). '
                             'Check pair, sampler/noise, cache and atom order; do not refit blocks to hide it.')
        return rms

    @classmethod
    @torch.no_grad()
    def fit_manifest(cls, path, raw, template):
        data, ordered = read_alignment_manifest(path, template)
        keys = template_atom_keys(template)
        if raw.shape != (len(keys),3) or not torch.isfinite(raw).all():
            raise ValueError('Decoder/template atom count mismatch or nonfinite coordinates')
        names = data['body_names']
        ids = torch.tensor([a['body_id'] for a in ordered],device=raw.device)
        if any(type(a['body_id']) is not int for a in ordered) or set(ids.tolist()) != set(range(len(names))):
            raise ValueError('Invalid block assignments')
        target = torch.tensor([a['target'] for a in ordered],device=raw.device,dtype=torch.float64)
        if target.shape != raw.shape or not torch.isfinite(target).all(): raise ValueError('Invalid target coordinates')
        core = torch.tensor([a['fit_core'] for a in ordered],device=raw.device)
        if core.dtype != torch.bool: raise ValueError('fit_core must be boolean')
        rs,ts,report = [],[],[]
        for i,name in enumerate(names):
            selected = (ids==i)&core
            x,y = raw[selected].double(),target[selected]
            r,t = fit_rigid_row(x,y)
            rms = ((x@r+t-y).square().sum(-1).mean().sqrt()).item()
            report.append(dict(body=name,fit_atoms=int(selected.sum()),core_rmsd_A=rms))
            rs.append(r);ts.append(t)
        return cls(torch.stack(rs).to(raw),torch.stack(ts).to(raw),ids,keys,names,raw,
                   fit_targets=target, fit_core=core,
                   report=dict(source=str(Path(path)),bodies=report,
                               manifest_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               grouping=data.get('grouping', 'explicit_bodies'),
                               provenance=copy.deepcopy(data.get('provenance', {}))))


def validate_update_threshold(value):
    if not np.isfinite(value) or not -1 < value < 3:
        raise ValueError('--block-update-trace-threshold must be finite and between -1 and 3 (exclusive)')


def prepare_coordinate_transform(args, raw, template, saved=None):
    """Saved geometry is authoritative; an old cache defaults to the legacy global path."""
    manifest = getattr(args,'block_alignment',None)
    if saved is not None:
        if manifest:
            raise ValueError('Saved block transform already exists; omit --block-alignment when inheriting/resuming')
        obj = CoordinateTransform.from_checkpoint(saved,template,raw.device)
        obj.check_handoff(raw,getattr(args,'coordinate_handoff_tolerance',.05))
        args.coordinate_mode = 'blocks'
        return obj
    if getattr(args,'coordinate_mode','global') == 'global':
        if manifest: raise ValueError('--block-alignment requires --coordinate-mode blocks')
        return None
    if not manifest: raise ValueError('First block initialization requires --block-alignment')
    return CoordinateTransform.fit_manifest(manifest,raw,template)
