"""Build a portable atom-identity-based block manifest from an assembled model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from coordinate_transform import template_atom_keys, validate_alignment_manifest
from structure_io import read_template


def build(target, atom_map, output):
    keys = template_atom_keys(target)
    coordinates, _, _ = read_template(target)
    if len(keys) != len(coordinates): raise ValueError('Target reader/order mismatch')
    rows = list(csv.DictReader(Path(atom_map).open(encoding='utf-8-sig')))
    by_id = {(r['chain'],int(r['residue_number']),r['insertion_code'],r['atom_name']):r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {tuple(k[:4]) for k in keys}:
        raise ValueError('Target must contain exactly the split source atoms, with the same author IDs')
    names = sorted({r['six_body_file'] for r in rows})
    atoms = []
    for key, point in zip(keys,coordinates.tolist()):
        row=by_id[tuple(key[:4])]
        atoms.append(dict(key=key,body_id=names.index(row['six_body_file']),target=point,
                          fit_core=(key[3]=='CA' and row['suggested_fit_core']=='True')))
    result=dict(format='cocofold2-block-manifest-v1',body_names=names,atoms=atoms,
                provenance=dict(target_name=Path(target).name,atom_map_name=Path(atom_map).name,
                    target_sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest(),
                    atom_map_sha256=hashlib.sha256(Path(atom_map).read_bytes()).hexdigest(),
                    note='Reference-assisted rigid assembly. Hinge is retained, not rebuilt.'))
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
    print(f'Saved {len(atoms)} atoms / {len(names)} blocks: {output}')
    return result


def build_chains(target, output, fit_atoms='ca'):
    """One alignment body per author chain; retain all atoms in reference order."""
    if fit_atoms not in ('ca', 'all'):
        raise ValueError('fit_atoms must be ca or all')
    keys = template_atom_keys(target)
    coordinates, _, _ = read_template(target)
    if len(keys) != len(coordinates):
        raise ValueError('Target reader/order mismatch')
    names = list(dict.fromkeys(k[0] for k in keys))
    result = dict(format='cocofold2-block-manifest-v1', grouping='auth_asym_id',
                  body_names=names, atoms=[dict(key=k, body_id=names.index(k[0]), target=point,
                    fit_core=fit_atoms == 'all' or k[3] == 'CA') for k, point in zip(keys, coordinates.tolist())],
                  provenance=dict(target_name=Path(target).name,
                    target_sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest(),
                    fit_atoms=fit_atoms, note='Reference-assisted per-chain rigid alignment; optimization groups are independent.'))
    validate_alignment_manifest(result, keys)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, separators=(',', ':'))
        handle.write('\n')
    print(f'Saved {len(keys)} atoms / {len(names)} chains: {output}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True,help='Assembled CIF/PDB retaining source atom identities')
    grouping = p.add_mutually_exclusive_group(required=True)
    grouping.add_argument('--atom-map',help='Split atom_mapping.csv with six_body_file/suggested_fit_core')
    grouping.add_argument('--by-chain',action='store_true',help='One alignment body per author chain; one whole cache can optimize all bodies')
    p.add_argument('--fit-atoms',choices=('ca','all'),default=None,help='For --by-chain: fit C-alpha atoms (default ca) or all atoms; retain all atoms in either case')
    p.add_argument('--output',required=True, help='New alignment manifest JSON path; parent directory is created if needed. Default: %(default)s.')
    args=p.parse_args()
    if args.by_chain:
        build_chains(args.target,args.output,args.fit_atoms or 'ca')
    else:
        if args.fit_atoms is not None:
            p.error('--fit-atoms requires --by-chain; the atom map defines the legacy fitting core')
        build(args.target,args.atom_map,args.output)


if __name__=='__main__':
    main()
