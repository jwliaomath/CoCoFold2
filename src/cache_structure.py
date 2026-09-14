"""Structure export from explicit Protenix v1.0.2 cache features.

No coordinates or residue identities are inferred from learned representations.
The template-free path supports unmodified standard proteins only. Encoding:
protenix/data/core/featurizer.py and protenix/data/constants.py at v1.0.2.
"""
from pathlib import Path

import gemmi
import numpy as np


PROTEIN_NAMES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)


def _array(features, key, tail):
    if key not in features:
        raise ValueError(
            f"Cache lacks {key}; re-split the original full cache with the updated "
            "contextual_cache.py, or supply --cif_path."
        )
    value = features[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy cannot represent bfloat16; preserve integer IDs without rounding.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    value = np.asarray(value)
    # Accept an unbatched cache or singleton leading batch dimensions only.
    while value.ndim > len(tail) and value.shape[0] == 1:
        value = value[0]
    if value.shape != tuple(tail) or not np.isfinite(value).all():
        raise ValueError(f"Invalid {key}: expected finite {tail}, got {value.shape}")
    return value


def _integers(features, key, size):
    value = _array(features, key, (size,))
    if not np.equal(value, np.floor(value)).all():
        raise ValueError(f"Noninteger {key}")
    return value.astype(np.int64)


def _onehot(features, key, shape):
    value = _array(features, key, shape)
    if not np.isin(value, [0, 1]).all() or not (value.sum(axis=-1) == 1).all():
        raise ValueError(f"{key} must contain exactly one active category per row")
    return value.argmax(axis=-1)


def chain_name(asym_id):
    """Protenix zero-based asym_id -> A..Z, AA..AZ, ...; never renumber a split."""
    if asym_id < 0:
        raise ValueError("Negative asym_id")
    value, result = int(asym_id) + 1, ""
    while value:
        value, digit = divmod(value - 1, 26)
        result = chr(65 + digit) + result
    return result


def structure_from_cache(features, name="prediction"):
    """Return a zero-coordinate topology and its chain mapping report."""
    if "asym_id" not in features or "atom_to_token_idx" not in features:
        raise ValueError("Cache lacks chain or atom-to-token mapping")
    nt = int(features["asym_id"].shape[-1])
    na = int(features["atom_to_token_idx"].shape[-1])
    if not nt or not na:
        raise ValueError("Empty topology")
    asym = _integers(features, "asym_id", nt)
    entity = _integers(features, "entity_id", nt)
    residue = _integers(features, "residue_index", nt)
    mapping = _integers(features, "atom_to_token_idx", na)
    if (entity < 0).any() or (residue < 1).any():
        raise ValueError("Expected nonnegative entity_id and positive source residue_index")
    if mapping.min() < 0 or mapping.max() >= nt or len(np.unique(mapping)) != nt:
        raise ValueError("Invalid/incomplete atom-to-token mapping")
    # Standard protein tokens must each be one contiguous residue, in cache order.
    runs = mapping[np.r_[True, mapping[1:] != mapping[:-1]]]
    if not np.array_equal(runs, np.arange(nt)):
        raise ValueError("Noncontiguous or reordered protein tokens; use an explicit template")
    restype = _onehot(features, "restype", (nt, 32))
    if (restype >= len(PROTEIN_NAMES)).any():
        raise ValueError("Template-free export requires standard amino acids; use --cif_path for UNK/nucleic acids")
    for key, expected in (
        ("is_protein", 1), ("is_ligand", 0), ("is_dna", 0),
        ("is_rna", 0), ("modified_res_mask", 0),
    ):
        if not (_integers(features, key, na) == expected).all():
            raise ValueError(f"Unsupported {key}: template-free export requires unmodified protein")
    chars = _onehot(features, "ref_atom_name_chars", (na, 4, 64))
    names = ["".join(chr(int(c) + 32) for c in row).rstrip() for row in chars]
    elements = _onehot(features, "ref_element", (na, 128)) + 1
    if (elements > 118).any():
        raise ValueError("Unknown atomic element in cache")

    structure = gemmi.Structure()
    structure.name = name
    structure.add_model(gemmi.Model("1"))
    model = structure[0]
    seen_chains, seen_residues, seen_atoms = set(), set(), set()
    chain_report, entity_sequences = [], {}
    previous_chain, previous_token = None, None
    for i, token in enumerate(mapping):
        aid, eid, rid = int(asym[token]), int(entity[token]), int(residue[token])
        cid = chain_name(aid)
        if cid != previous_chain:
            if cid in seen_chains:
                raise ValueError("Noncontiguous chain atoms; cannot preserve order")
            seen_chains.add(cid)
            model.add_chain(gemmi.Chain(cid))
            chain = model[-1]
            chain_report.append(dict(asym_id=aid, chain_id=cid, entity_id=eid,
                                     n_residue=0, n_atom=0))
            previous_chain = cid
        if eid != chain_report[-1]["entity_id"]:
            raise ValueError("A chain contains more than one entity")
        if token != previous_token:
            if (cid, rid) in seen_residues:
                raise ValueError("Duplicate residue identity in cache")
            seen_residues.add((cid, rid))
            res = gemmi.Residue()
            res.name = PROTEIN_NAMES[int(restype[token])]
            res.seqid = gemmi.SeqId(rid, " ")
            res.label_seq = rid
            res.subchain = cid
            res.entity_id = str(eid + 1)
            res.entity_type = gemmi.EntityType.Polymer
            res.het_flag = "A"
            chain.add_residue(res)
            current_residue = chain[-1]
            chain_report[-1]["n_residue"] += 1
            previous_token = token
        atom_name = names[i]
        if not atom_name or " " in atom_name or (cid, rid, atom_name) in seen_atoms:
            raise ValueError("Empty/invalid/duplicate atom name in cache")
        seen_atoms.add((cid, rid, atom_name))
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(int(elements[i]))
        atom.occ, atom.b_iso = 1.0, 0.0  # No confidence is invented.
        current_residue.add_atom(atom)
        chain_report[-1]["n_atom"] += 1
    for chain in model:
        eid = chain[0].entity_id
        sequence = [(r.seqid.num, r.name) for r in chain]
        if eid in entity_sequences and sequence != entity_sequences[eid]:
            raise ValueError("Different sequences share entity_id; supply a topology template")
        entity_sequences[eid] = sequence
    for eid, sequence in entity_sequences.items():
        ent = gemmi.Entity(eid)
        ent.entity_type = gemmi.EntityType.Polymer
        ent.polymer_type = gemmi.PolymerType.PeptideL
        # Do not fabricate missing sequence positions or mislabel non-1-based
        # residue numbering as a contiguous entity_poly_seq.
        if [rid for rid, _ in sequence] == list(range(1, len(sequence) + 1)):
            ent.full_sequence = [resname for _, resname in sequence]
        ent.subchains = [c.name for c in model if c[0].entity_id == eid]
        structure.entities.append(ent)
    report = dict(source="cache_features", encoding="protenix_v1.0.2",
                  n_token=nt, n_atom=na, chains=chain_report,
                  confidence="not supplied; occupancy=1, B-factor=0")
    return structure, report


def export_structure(structure, coordinates, output_stem, output_format):
    """Write actual CIF/PDB formats, preserving atom traversal order."""
    xyz = np.asarray(coordinates)
    atoms = [a for m in structure for c in m for r in c for a in r]
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError(f"Expected finite coordinates {(len(atoms), 3)}, got {xyz.shape}")
    if len(structure) != 1:
        raise ValueError("Export requires a single model")
    if output_format not in ("cif", "pdb", "both"):
        raise ValueError("Unknown output format")
    formats = ("cif", "pdb") if output_format == "both" else (output_format,)
    if "pdb" in formats:
        if len(atoms) + sum(len(m) for m in structure) > 99999:
            raise ValueError("PDB atom/TER serial limit exceeded; use --output_format cif")
        if any(len(c.name) != 1 for c in structure[0]):
            raise ValueError("PDB requires single-character chain IDs; use --output_format cif")
        if any(not -999 <= r.seqid.num <= 9999 for c in structure[0] for r in c):
            raise ValueError("PDB residue numbering limit exceeded; use --output_format cif")
        if any(len(f"{float(v):8.3f}") > 8 for v in xyz.flat):
            raise ValueError("PDB coordinate field overflow; use --output_format cif")
    result = structure.clone()
    for atom, point in zip((a for c in result[0] for r in c for a in r), xyz):
        atom.pos = gemmi.Position(*map(float, point))
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = str(stem) + "." + fmt
        if fmt == "cif":
            result.make_mmcif_document().write_file(path)
        else:
            result.write_pdb(path)
        paths.append(path)
    return paths
