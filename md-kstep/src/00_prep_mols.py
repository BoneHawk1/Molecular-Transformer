"""Generate 3D conformers and OpenMM-ready parameters for a set of SMILES."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import Vec3, XmlSerializer, unit
from openmm.app import PDBFile

from utils import configure_logging, read_smiles, ensure_dir, set_seed, LOGGER


def _embed_molecule(smiles: str, embed_seed: int, max_attempts: int) -> Chem.Mol:
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    rdmol = Chem.AddHs(rdmol)
    params = AllChem.ETKDGv3()
    params.randomSeed = embed_seed
    params.maxAttempts = max_attempts
    if AllChem.EmbedMolecule(rdmol, params) != 0:
        raise RuntimeError(f"Embedding failed for {smiles}")
    if AllChem.MMFFOptimizeMolecule(rdmol, maxIters=1024) != 0:
        LOGGER.warning("MMFF optimization did not fully converge for %s", smiles)
    return rdmol


def _rdmol_to_openmm_positions(rdmol: Chem.Mol) -> Tuple[unit.Quantity, int]:
    conf = rdmol.GetConformer()
    num_atoms = rdmol.GetNumAtoms()
    positions = []
    for idx in range(num_atoms):
        pos = conf.GetAtomPosition(idx)
        positions.append(Vec3(pos.x * 0.1, pos.y * 0.1, pos.z * 0.1))
    return unit.Quantity(positions, unit.nanometer), num_atoms


def _parameterize(off_mol: Molecule, ff: ForceField):
    topology = off_mol.to_topology().to_openmm()
    system = ff.create_openmm_system(off_mol.to_topology())
    return topology, system


def _write_outputs(smiles: str, out_dir: Path, topology, system, positions) -> Dict:
    sanitized = smiles.replace("/", "_").replace("\\", "_").replace(" ", "")
    mol_dir = out_dir / sanitized
    ensure_dir(mol_dir)
    pdb_path = mol_dir / "structure.pdb"
    xml_path = mol_dir / "forcefield.xml"

    with pdb_path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, positions, handle)

    with xml_path.open("w", encoding="utf-8") as handle:
        handle.write(XmlSerializer.serialize(system))

    return {
        "smiles": smiles,
        "pdb": str(pdb_path),
        "xml": str(xml_path),
        "num_atoms": topology.getNumAtoms(),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", type=Path, required=True, help="Path to SMILES list")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for raw files")
    parser.add_argument("--force-field", default="openff_unconstrained-2.2.1.offxml", help="OpenFF force field")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-attempts", type=int, default=200, help="Max embedding attempts per molecule")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    configure_logging()
    set_seed(args.seed)

    smiles_list = read_smiles(args.smiles)
    if not smiles_list:
        raise RuntimeError("SMILES list is empty")

    force_field = ForceField(args.force_field)
    ensure_dir(args.out_dir)

    summary = {}
    for idx, smiles in enumerate(smiles_list):
        LOGGER.info("[%d/%d] Processing %s", idx + 1, len(smiles_list), smiles)
        rdmol = _embed_molecule(smiles, args.seed + idx, args.max_attempts)
        off_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
        topology, system = _parameterize(off_mol, force_field)
        positions, _ = _rdmol_to_openmm_positions(rdmol)
        summary[smiles] = _write_outputs(smiles, args.out_dir, topology, system, positions)

    metadata_path = args.out_dir / "metadata.json"
    LOGGER.info("Writing metadata to %s", metadata_path)
    import json

    metadata = {
        "force_field": args.force_field,
        "seed": args.seed,
        "molecules": summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
