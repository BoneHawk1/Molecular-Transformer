from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

from utils import ensure_dir, get_logger


def read_smiles(smiles_path: Path) -> List[str]:
    smiles = []
    with open(smiles_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            smiles.append(s)
    return smiles


def rdkit_embed_minimize(smiles: str) -> Chem.Mol:
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    rdmol = Chem.AddHs(rdmol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ok = AllChem.EmbedMolecule(rdmol, params)
    if ok != 0:
        # Try once more with random coordinates
        ok = AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
        if ok != 0:
            raise RuntimeError(f"Failed to embed molecule: {smiles}")
    AllChem.MMFFOptimizeMolecule(rdmol)
    return rdmol


def write_outputs(offmol: Molecule, rdmol: Chem.Mol, out_dir: Path, name: str) -> None:
    from openmm import app
    from openff.toolkit.utils import get_data_file_path

    pdb_path = out_dir / f"{name}.pdb"
    with Chem.SDWriter(str(out_dir / f"{name}.sdf")) as w:
        w.write(rdmol)
    Chem.MolToPDBFile(rdmol, str(pdb_path))

    # Save topology via OpenMM PDBFile too (redundant but handy)
    pdb = app.PDBFile(str(pdb_path))
    np.savez_compressed(out_dir / f"{name}_meta.npz", num_atoms=pdb.topology.getNumAtoms())


def main():
    parser = argparse.ArgumentParser(description="Prepare 3D molecules and metadata")
    parser.add_argument("--smiles", type=str, required=True, help="Path to molecules.smi")
    parser.add_argument("--out", type=str, required=True, help="Output directory for PDB/SDF and metadata")
    args = parser.parse_args()

    smiles_path = Path(args.smiles)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    logger = get_logger()
    ff = ForceField("openff_unconstrained-2.2.1.offxml")

    for i, smi in enumerate(read_smiles(smiles_path)):
        name = f"mol_{i:03d}"
        logger.info(f"Embedding and minimizing {name}: {smi}")
        rdmol = rdkit_embed_minimize(smi)

        # OpenFF molecule for downstream param / masses
        offmol = Molecule.from_rdkit(rdmol, allow_unsupported_conformers=True)

        write_outputs(offmol, rdmol, out_dir, name)

    logger.info("Done.")


if __name__ == "__main__":
    main()

