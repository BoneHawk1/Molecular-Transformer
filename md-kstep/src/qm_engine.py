"""QM calculation engine interface for PySCF and ORCA integration."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("md_kstep.qm")

try:
    from pyscf import gto, scf, dft, grad
    from pyscf.lib import param
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    LOGGER.warning("PySCF not available. Install with: pip install pyscf")


@dataclass
class QMConfig:
    """Configuration for QM calculations."""
    method: str = "HF"  # "HF", "DFT", "MP2", etc.
    basis: str = "sto-3g"  # Basis set name
    functional: str = "B3LYP"  # For DFT
    charge: int = 0
    spin: int = 0  # 2S+1, where S is spin
    scf_max_cycles: int = 100
    scf_conv_tol: float = 1e-6
    scf_level_shift: float = 0.0
    scf_diis_space: int = 8
    scf_diis_start_cycle: int = 1
    scf_damp_factor: float = 0.0
    scf_mix_alpha: float = 0.5
    scf_verbose: int = 0
    # For QM/MM
    qm_atom_indices: Optional[list[int]] = None
    mm_charges: Optional[np.ndarray] = None
    mm_positions: Optional[np.ndarray] = None
    # Caching
    cache_dir: Optional[Path] = None
    use_cache: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "QMConfig":
        """Load QM config from YAML file."""
        from utils import load_yaml
        data = load_yaml(path)
        if "cache_dir" in data:
            data["cache_dir"] = Path(data["cache_dir"])
        if "qm_atom_indices" in data:
            data["qm_atom_indices"] = np.array(data["qm_atom_indices"], dtype=np.int32)
        if "mm_charges" in data:
            data["mm_charges"] = np.array(data["mm_charges"], dtype=np.float64)
        if "mm_positions" in data:
            data["mm_positions"] = np.array(data["mm_positions"], dtype=np.float64)
        return cls(**data)


class QMEngine:
    """Interface for QM calculations using PySCF."""
    
    def __init__(self, config: QMConfig, atom_symbols: list[str]):
        """
        Initialize QM engine.
        
        Args:
            config: QM calculation configuration
            atom_symbols: List of atomic symbols (e.g., ['H', 'H', 'O'])
        """
        if not PYSCF_AVAILABLE:
            raise ImportError("PySCF is required for QM calculations. Install with: pip install pyscf")
        
        self.config = config
        self.atom_symbols = atom_symbols
        self._mol_cache: Optional[gto.Mole] = None
        self._scf_cache: Optional[scf.SCF] = None
        
    def _create_molecule(self, positions: np.ndarray) -> gto.Mole:
        """Create PySCF molecule object from positions."""
        mol = gto.Mole()
        mol.atom = [[sym, pos] for sym, pos in zip(self.atom_symbols, positions)]
        mol.basis = self.config.basis
        mol.charge = self.config.charge
        mol.spin = self.config.spin
        mol.verbose = self.config.scf_verbose
        mol.build()
        return mol
    
    def _create_scf(self, mol: gto.Mole) -> scf.SCF:
        """Create SCF object based on method."""
        if self.config.method.upper() == "HF":
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        elif self.config.method.upper() == "DFT":
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc = self.config.functional
        else:
            raise ValueError(f"Unsupported method: {self.config.method}")
        
        # Configure SCF convergence
        mf.max_cycle = self.config.scf_max_cycles
        mf.conv_tol = self.config.scf_conv_tol
        if self.config.scf_level_shift > 0:
            mf.level_shift = self.config.scf_level_shift
        if self.config.scf_diis_space > 0:
            mf.diis_space = self.config.scf_diis_space
        if self.config.scf_diis_start_cycle > 0:
            mf.diis_start_cycle = self.config.scf_diis_start_cycle
        if self.config.scf_damp_factor > 0:
            mf.damp = self.config.scf_damp_factor
        if self.config.scf_mix_alpha > 0:
            mf.mix = self.config.scf_mix_alpha
        
        return mf
    
    def _cache_key(self, positions: np.ndarray) -> str:
        """Generate cache key for positions."""
        pos_str = np.array2string(positions, precision=6, separator=',')
        return f"{self.config.method}_{self.config.basis}_{hash(pos_str)}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load calculation result from cache if available."""
        if not self.config.use_cache or self.config.cache_dir is None:
            return None
        
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with cache_file.open("r") as f:
                    data = json.load(f)
                return {
                    "energy": data["energy"],
                    "forces": np.array(data["forces"]),
                    "converged": data.get("converged", True),
                    "scf_cycles": data.get("scf_cycles", 0),
                }
            except Exception as e:
                LOGGER.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict) -> None:
        """Save calculation result to cache."""
        if not self.config.use_cache or self.config.cache_dir is None:
            return
        
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "energy": float(result["energy"]),
                "forces": result["forces"].tolist(),
                "converged": result.get("converged", True),
                "scf_cycles": result.get("scf_cycles", 0),
            }
            with cache_file.open("w") as f:
                json.dump(data, f)
        except Exception as e:
            LOGGER.warning(f"Failed to save cache: {e}")
    
    def calculate(self, positions: np.ndarray) -> Dict[str, np.ndarray | float | bool | int]:
        """
        Perform QM calculation for given nuclear positions.
        
        Args:
            positions: Nuclear positions in Angstroms [N, 3]
            
        Returns:
            Dictionary with:
                - energy: Total QM energy in Hartree
                - forces: Nuclear forces in Hartree/Bohr [N, 3]
                - converged: Whether SCF converged
                - scf_cycles: Number of SCF cycles
        """
        # Check cache
        cache_key = self._cache_key(positions)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Convert positions from nm to Angstroms (PySCF uses Angstroms)
        positions_ang = positions * 10.0  # nm to Angstrom
        
        # Create molecule
        mol = self._create_molecule(positions_ang)
        
        # Create and run SCF
        mf = self._create_scf(mol)
        energy = mf.kernel()
        
        converged = mf.converged
        scf_cycles = mf.scf_summary.get("niter", 0)
        
        if not converged:
            LOGGER.warning(f"SCF did not converge after {scf_cycles} cycles")
        
        # Calculate forces
        grad_calc = mf.nuc_grad_method()
        forces = -grad_calc.kernel()  # Negative gradient = forces
        
        # Convert forces from Hartree/Bohr to kJ/mol/nm
        # 1 Hartree/Bohr = 2625.5 kJ/mol/nm
        forces_kj_per_mol_nm = forces * 2625.5
        
        result = {
            "energy": float(energy),  # Hartree
            "forces": forces_kj_per_mol_nm.astype(np.float32),  # kJ/mol/nm
            "converged": converged,
            "scf_cycles": scf_cycles,
        }
        
        # Save to cache
        self._save_to_cache(cache_key, result)
        
        return result
    
    def calculate_energy_only(self, positions: np.ndarray) -> float:
        """Calculate energy only (faster, no forces)."""
        cache_key = self._cache_key(positions)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached["energy"]
        
        positions_ang = positions * 10.0
        mol = self._create_molecule(positions_ang)
        mf = self._create_scf(mol)
        energy = mf.kernel()
        
        return float(energy)
    
    def get_electronic_structure(self, positions: np.ndarray) -> Dict:
        """
        Get electronic structure information (orbitals, density, etc.).
        
        Returns:
            Dictionary with:
                - mo_energy: Molecular orbital energies
                - mo_coeff: Molecular orbital coefficients
                - mo_occ: Orbital occupations
                - density: Density matrix
        """
        positions_ang = positions * 10.0
        mol = self._create_molecule(positions_ang)
        mf = self._create_scf(mol)
        energy = mf.kernel()
        
        if not mf.converged:
            LOGGER.warning("SCF did not converge, electronic structure may be unreliable")
        
        return {
            "mo_energy": mf.mo_energy,
            "mo_coeff": mf.mo_coeff,
            "mo_occ": mf.mo_occ,
            "density": mf.make_rdm1(),
            "energy": float(energy),
        }


class QMMMEngine:
    """QM/MM engine with electrostatic embedding."""
    
    def __init__(self, qm_engine: QMEngine, mm_charges: np.ndarray, mm_positions: np.ndarray):
        """
        Initialize QM/MM engine.
        
        Args:
            qm_engine: QM engine for QM region
            mm_charges: MM point charges [M]
            mm_positions: MM positions in nm [M, 3]
        """
        self.qm_engine = qm_engine
        self.mm_charges = mm_charges
        self.mm_positions = mm_positions
    
    def calculate(self, qm_positions: np.ndarray) -> Dict:
        """
        Calculate QM/MM energy and forces.
        
        Args:
            qm_positions: QM region positions in nm [N, 3]
            
        Returns:
            Dictionary with QM/MM energy and forces
        """
        # For now, use simple electrostatic embedding
        # In production, this would integrate with OpenMM for MM forces
        qm_result = self.qm_engine.calculate(qm_positions)
        
        # Add MM-QM interaction energy (simplified)
        # This is a placeholder - full implementation would compute
        # electrostatic interactions between QM and MM regions
        qm_pos_ang = qm_positions * 10.0  # nm to Angstrom
        mm_pos_ang = self.mm_positions * 10.0
        
        # Simple Coulomb interaction (in atomic units)
        # E = sum_i sum_j q_i * q_j / r_ij
        interaction_energy = 0.0
        for qm_pos in qm_pos_ang:
            for mm_pos, mm_charge in zip(mm_pos_ang, self.mm_charges):
                r_vec = qm_pos - mm_pos
                r = np.linalg.norm(r_vec)
                if r > 1e-6:  # Avoid division by zero
                    # Convert charge to atomic units (elementary charge)
                    # and distance to Bohr
                    q_au = mm_charge  # Assuming charges are already in e
                    r_bohr = r / 0.529177  # Angstrom to Bohr
                    interaction_energy += q_au / r_bohr
        
        # Convert to Hartree (1 e^2/Bohr = 1 Hartree for unit charges)
        interaction_energy_hartree = interaction_energy * 0.529177  # Approximate conversion
        
        return {
            "energy": qm_result["energy"] + interaction_energy_hartree,
            "forces": qm_result["forces"],  # MM forces would be added here
            "converged": qm_result["converged"],
            "scf_cycles": qm_result["scf_cycles"],
        }

