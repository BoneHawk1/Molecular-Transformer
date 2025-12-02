"""ASE calculator wrapper for PySCF with optional GPU4PySCF acceleration."""
from __future__ import annotations

import logging
from typing import Any, Iterable, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pyscf import dft, gto, scf

try:
    from gpu4pyscf import dft as g4dft
    from gpu4pyscf import scf as g4scf

    GPU4PYSCF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GPU4PYSCF_AVAILABLE = False
LOGGER = logging.getLogger(__name__)


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


class PySCFGPUCalculator(Calculator):
    """PySCF calculator for ASE with optional GPU acceleration.

    Parameters
    ----------
    method:
        Quantum chemistry method (``"rhf"``, ``"uhf"``, ``"rks"``, ``"uks"``).
    basis:
        Basis set string (e.g., ``"sto-3g"``, ``"6-31g*"``).
    xc:
        Exchange-correlation functional (for DFT methods).
    charge:
        Total charge of the system.
    spin:
        2 * S (number of unpaired electrons). Use 0 for closed-shell.
    use_gpu:
        Attempt to use GPU4PySCF if installed. Falls back to CPU PySCF otherwise.
    conv_tol:
        SCF convergence tolerance in Hartree.
    max_cycle:
        Maximum number of SCF iterations.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        method: str = "rhf",
        basis: str = "sto-3g",
        xc: str = "pbe",
        charge: int = 0,
        spin: int = 0,
        use_gpu: bool = True,
        conv_tol: float = 1e-8,
        max_cycle: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.method = method.lower()
        self.basis = basis
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.use_gpu = use_gpu and GPU4PYSCF_AVAILABLE
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle

        if use_gpu and not GPU4PYSCF_AVAILABLE:
            LOGGER.warning("GPU4PySCF not available; running PySCF on CPU")

        if self.method not in {"rhf", "uhf", "rks", "uks"}:
            raise ValueError(f"Unsupported PySCF method: {method}")

    def calculate(self, atoms: Atoms = None, properties: Iterable[str] = ("energy",), system_changes: Tuple[str, ...] = all_changes) -> None:  # type: ignore[override]
        Calculator.calculate(self, atoms, properties, system_changes)

        mol = self._build_molecule(self.atoms)
        mf = self._build_mean_field(mol)

        energy_hartree = mf.kernel()
        if not mf.converged:
            LOGGER.warning("PySCF SCF did not converge; results may be inaccurate")

        grad_method = mf.nuc_grad_method()
        grad_hartree_bohr = grad_method.kernel()

        forces = self._grad_to_forces(grad_hartree_bohr)
        energy = energy_hartree * HARTREE_TO_EV

        self.results = {
            "energy": energy,
            "forces": forces,
        }

    def _build_molecule(self, atoms: Atoms) -> gto.Mole:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()  # Angstrom

        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = list(zip(symbols, positions))
        mol.unit = "Angstrom"
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = self.spin
        mol.build()
        return mol

    def _build_mean_field(self, mol: gto.Mole):
        if self.method in {"rhf", "uhf"}:
            if self.use_gpu:
                mf_cls = g4scf.RHF if self.method == "rhf" else g4scf.UHF
            else:
                mf_cls = scf.RHF if self.method == "rhf" else scf.UHF
        else:  # RKS / UKS
            if self.use_gpu:
                mf_cls = g4dft.RKS if self.method == "rks" else g4dft.UKS
            else:
                mf_cls = dft.RKS if self.method == "rks" else dft.UKS

        mf = mf_cls(mol)
        if self.method in {"rks", "uks"}:
            mf.xc = self.xc

        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        return mf

    def _grad_to_forces(self, grad_hartree_bohr: np.ndarray) -> np.ndarray:
        """Convert nuclear gradients (Hartree/Bohr) to ASE forces (eV/Angstrom)."""
        factor = HARTREE_TO_EV / BOHR_TO_ANGSTROM
        # Force is negative gradient.
        return -grad_hartree_bohr * factor


