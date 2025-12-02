"""Example QM/MM integration using ASE and k-step QM model.

This is a proof-of-concept demonstrating how to integrate the k-step QM integrator
into a QM/MM simulation using ASE's QM/MM calculator.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    from ase import Atoms, units
    from ase.calculators.qmmm import SimpleQMMM, EIQMMM
    from ase.calculators.xtb import XTB
    from ase.calculators.tip3p import TIP3P
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
except ImportError as e:
    raise ImportError("QM/MM integration requires ASE: pip install ase") from e

from utils import configure_logging, ensure_dir, load_yaml, remove_com, LOGGER

# Import model builder
SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "03_model.py"
_spec = importlib.util.spec_from_file_location("model_module", MODEL_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]
build_model_from_config = _module.build_model_from_config  # type: ignore[attr-defined]


class HybridQMMMIntegrator:
    """Hybrid QM/MM integrator using k-step ML model for QM region."""
    
    def __init__(
        self,
        atoms: Atoms,
        qm_indices: List[int],
        ml_model: torch.nn.Module,
        qm_calc,  # xTB or other QM calculator
        mm_calc,  # MM calculator for MM region
        k_steps: int = 4,
        dt_fs: float = 0.25,
        temperature_K: float = 300.0,
        friction: float = 0.002,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize hybrid QM/MM integrator.
        
        Args:
            atoms: ASE Atoms object for full system
            qm_indices: List of atom indices in QM region
            ml_model: Trained k-step ML model
            qm_calc: QM calculator (e.g., XTB)
            mm_calc: MM calculator for MM region
            k_steps: Number of micro-steps per ML jump
            dt_fs: Base timestep (femtoseconds)
            temperature_K: Temperature (Kelvin)
            friction: Langevin friction (1/fs)
            device: PyTorch device
        """
        self.atoms = atoms
        self.qm_indices = np.array(qm_indices)
        self.mm_indices = np.array([i for i in range(len(atoms)) if i not in qm_indices])
        self.ml_model = ml_model
        self.k_steps = k_steps
        self.dt_fs = dt_fs
        self.temperature_K = temperature_K
        self.friction = friction
        self.device = device
        
        # Setup QM/MM calculator
        self.qmmm_calc = SimpleQMMM(
            qm_indices=qm_indices,
            qm_calculator=qm_calc,
            mm_calculator=mm_calc,
            vacuum=False,  # Include MM charges in QM calculation
        )
        atoms.calc = self.qmmm_calc
        
        # Setup MM-only dynamics for MM region (when QM is jumping)
        self.mm_dyn = Langevin(
            atoms,
            timestep=dt_fs * units.fs,
            temperature_K=temperature_K,
            friction=friction / (1000.0 / units.fs),
        )
        
        self.step_count = 0
        
    def get_qm_state(self) -> Dict[str, np.ndarray]:
        """Extract QM region state."""
        positions = self.atoms.get_positions()[self.qm_indices] / 10.0  # Angstrom -> nm
        velocities = self.atoms.get_velocities()[self.qm_indices] * 100.0  # Angstrom/fs -> nm/ps
        masses = self.atoms.get_masses()[self.qm_indices]
        atomic_numbers = self.atoms.get_atomic_numbers()[self.qm_indices]
        
        return {
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "atomic_numbers": atomic_numbers,
        }
    
    def set_qm_state(self, positions_nm: np.ndarray, velocities_nm_per_ps: np.ndarray):
        """Update QM region state."""
        # Convert units
        positions_A = positions_nm * 10.0
        velocities_A_per_fs = velocities_nm_per_ps * 0.01
        
        # Update atoms
        full_pos = self.atoms.get_positions()
        full_vel = self.atoms.get_velocities()
        
        full_pos[self.qm_indices] = positions_A
        full_vel[self.qm_indices] = velocities_A_per_fs
        
        self.atoms.set_positions(full_pos)
        self.atoms.set_velocities(full_vel)
    
    def ml_qm_jump(self) -> bool:
        """Perform ML k-step jump for QM region."""
        qm_state = self.get_qm_state()
        
        # Prepare for ML model
        positions = torch.from_numpy(qm_state["positions"]).float().to(self.device)
        velocities = torch.from_numpy(qm_state["velocities"]).float().to(self.device)
        atom_types = torch.from_numpy(qm_state["atomic_numbers"]).long().to(self.device)
        masses = torch.from_numpy(qm_state["masses"]).float().to(self.device)
        batch_index = torch.zeros(len(positions), dtype=torch.long, device=self.device)
        
        # ML prediction
        with torch.no_grad():
            try:
                batch = {
                    "x_t": positions,
                    "v_t": velocities,
                    "atom_types": atom_types,
                    "masses": masses,
                    "batch": batch_index,
                }
                outputs = self.ml_model(batch)
                delta_pos = outputs["delta_pos"].cpu().numpy()
                delta_vel = outputs["delta_vel"].cpu().numpy()
            except Exception as e:
                LOGGER.error("ML prediction failed: %s", e)
                return False
        
        # Apply jump
        new_positions = qm_state["positions"] + delta_pos
        new_velocities = qm_state["velocities"] + delta_vel
        
        self.set_qm_state(new_positions, new_velocities)
        return True
    
    def qmmm_corrector(self, steps: int = 1):
        """Apply QM/MM corrector step."""
        # Use ASE dynamics for corrector (includes QM/MM forces)
        self.mm_dyn.run(steps)
    
    def run(self, num_macro_steps: int) -> Dict[str, np.ndarray]:
        """
        Run hybrid QM/MM dynamics.
        
        Args:
            num_macro_steps: Number of macro-steps (each = k micro-steps)
        
        Returns:
            Trajectory data
        """
        n_atoms = len(self.atoms)
        trajectory = {
            "positions": [],
            "velocities": [],
            "energies": [],
            "qm_energies": [],
            "mm_energies": [],
        }
        
        LOGGER.info("Starting hybrid QM/MM integration: %d macro-steps (k=%d)", 
                   num_macro_steps, self.k_steps)
        
        for macro_step in range(num_macro_steps):
            # Save current state
            trajectory["positions"].append(self.atoms.get_positions().copy())
            trajectory["velocities"].append(self.atoms.get_velocities().copy())
            
            try:
                # Get current energy
                energy = self.atoms.get_potential_energy()
                trajectory["energies"].append(energy)
                
                # Try to get QM/MM energy breakdown (if available)
                try:
                    qm_energy = self.qmmm_calc.qm_calculator.get_potential_energy()
                    mm_energy = energy - qm_energy
                    trajectory["qm_energies"].append(qm_energy)
                    trajectory["mm_energies"].append(mm_energy)
                except Exception:
                    trajectory["qm_energies"].append(float('nan'))
                    trajectory["mm_energies"].append(float('nan'))
            except Exception as e:
                LOGGER.warning("Energy calculation failed at step %d: %s", macro_step, e)
                trajectory["energies"].append(float('nan'))
                trajectory["qm_energies"].append(float('nan'))
                trajectory["mm_energies"].append(float('nan'))
            
            # ML k-step jump for QM region
            if not self.ml_qm_jump():
                LOGGER.error("ML jump failed at step %d", macro_step)
                break
            
            # Propagate MM region (standard dynamics)
            # Note: In a real implementation, you'd want to propagate MM region
            # in sync with QM region or use a multiple timestepping scheme
            
            # Apply QM/MM corrector
            self.qmmm_corrector(steps=1)
            
            self.step_count += 1
            
            if (macro_step + 1) % max(1, num_macro_steps // 10) == 0:
                LOGGER.info("Progress: %d/%d macro-steps", macro_step + 1, num_macro_steps)
        
        # Final state
        trajectory["positions"].append(self.atoms.get_positions().copy())
        trajectory["velocities"].append(self.atoms.get_velocities().copy())
        
        # Convert to arrays
        return {
            "positions": np.array(trajectory["positions"]),
            "velocities": np.array(trajectory["velocities"]),
            "energies": np.array(trajectory["energies"]),
            "qm_energies": np.array(trajectory["qm_energies"]),
            "mm_energies": np.array(trajectory["mm_energies"]),
        }


def create_example_system():
    """Create a simple QM/MM test system: small molecule in water box."""
    # This is a placeholder - in practice, you'd load from PDB or build properly
    # For now, just demonstrate the interface
    
    # Example: formaldehyde (QM) + 10 water molecules (MM)
    # Formaldehyde: CH2O
    qm_atoms = Atoms('CH2O', 
                     positions=[(0, 0, 0), (1.2, 0, 0), (-0.6, 1.0, 0), (0, -1.2, 0)])
    
    # Add water molecules (simplified)
    # In practice, use ASE's water generation or load from file
    water_positions = np.random.randn(30, 3) * 5.0 + 3.0  # Random waters around QM region
    water_atoms = Atoms('H2O' * 10, positions=water_positions)
    
    # Combine
    full_system = qm_atoms + water_atoms
    
    # QM region is first 4 atoms (CH2O)
    qm_indices = [0, 1, 2, 3]
    
    return full_system, qm_indices


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained k-step QM model")
    parser.add_argument("--model-config", type=Path, required=True, help="Model config YAML")
    parser.add_argument("--steps", type=int, default=100, help="Number of macro-steps")
    parser.add_argument("--k-steps", type=int, default=4, help="k-step horizon")
    parser.add_argument("--out", type=Path, required=True, help="Output trajectory npz")
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    return parser


def main() -> None:
    """Example QM/MM integration."""
    args = build_argparser().parse_args()
    configure_logging()
    
    # Load ML model
    model_cfg = load_yaml(args.model_config)
    device = torch.device(args.device)
    
    model = build_model_from_config(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(device)
    model.eval()
    
    LOGGER.info("Loaded k-step model from %s", args.checkpoint)
    
    # Create test system
    LOGGER.info("Creating example QM/MM system")
    atoms, qm_indices = create_example_system()
    
    # Setup calculators
    qm_calc = XTB(method='GFN2-xTB', charge=0)
    mm_calc = TIP3P()  # Simple water model for MM region
    
    # Create hybrid integrator
    integrator = HybridQMMMIntegrator(
        atoms=atoms,
        qm_indices=qm_indices,
        ml_model=model,
        qm_calc=qm_calc,
        mm_calc=mm_calc,
        k_steps=args.k_steps,
        dt_fs=0.25,
        temperature_K=300.0,
        device=device,
    )
    
    # Run dynamics
    LOGGER.info("Running hybrid QM/MM dynamics")
    trajectory = integrator.run(args.steps)
    
    # Save trajectory
    ensure_dir(Path(args.out).parent)
    np.savez(
        args.out,
        positions=trajectory["positions"],
        velocities=trajectory["velocities"],
        energies=trajectory["energies"],
        qm_energies=trajectory["qm_energies"],
        mm_energies=trajectory["mm_energies"],
        qm_indices=qm_indices,
        k_steps=args.k_steps,
    )
    
    LOGGER.info("Saved QM/MM trajectory to %s", args.out)
    LOGGER.info("QM/MM integration complete!")
    LOGGER.info("NOTE: This is a proof-of-concept. Production use requires:")
    LOGGER.info("  - Proper system setup (PDB, topology)")
    LOGGER.info("  - Careful QM/MM boundary treatment")
    LOGGER.info("  - Multiple timestepping for MM region")
    LOGGER.info("  - Extensive validation")


if __name__ == "__main__":
    main()

