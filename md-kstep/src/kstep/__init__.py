"""Core library for the md-kstep hybrid integrator.

The numbered scripts in ``src/`` are thin CLIs around this package:

- :mod:`kstep.common`   – COM removal, logging, seeding, YAML/JSON helpers
- :mod:`kstep.geometry` – bond/angle/dihedral perception and measurement
- :mod:`kstep.model`    – E(3)-equivariant k-step models (deterministic + flow matching)
- :mod:`kstep.data`     – k-step window datasets and GPU-resident batching
- :mod:`kstep.hybrid`   – predictor/corrector loop shared by the MM and QM integrators
- :mod:`kstep.metrics`  – structural, distributional and dynamical evaluation metrics
"""

__all__ = ["common", "geometry", "model", "data", "hybrid", "metrics"]
