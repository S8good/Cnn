"""LSPR Meta-Baseline utilities."""

# Keep __init__ lightweight to avoid importing heavy deps (e.g. torch) for data generation.
from .spectra import default_wavelengths, generate_base_spectra_grid, simulate_extinction_spectrum
from .noise import (
    add_baseline_drift,
    add_fwhm_broadening,
    add_gaussian_noise,
    apply_noise_pipeline,
)

__all__ = [
    "default_wavelengths",
    "generate_base_spectra_grid",
    "simulate_extinction_spectrum",
    "add_baseline_drift",
    "add_fwhm_broadening",
    "add_gaussian_noise",
    "apply_noise_pipeline",
]
