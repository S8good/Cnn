from typing import Optional, Tuple

import numpy as np


def add_gaussian_noise(
    spectrum: np.ndarray,
    sigma_frac: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    spectrum = np.asarray(spectrum, dtype=np.float64)
    scale = float(np.max(spectrum) - np.min(spectrum))
    if scale <= 0:
        scale = float(np.max(np.abs(spectrum))) or 1.0
    sigma = sigma_frac * scale
    noise = rng.normal(0.0, sigma, size=spectrum.shape)
    return spectrum + noise


def add_baseline_drift(
    spectrum: np.ndarray,
    drift_frac: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    spectrum = np.asarray(spectrum, dtype=np.float64)
    scale = float(np.max(spectrum) - np.min(spectrum)) or 1.0

    x = np.linspace(0.0, 1.0, spectrum.size, dtype=np.float64)
    slope = rng.uniform(-drift_frac, drift_frac) * scale
    amp = rng.uniform(0.0, drift_frac) * scale
    freq = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)

    baseline = slope * (x - 0.5) + amp * np.sin(2.0 * np.pi * freq * x + phase)
    return spectrum + baseline


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def add_fwhm_broadening(
    spectrum: np.ndarray,
    wavelength_step_nm: float,
    fwhm_range_nm: Tuple[float, float] = (2.0, 10.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    spectrum = np.asarray(spectrum, dtype=np.float64)

    fwhm_nm = rng.uniform(fwhm_range_nm[0], fwhm_range_nm[1])
    sigma_nm = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pts = sigma_nm / max(wavelength_step_nm, 1e-6)
    kernel_size = int(np.ceil(sigma_pts * 6.0)) | 1
    kernel = _gaussian_kernel(kernel_size, sigma_pts)
    return np.convolve(spectrum, kernel, mode="same")


def apply_noise_pipeline(
    spectrum: np.ndarray,
    wavelength_step_nm: float,
    sigma_frac: float = 0.01,
    drift_frac: float = 0.02,
    fwhm_range_nm: Tuple[float, float] = (2.0, 10.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = add_gaussian_noise(spectrum, sigma_frac=sigma_frac, rng=rng)
    out = add_baseline_drift(out, drift_frac=drift_frac, rng=rng)
    out = add_fwhm_broadening(
        out,
        wavelength_step_nm=wavelength_step_nm,
        fwhm_range_nm=fwhm_range_nm,
        rng=rng,
    )
    return out
