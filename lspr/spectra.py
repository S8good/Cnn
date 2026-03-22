import json
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np


def default_wavelengths(start_nm: float = 400.0, stop_nm: float = 800.0, points: int = 400) -> np.ndarray:
    return np.linspace(start_nm, stop_nm, points, dtype=np.float64)


def _load_optical_constants_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if not {"wavelength_nm", "n", "k"}.issubset(set(data.dtype.names or [])):
        raise ValueError("CSV must contain columns: wavelength_nm,n,k")
    wl = np.asarray(data["wavelength_nm"], dtype=np.float64)
    n = np.asarray(data["n"], dtype=np.float64)
    k = np.asarray(data["k"], dtype=np.float64)
    return wl, n, k


def get_gold_refractive_index(
    wavelengths_nm: np.ndarray,
    data_path: Optional[str] = None,
    fallback_n: float = 0.2,
    fallback_k: float = 3.3,
) -> np.ndarray:
    """
    Returns complex refractive index for gold at given wavelengths.
    If data_path is provided and exists, it interpolates the CSV data.
    Otherwise it returns a constant fallback value.
    """
    if data_path and os.path.isfile(data_path):
        wl_data, n_data, k_data = _load_optical_constants_csv(data_path)
        n_interp = np.interp(wavelengths_nm, wl_data, n_data)
        k_interp = np.interp(wavelengths_nm, wl_data, k_data)
        return n_interp + 1j * k_interp

    # Fallback: constant n,k (replace with real data for better physics)
    n_const = np.full_like(wavelengths_nm, fallback_n, dtype=np.float64)
    k_const = np.full_like(wavelengths_nm, fallback_k, dtype=np.float64)
    return n_const + 1j * k_const


def simulate_extinction_spectrum(
    wavelengths_nm: np.ndarray,
    n_medium: float,
    diameter_nm: float,
    gold_nk: Optional[np.ndarray] = None,
    gold_nk_data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Simulate the extinction spectrum using PyMieScatt.MieQ.
    Returns an array of Qext values for each wavelength.
    """
    try:
        import PyMieScatt as pms
    except ImportError as exc:
        raise ImportError("PyMieScatt is required. Please install it in your environment.") from exc

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    if gold_nk is None:
        gold_nk = get_gold_refractive_index(wavelengths_nm, data_path=gold_nk_data_path)

    qext = np.zeros_like(wavelengths_nm, dtype=np.float64)
    for i, wl in enumerate(wavelengths_nm):
        m_rel = gold_nk[i] / n_medium
        # MieQ returns (Qext, Qsca, Qabs, g, Qpr, Qback, Qratio)
        qext[i] = pms.MieQ(m_rel, wl, diameter_nm, nMedium=n_medium)[0]

    return qext


def generate_base_spectra_grid(
    n_values: Iterable[float],
    d_values: Iterable[float],
    wavelengths_nm: np.ndarray,
    gold_nk_data_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Generates base spectra for all (n, d) combinations.
    Returns spectra array of shape [num_classes, num_wavelengths]
    and a list mapping class index -> (n, d).
    """
    n_list = list(n_values)
    d_list = list(d_values)
    gold_nk = get_gold_refractive_index(wavelengths_nm, data_path=gold_nk_data_path)

    spectra = []
    class_map = []
    for n_medium in n_list:
        for diameter_nm in d_list:
            spec = simulate_extinction_spectrum(
                wavelengths_nm,
                n_medium=n_medium,
                diameter_nm=diameter_nm,
                gold_nk=gold_nk,
                gold_nk_data_path=None,
            )
            spectra.append(spec)
            class_map.append((n_medium, diameter_nm))

    return np.asarray(spectra, dtype=np.float64), class_map


def save_grid_metadata(
    path: str,
    wavelengths_nm: np.ndarray,
    n_values: Iterable[float],
    d_values: Iterable[float],
    class_map: List[Tuple[float, float]],
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "wavelengths_nm": [float(x) for x in wavelengths_nm],
        "n_values": [float(x) for x in n_values],
        "d_values": [float(x) for x in d_values],
        "class_map": [{"n": float(n), "d_nm": float(d)} for n, d in class_map],
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
