import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _long_path(path: Path) -> str:
    p = str(path)
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + p
    return p


def _safe_save_npy(path: Path, array) -> None:
    try:
        np.save(str(path), array)
        return
    except OSError as exc:
        if os.name != "nt" or exc.errno != 22:
            raise
        np.save(_long_path(path), array)

import numpy as np

from lspr.noise import apply_noise_pipeline
from lspr.spectra import (
    default_wavelengths,
    generate_base_spectra_grid,
    save_grid_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LSPR pretraining spectra dataset")
    parser.add_argument("--out-dir", type=str, default="data/pretrain", help="Output directory")
    parser.add_argument("--wavelength-start", type=float, default=400.0)
    parser.add_argument("--wavelength-stop", type=float, default=800.0)
    parser.add_argument("--wavelength-points", type=int, default=400)
    parser.add_argument("--n-min", type=float, default=1.33)
    parser.add_argument("--n-max", type=float, default=1.40)
    parser.add_argument("--n-step", type=float, default=0.01)
    parser.add_argument("--d-min", type=float, default=30.0)
    parser.add_argument("--d-max", type=float, default=60.0)
    parser.add_argument("--d-step", type=float, default=2.0)
    parser.add_argument("--variants-per-class", type=int, default=50)
    parser.add_argument("--sigma-frac", type=float, default=0.01)
    parser.add_argument("--drift-frac", type=float, default=0.02)
    parser.add_argument("--fwhm-min", type=float, default=2.0)
    parser.add_argument("--fwhm-max", type=float, default=10.0)
    parser.add_argument("--gold-nk-csv", type=str, default="data/au_johnson_nk.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = default_wavelengths(
        start_nm=args.wavelength_start,
        stop_nm=args.wavelength_stop,
        points=args.wavelength_points,
    )
    n_values = np.round(np.arange(args.n_min, args.n_max + 1e-9, args.n_step), 3).tolist()
    d_values = np.round(np.arange(args.d_min, args.d_max + 1e-9, args.d_step), 3).tolist()

    gold_nk_csv = args.gold_nk_csv if args.gold_nk_csv else None
    base_spectra, class_map = generate_base_spectra_grid(
        n_values,
        d_values,
        wavelengths_nm=wavelengths,
        gold_nk_data_path=gold_nk_csv,
    )

    num_classes = len(class_map)
    total_samples = num_classes * args.variants_per_class
    spectra_path = (out_dir / "pretrain_spectra.npy").resolve()
    labels_path = (out_dir / "pretrain_labels.npy").resolve()

    spectra_mm = None
    labels_mm = None
    try:
        spectra_mm = np.lib.format.open_memmap(
            str(spectra_path), mode="w+", dtype="float32", shape=(total_samples, wavelengths.size)
        )
        labels_mm = np.lib.format.open_memmap(
            str(labels_path), mode="w+", dtype="int64", shape=(total_samples,)
        )
    except OSError as exc:
        try:
            spectra_mm = np.lib.format.open_memmap(
                _long_path(spectra_path),
                mode="w+",
                dtype="float32",
                shape=(total_samples, wavelengths.size),
            )
            labels_mm = np.lib.format.open_memmap(
                _long_path(labels_path),
                mode="w+",
                dtype="int64",
                shape=(total_samples,),
            )
        except OSError:
            print(f"WARNING: open_memmap failed ({exc}). Falling back to in-memory arrays.")
            spectra_mm = np.zeros((total_samples, wavelengths.size), dtype=np.float32)
            labels_mm = np.zeros((total_samples,), dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    step_nm = float(wavelengths[1] - wavelengths[0])

    idx = 0
    for class_idx, base in enumerate(base_spectra):
        for _ in range(args.variants_per_class):
            noisy = apply_noise_pipeline(
                base,
                wavelength_step_nm=step_nm,
                sigma_frac=args.sigma_frac,
                drift_frac=args.drift_frac,
                fwhm_range_nm=(args.fwhm_min, args.fwhm_max),
                rng=rng,
            )
            spectra_mm[idx] = noisy.astype(np.float32)
            labels_mm[idx] = class_idx
            idx += 1
        if (class_idx + 1) % 8 == 0:
            print(f"Generated classes: {class_idx + 1}/{num_classes}")

    if hasattr(spectra_mm, "flush"):
        spectra_mm.flush()
    if hasattr(labels_mm, "flush"):
        labels_mm.flush()
    if not isinstance(spectra_mm, np.memmap):
        _safe_save_npy(spectra_path, spectra_mm)
    if not isinstance(labels_mm, np.memmap):
        _safe_save_npy(labels_path, labels_mm)

    save_grid_metadata(
        str(out_dir / "grid_meta.json"),
        wavelengths_nm=wavelengths,
        n_values=n_values,
        d_values=d_values,
        class_map=class_map,
        extra={
            "variants_per_class": args.variants_per_class,
            "sigma_frac": args.sigma_frac,
            "drift_frac": args.drift_frac,
            "fwhm_range_nm": [args.fwhm_min, args.fwhm_max],
            "gold_nk_csv": gold_nk_csv or "",
        },
    )

    print("Done.")
    print(f"Spectra: {spectra_path}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()
