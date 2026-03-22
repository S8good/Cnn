import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot random spectra samples for sanity check")
    parser.add_argument("--data-dir", type=str, default="data/pretrain_small")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--class-idx", type=int, default=-1, help="If set >=0, sample only this class")
    parser.add_argument("--out", type=str, default="outputs/random_spectra_check.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    spectra_path = data_dir / "pretrain_spectra.npy"
    labels_path = data_dir / "pretrain_labels.npy"
    meta_path = data_dir / "grid_meta.json"

    spectra = np.load(spectra_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        wavelengths = np.asarray(meta.get("wavelengths_nm", []), dtype=np.float64)
        class_map = meta.get("class_map", [])
    else:
        wavelengths = np.arange(spectra.shape[1], dtype=np.float64)
        class_map = []

    rng = np.random.default_rng(args.seed)

    if args.class_idx >= 0:
        mask = labels == args.class_idx
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            raise ValueError(f"No samples for class_idx={args.class_idx}")
        chosen = rng.choice(candidates, size=min(args.num_samples, candidates.size), replace=False)
    else:
        chosen = rng.choice(spectra.shape[0], size=min(args.num_samples, spectra.shape[0]), replace=False)

    plt.figure(figsize=(8, 5), dpi=150)
    for idx in chosen:
        y = spectra[idx]
        lbl = int(labels[idx])
        if class_map and lbl < len(class_map):
            n_val = class_map[lbl].get("n", None)
            d_val = class_map[lbl].get("d_nm", None)
            label = f"c{lbl} (n={n_val:.3f}, d={d_val:.1f}nm)"
        else:
            label = f"c{lbl}"
        plt.plot(wavelengths, y, alpha=0.7, linewidth=1.0, label=label)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Extinction (a.u.)")
    plt.title("Random Spectra Sanity Check")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
