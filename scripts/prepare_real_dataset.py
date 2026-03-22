import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real LSPR dataset for Stage-B few-shot adaptation")
    parser.add_argument("--input-csv", type=str, required=True, help="Input CSV file (wide format)")
    parser.add_argument("--out-dir", type=str, default="data/real_fewshot", help="Output directory")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--sample-id-col", type=str, default="sample_id")
    parser.add_argument("--concentration-col", type=str, default="concentration")
    parser.add_argument("--source-col", type=str, default="source")
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument(
        "--spectrum-cols",
        type=str,
        default="",
        help="Comma-separated spectrum columns. If empty, auto detect by numeric names or prefixes.",
    )
    parser.add_argument(
        "--spectrum-prefixes",
        type=str,
        default="wl_,lambda_,nm_",
        help="Prefixes used by auto detection, comma separated.",
    )
    parser.add_argument("--source-wavelength-start", type=float, default=None)
    parser.add_argument("--source-wavelength-step", type=float, default=None)
    parser.add_argument("--target-start", type=float, default=400.0)
    parser.add_argument("--target-stop", type=float, default=800.0)
    parser.add_argument("--target-points", type=int, default=400)
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument(
        "--normalize",
        type=str,
        default="minmax",
        choices=["none", "minmax", "zscore"],
        help="Per-spectrum normalization mode",
    )
    parser.add_argument("--drop-missing", action="store_true", help="Drop rows with non-numeric spectrum values")
    return parser.parse_args()


def try_parse_wavelength(col: str, prefixes: List[str]) -> Optional[float]:
    raw = col.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    low = raw.lower()
    for p in prefixes:
        pp = p.strip().lower()
        if pp and low.startswith(pp):
            tail = low[len(pp) :]
            try:
                return float(tail)
            except ValueError:
                return None
    return None


def detect_spectrum_columns(
    fieldnames: List[str],
    spectrum_cols_arg: str,
    spectrum_prefixes_arg: str,
) -> Tuple[List[str], Optional[np.ndarray]]:
    if spectrum_cols_arg.strip():
        cols = [x.strip() for x in spectrum_cols_arg.split(",") if x.strip()]
        return cols, None

    prefixes = [x.strip() for x in spectrum_prefixes_arg.split(",") if x.strip()]
    pairs = []
    for c in fieldnames:
        wl = try_parse_wavelength(c, prefixes)
        if wl is not None:
            pairs.append((wl, c))
    if not pairs:
        raise ValueError(
            "No spectrum columns detected. Provide --spectrum-cols explicitly or set --spectrum-prefixes."
        )
    pairs.sort(key=lambda x: x[0])
    wavelengths = np.asarray([p[0] for p in pairs], dtype=np.float64)
    cols = [p[1] for p in pairs]
    return cols, wavelengths


def build_source_wavelengths(
    num_cols: int,
    detected_wl: Optional[np.ndarray],
    source_start: Optional[float],
    source_step: Optional[float],
) -> np.ndarray:
    if detected_wl is not None:
        return detected_wl
    if source_start is None or source_step is None:
        raise ValueError(
            "Cannot infer source wavelengths from column names. "
            "Please set --source-wavelength-start and --source-wavelength-step."
        )
    return source_start + source_step * np.arange(num_cols, dtype=np.float64)


def normalize_spectrum(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "minmax":
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        span = x_max - x_min
        if span <= 1e-12:
            return np.zeros_like(x)
        return (x - x_min) / span
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std <= 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std


def to_label_id(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    uniq = sorted(set(labels))
    label_to_id = {name: i for i, name in enumerate(uniq)}
    ids = np.asarray([label_to_id[x] for x in labels], dtype=np.int64)
    return ids, label_to_id


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    with in_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        fieldnames = list(reader.fieldnames)
        spectrum_cols, detected_wl = detect_spectrum_columns(
            fieldnames,
            args.spectrum_cols,
            args.spectrum_prefixes,
        )
        source_wl = build_source_wavelengths(
            len(spectrum_cols),
            detected_wl,
            args.source_wavelength_start,
            args.source_wavelength_step,
        )

        rows = []
        bad_rows = 0
        for row_idx, row in enumerate(reader):
            label = str(row.get(args.label_col, "")).strip()
            if not label:
                bad_rows += 1
                continue

            values = []
            failed = False
            for c in spectrum_cols:
                raw = str(row.get(c, "")).strip()
                try:
                    values.append(float(raw))
                except ValueError:
                    failed = True
                    break
            if failed:
                if args.drop_missing:
                    bad_rows += 1
                    continue
                raise ValueError(f"Non-numeric spectrum value at row {row_idx + 2}")

            rows.append(
                {
                    "sample_id": str(row.get(args.sample_id_col, f"s{row_idx:06d}")).strip(),
                    "label": label,
                    "concentration": str(row.get(args.concentration_col, "")).strip(),
                    "source": str(row.get(args.source_col, "")).strip(),
                    "split": str(row.get(args.split_col, "")).strip(),
                    "spectrum": np.asarray(values, dtype=np.float64),
                }
            )

    if not rows:
        raise ValueError("No valid rows found in CSV")

    if args.no_resample:
        target_wl = source_wl.astype(np.float64)
    else:
        target_wl = np.linspace(args.target_start, args.target_stop, args.target_points, dtype=np.float64)

    spectra = []
    labels = []
    sample_ids = []
    concentrations = []
    sources = []
    splits = []

    for r in rows:
        x = r["spectrum"]
        if args.no_resample:
            y = x
        else:
            y = np.interp(target_wl, source_wl, x)
        y = normalize_spectrum(y, args.normalize).astype(np.float32)
        spectra.append(y)
        labels.append(r["label"])
        sample_ids.append(r["sample_id"])
        concentrations.append(r["concentration"])
        sources.append(r["source"])
        splits.append(r["split"])

    spectra_arr = np.asarray(spectra, dtype=np.float32)
    label_ids, label_to_id = to_label_id(labels)
    id_to_label = {int(v): k for k, v in label_to_id.items()}

    np.save(out_dir / "spectra.npy", spectra_arr)
    np.save(out_dir / "labels.npy", label_ids)
    np.save(out_dir / "wavelengths.npy", target_wl.astype(np.float32))

    with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "label_id", "concentration", "source", "split"])
        for i in range(len(sample_ids)):
            writer.writerow(
                [
                    sample_ids[i],
                    labels[i],
                    int(label_ids[i]),
                    concentrations[i],
                    sources[i],
                    splits[i],
                ]
            )

    payload = {
        "input_csv": str(in_csv.resolve()),
        "num_samples": int(len(labels)),
        "num_points": int(spectra_arr.shape[1]),
        "num_classes": int(len(label_to_id)),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "spectrum_columns": spectrum_cols,
        "source_wavelength_min": float(np.min(source_wl)),
        "source_wavelength_max": float(np.max(source_wl)),
        "target_wavelength_min": float(np.min(target_wl)),
        "target_wavelength_max": float(np.max(target_wl)),
        "target_points": int(len(target_wl)),
        "normalize": args.normalize,
        "dropped_rows": int(bad_rows),
    }
    with (out_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Prepared real dataset:")
    print(f"  out_dir: {out_dir}")
    print(f"  samples: {len(labels)}")
    print(f"  classes: {len(label_to_id)}")
    print(f"  points : {spectra_arr.shape[1]}")


if __name__ == "__main__":
    main()
