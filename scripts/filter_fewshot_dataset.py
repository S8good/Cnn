import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter few-shot dataset by minimum samples per class and remap labels."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input prepared dataset directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Output filtered dataset directory")
    parser.add_argument("--min-samples", type=int, default=0, help="Min sample count per class")
    parser.add_argument("--k-shot", type=int, default=0, help="If >0, used with n-query to derive min-samples")
    parser.add_argument("--n-query", type=int, default=0, help="If >0, used with k-shot to derive min-samples")
    parser.add_argument(
        "--keep-classes",
        type=str,
        default="",
        help="Optional raw class id list, e.g. 0,1,2",
    )
    parser.add_argument("--copy-wavelengths", action="store_true", help="Copy wavelengths.npy when available")
    return parser.parse_args()


def parse_keep_classes(raw: str) -> Optional[List[int]]:
    if not raw.strip():
        return None
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return sorted(set(vals))


def class_counts(labels: np.ndarray) -> Dict[int, int]:
    uniq, cnt = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}


def load_label_names(label_map_path: Path) -> Dict[int, str]:
    if not label_map_path.exists():
        return {}
    with label_map_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("id_to_label", {})
    return {int(k): str(v) for k, v in raw.items()}


def remap_labels(labels_raw: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    uniq = sorted(np.unique(labels_raw).tolist())
    mapping = {int(old): int(new) for new, old in enumerate(uniq)}
    labels_new = np.asarray([mapping[int(y)] for y in labels_raw.tolist()], dtype=np.int64)
    return labels_new, mapping


def filter_metadata(metadata_path: Path, keep_indices: np.ndarray, labels_new: np.ndarray, out_path: Path) -> None:
    if not metadata_path.exists():
        return
    keep_set = set(int(i) for i in keep_indices.tolist())
    label_new_map = {int(i): int(labels_new[pos]) for pos, i in enumerate(keep_indices.tolist())}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        return
    if "label_id" not in fields:
        fields.append("label_id")

    out_rows = []
    for idx, row in enumerate(rows):
        if idx not in keep_set:
            continue
        row["label_id"] = str(label_new_map[idx])
        out_rows.append(row)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spectra_path = in_dir / "spectra.npy"
    labels_path = in_dir / "labels.npy"
    wavelengths_path = in_dir / "wavelengths.npy"
    metadata_path = in_dir / "metadata.csv"
    label_map_path = in_dir / "label_map.json"

    if not spectra_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Input dir must include spectra.npy and labels.npy")

    spectra = np.load(spectra_path)
    labels = np.load(labels_path).astype(np.int64)
    if spectra.shape[0] != labels.shape[0]:
        raise ValueError("spectra and labels length mismatch")

    before_counts = class_counts(labels)
    label_names = load_label_names(label_map_path)
    keep_classes = parse_keep_classes(args.keep_classes)

    required_min = args.min_samples
    if args.k_shot > 0 and args.n_query > 0:
        required_min = max(required_min, args.k_shot + args.n_query)
    required_min = max(required_min, 1)

    keep_class_set = set()
    for cls, cnt in before_counts.items():
        if cnt >= required_min:
            keep_class_set.add(int(cls))
    if keep_classes is not None:
        keep_class_set = keep_class_set.intersection(set(keep_classes))
    if not keep_class_set:
        raise ValueError("No class remains after filtering; please lower min-samples or adjust k-shot/n-query")

    keep_mask = np.asarray([int(y) in keep_class_set for y in labels.tolist()], dtype=bool)
    keep_idx = np.where(keep_mask)[0]
    spectra_f = spectra[keep_idx].astype(np.float32)
    labels_raw_f = labels[keep_idx].astype(np.int64)

    labels_new, old_to_new = remap_labels(labels_raw_f)
    new_to_old = {int(v): int(k) for k, v in old_to_new.items()}

    np.save(out_dir / "spectra.npy", spectra_f)
    np.save(out_dir / "labels.npy", labels_new)
    if args.copy_wavelengths and wavelengths_path.exists():
        np.save(out_dir / "wavelengths.npy", np.load(wavelengths_path).astype(np.float32))

    filter_metadata(metadata_path, keep_idx, labels_new, out_dir / "metadata.csv")

    id_to_label_new = {}
    label_to_id_new = {}
    for new_id, old_id in sorted(new_to_old.items()):
        name = label_names.get(old_id, f"class_{old_id}")
        id_to_label_new[int(new_id)] = name
        label_to_id_new[name] = int(new_id)

    report = {
        "input_dir": str(in_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "required_min_samples": int(required_min),
        "k_shot": int(args.k_shot),
        "n_query": int(args.n_query),
        "num_samples_before": int(len(labels)),
        "num_samples_after": int(len(labels_new)),
        "class_counts_before": before_counts,
        "class_counts_after_raw": class_counts(labels_raw_f),
        "class_mapping_old_to_new": {int(k): int(v) for k, v in old_to_new.items()},
        "class_mapping_new_to_old": {int(k): int(v) for k, v in new_to_old.items()},
        "label_to_id": label_to_id_new,
        "id_to_label": id_to_label_new,
    }
    with (out_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Filtered few-shot dataset:")
    print(f"  out_dir: {out_dir}")
    print(f"  samples: {len(labels_new)} / {len(labels)}")
    print(f"  classes: {len(np.unique(labels_new))}")
    print(f"  min-samples: {required_min}")


if __name__ == "__main__":
    main()
