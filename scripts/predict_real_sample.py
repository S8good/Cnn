import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_fewshot import (
    build_fewshot_split,
    choose_device,
    extract_embeddings,
    load_real_dataset,
)
from lspr.model import ResNet1DEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one real LSPR sample with Stage-B few-shot models")
    parser.add_argument("--data-dir", type=str, default="data/real_fewshot")
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="prototype", choices=["prototype", "linear_head"])
    parser.add_argument("--sample-index", type=int, default=-1, help="Index in prepared dataset spectra.npy")
    parser.add_argument("--spectrum-npy", type=str, default="", help="Optional path to 1D spectrum .npy")
    parser.add_argument("--n-way", type=int, default=0, help="0 means all classes")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapted-head", type=str, default="", help="Required in linear_head mode")
    parser.add_argument(
        "--fewshot-metrics-json",
        type=str,
        default="",
        help="Optional few-shot metrics json from train_fewshot.py to restore class subset",
    )
    parser.add_argument("--classes", type=str, default="", help="Optional class ids, e.g. 0,1,2")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def parse_classes(raw: str) -> Optional[np.ndarray]:
    if not raw.strip():
        return None
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        return None
    return np.asarray(sorted(set(vals)), dtype=np.int64)


def load_query_spectrum(args: argparse.Namespace, spectra: np.ndarray) -> tuple:
    if args.spectrum_npy.strip():
        arr = np.load(args.spectrum_npy).astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError("spectrum-npy must be 1D or shape [1, L]")
        return arr, None, "external_npy"

    idx = args.sample_index if args.sample_index >= 0 else 0
    if idx < 0 or idx >= len(spectra):
        raise ValueError(f"sample-index out of range: {idx}")
    return spectra[idx], int(idx), "dataset_index"


def compute_prototype_prediction(
    query_emb: np.ndarray,
    support_emb: np.ndarray,
    support_y_raw: np.ndarray,
    classes: np.ndarray,
    temperature: float,
) -> tuple:
    support_t = torch.from_numpy(support_emb).float()
    query_t = torch.from_numpy(query_emb).float().unsqueeze(0)

    class_to_pos = {int(c): i for i, c in enumerate(classes.tolist())}
    support_pos = torch.from_numpy(np.asarray([class_to_pos[int(y)] for y in support_y_raw], dtype=np.int64))

    protos = []
    for c_pos in range(len(classes)):
        m = support_pos == c_pos
        protos.append(support_t[m].mean(dim=0))
    protos = torch.stack(protos, dim=0)

    query_n = F.normalize(query_t, dim=1)
    proto_n = F.normalize(protos, dim=1)
    logits = (query_n @ proto_n.t()) / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_pos = int(np.argmax(probs))
    pred_raw = int(classes[pred_pos])
    return pred_raw, probs


def resolve_linear_classes(args: argparse.Namespace, labels: np.ndarray) -> np.ndarray:
    if args.fewshot_metrics_json.strip():
        with Path(args.fewshot_metrics_json).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        classes = payload.get("split", {}).get("classes")
        if classes is None:
            raise ValueError("fewshot-metrics-json has no split.classes")
        return np.asarray(classes, dtype=np.int64)

    manual = parse_classes(args.classes)
    if manual is not None:
        return manual

    if args.n_way > 0:
        rng = np.random.default_rng(args.seed)
        uniq = np.unique(labels)
        if args.n_way > len(uniq):
            raise ValueError(f"n-way={args.n_way} larger than available classes={len(uniq)}")
        return np.sort(rng.choice(uniq, size=args.n_way, replace=False)).astype(np.int64)

    return np.unique(labels).astype(np.int64)


def build_topk(probs: np.ndarray, classes: np.ndarray, id_to_label: Dict[int, str], topk: int) -> List[Dict]:
    order = np.argsort(-probs)[: max(1, topk)]
    out = []
    for pos in order:
        class_id = int(classes[int(pos)])
        out.append(
            {
                "class_id": class_id,
                "class_name": id_to_label.get(class_id, str(class_id)),
                "prob": float(probs[int(pos)]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    data_dir = Path(args.data_dir)
    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    spectra, labels, id_to_label = load_real_dataset(data_dir)
    query_spectrum, sample_idx, input_mode = load_query_spectrum(args, spectra)
    if query_spectrum.shape[0] != spectra.shape[1]:
        raise ValueError(
            f"Query length={query_spectrum.shape[0]} does not match dataset length={spectra.shape[1]}"
        )

    encoder = ResNet1DEncoder(embedding_dim=128)
    encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu"))
    encoder.to(device)

    query_emb = extract_embeddings(
        encoder=encoder,
        spectra=query_spectrum.reshape(1, -1).astype(np.float32),
        device=device,
        batch_size=1,
        num_workers=0,
    )[0]

    if args.mode == "prototype":
        classes, support_idx, _ = build_fewshot_split(
            labels=labels,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_query=1,
            seed=args.seed,
        )
        support_emb = extract_embeddings(
            encoder=encoder,
            spectra=spectra[support_idx],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        support_y = labels[support_idx]
        pred_class_id, probs = compute_prototype_prediction(
            query_emb=query_emb,
            support_emb=support_emb,
            support_y_raw=support_y,
            classes=classes,
            temperature=args.temperature,
        )
    else:
        if not args.adapted_head.strip():
            raise ValueError("--adapted-head is required when mode=linear_head")
        classes = resolve_linear_classes(args, labels)
        head = nn.Linear(128, len(classes))
        head.load_state_dict(torch.load(args.adapted_head, map_location="cpu"))
        head.to(device)
        head.eval()
        with torch.no_grad():
            x = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)
            logits = head(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_pos = int(np.argmax(probs))
        pred_class_id = int(classes[pred_pos])

    topk_list = build_topk(probs, classes, id_to_label, args.topk)

    result = {
        "mode": args.mode,
        "device_used": str(device),
        "input_mode": input_mode,
        "sample_index": sample_idx,
        "pred_class_id": pred_class_id,
        "pred_class_name": id_to_label.get(pred_class_id, str(pred_class_id)),
        "topk": topk_list,
    }

    if sample_idx is not None:
        true_id = int(labels[sample_idx])
        result["true_class_id"] = true_id
        result["true_class_name"] = id_to_label.get(true_id, str(true_id))
        result["correct"] = bool(true_id == pred_class_id)

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    if args.save_json.strip():
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {save_path}", flush=True)


if __name__ == "__main__":
    main()
