import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_fewshot import (
    build_fewshot_split,
    choose_device,
    extract_embeddings,
    load_real_dataset,
    run_linear_head_mode,
    run_prototype_mode,
)
from lspr.model import ResNet1DEncoder
import torch


def find_default_encoder_path() -> str:
    stage2_root = PROJECT_ROOT / "outputs" / "stage2_domain_pretrain"
    if stage2_root.exists():
        candidates = sorted(stage2_root.glob("run_*/encoder_stage2_best.pth"))
        if candidates:
            return str(candidates[-1])
    return str(PROJECT_ROOT / "outputs" / "exp_20260324_real20260204_v1" / "pretrain" / "enc_s2026" / "lspr_encoder_v1.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-B few-shot performance over multiple episodes")
    parser.add_argument("--data-dir", type=str, default="data/real_fewshot")
    parser.add_argument("--encoder-path", type=str, default=find_default_encoder_path())
    parser.add_argument("--mode", type=str, default="prototype", choices=["prototype", "linear_head"])
    parser.add_argument("--n-way", type=int, default=0, help="0 means all classes")
    parser.add_argument("--k-shots", type=str, default="1,3,5", help="Comma-separated k values")
    parser.add_argument("--n-query", type=int, default=20, help="<=0 means use all remaining samples per class")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per k-shot")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40, help="Used in linear_head mode")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="")
    return parser.parse_args()


def parse_k_shots(raw: str) -> List[int]:
    out = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("k-shots is empty")
    if any(k <= 0 for k in out):
        raise ValueError("All k-shot values must be > 0")
    return out


def resolve_save_dir(save_dir_arg: str) -> Path:
    if save_dir_arg.strip():
        out = Path(save_dir_arg)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs") / f"fewshot_eval_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def aggregate(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    metrics = ["acc", "macro_f1", "macro_precision", "macro_recall", "query_loss", "top2_acc"]
    if any("mae" in r for r in rows):
        metrics.append("mae")
    by_k: Dict[int, List[Dict[str, float]]] = {}
    for r in rows:
        k = int(r["k_shot"])
        by_k.setdefault(k, []).append(r)

    out = []
    for k in sorted(by_k.keys()):
        grp = by_k[k]
        record: Dict[str, float] = {"k_shot": k, "episodes": len(grp)}
        for m in metrics:
            vals = np.asarray([float(x[m]) for x in grp], dtype=np.float64)
            record[f"{m}_mean"] = float(np.mean(vals))
            record[f"{m}_std"] = float(np.std(vals))
        out.append(record)
    return out


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, summary_rows: List[Dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    ks = [int(r["k_shot"]) for r in summary_rows]
    acc_m = [float(r["acc_mean"]) for r in summary_rows]
    acc_s = [float(r["acc_std"]) for r in summary_rows]
    f1_m = [float(r["macro_f1_mean"]) for r in summary_rows]
    f1_s = [float(r["macro_f1_std"]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.errorbar(ks, acc_m, yerr=acc_s, marker="o", capsize=4, label="accuracy")
    ax.errorbar(ks, f1_m, yerr=f1_s, marker="s", capsize=4, label="macro_f1")
    ax.set_xlabel("K-shot")
    ax.set_ylabel("Score")
    ax.set_title("Few-shot Episode Evaluation")
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_label_value(label: str) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+", str(label))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def compute_mae(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, id_to_label: Dict[int, str]) -> Optional[float]:
    if not id_to_label:
        return None
    ids = set(int(x) for x in np.unique(np.concatenate([y_true_raw, y_pred_raw])).tolist())
    id_to_value: Dict[int, float] = {}
    for cid in ids:
        label = id_to_label.get(int(cid), "")
        val = parse_label_value(label)
        if val is None:
            return None
        id_to_value[int(cid)] = float(val)
    true_vals = np.asarray([id_to_value[int(x)] for x in y_true_raw.tolist()], dtype=np.float64)
    pred_vals = np.asarray([id_to_value[int(x)] for x in y_pred_raw.tolist()], dtype=np.float64)
    return float(np.mean(np.abs(true_vals - pred_vals)))


def main() -> None:
    args = parse_args()
    k_shots = parse_k_shots(args.k_shots)
    device = choose_device(args.device)
    out_dir = resolve_save_dir(args.save_dir)

    data_dir = Path(args.data_dir)
    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    print("Loading data and encoder...", flush=True)
    spectra, labels, id_to_label = load_real_dataset(data_dir)
    encoder = ResNet1DEncoder(embedding_dim=128)
    encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu"))
    encoder.to(device)

    print("Extracting embeddings for all samples...", flush=True)
    all_emb = extract_embeddings(
        encoder=encoder,
        spectra=spectra,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    episode_rows: List[Dict] = []
    for k in k_shots:
        print(f"Evaluating k-shot={k} ...", flush=True)
        for ep in range(args.episodes):
            split_seed = args.seed + k * 1000 + ep
            classes, support_idx, query_idx = build_fewshot_split(
                labels=labels,
                n_way=args.n_way,
                k_shot=k,
                n_query=args.n_query,
                seed=split_seed,
            )
            support_emb = all_emb[support_idx]
            query_emb = all_emb[query_idx]
            support_y = labels[support_idx]
            query_y = labels[query_idx]

            if args.mode == "prototype":
                scores, pred_raw = run_prototype_mode(
                    support_emb=support_emb,
                    support_y_raw=support_y,
                    query_emb=query_emb,
                    query_y_raw=query_y,
                    classes=classes,
                    temperature=args.temperature,
                )
            else:
                linear_args = SimpleNamespace(lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs)
                scores, _, pred_raw, _ = run_linear_head_mode(
                    support_emb=support_emb,
                    support_y_raw=support_y,
                    query_emb=query_emb,
                    query_y_raw=query_y,
                    classes=classes,
                    args=linear_args,
                    device=device,
                )
            mae = compute_mae(query_y, pred_raw, id_to_label)
            if mae is not None:
                scores["mae"] = float(mae)

            row = {
                "k_shot": int(k),
                "episode": int(ep),
                "num_classes": int(len(classes)),
                "num_support": int(len(support_idx)),
                "num_query": int(len(query_idx)),
                "acc": float(scores["acc"]),
                "macro_f1": float(scores["macro_f1"]),
                "macro_precision": float(scores["macro_precision"]),
                "macro_recall": float(scores["macro_recall"]),
                "top2_acc": float(scores.get("top2_acc", 0.0)),
                "query_loss": float(scores["query_loss"]),
                "split_seed": int(split_seed),
            }
            if "mae" in scores:
                row["mae"] = float(scores["mae"])
            episode_rows.append(row)

            if ep == 0 or (ep + 1) % max(1, args.episodes // 2) == 0 or ep + 1 == args.episodes:
                print(
                    f"  episode {ep + 1:02d}/{args.episodes:02d} | "
                    f"acc={row['acc']:.3f} | f1={row['macro_f1']:.3f}",
                    flush=True,
                )

    summary_rows = aggregate(episode_rows)

    episode_csv = out_dir / "episode_metrics.csv"
    summary_csv = out_dir / "summary_metrics.csv"
    summary_json = out_dir / "summary_metrics.json"
    plot_png = out_dir / "fewshot_eval_curve.png"

    save_csv(episode_csv, episode_rows)
    save_csv(summary_csv, summary_rows)
    save_plot(plot_png, summary_rows)

    payload = {
        "config": vars(args),
        "data_dir": str(data_dir.resolve()),
        "encoder_path": str(encoder_path.resolve()),
        "device_used": str(device),
        "summary": summary_rows,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved evaluation outputs:", flush=True)
    print(f"  {episode_csv}", flush=True)
    print(f"  {summary_csv}", flush=True)
    print(f"  {summary_json}", flush=True)
    print(f"  {plot_png}", flush=True)


if __name__ == "__main__":
    main()
