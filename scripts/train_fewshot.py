import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lspr.model import ResNet1DEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-B few-shot adaptation on real LSPR data")
    parser.add_argument("--data-dir", type=str, default="data/real_fewshot")
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="prototype",
        choices=["prototype", "linear_head"],
        help="prototype: no training, linear_head: train a small head on support set",
    )
    parser.add_argument("--n-way", type=int, default=0, help="0 means all classes")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=20, help="<=0 means use all remaining samples per class")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80, help="Used in linear_head mode")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_save_dir(save_dir_arg: str) -> Path:
    if save_dir_arg.strip():
        out = Path(save_dir_arg)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs") / f"fewshot_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_real_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    spectra_path = data_dir / "spectra.npy"
    labels_path = data_dir / "labels.npy"
    label_map_path = data_dir / "label_map.json"
    if not spectra_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing spectra.npy or labels.npy. Run scripts/prepare_real_dataset.py first."
        )
    spectra = np.load(spectra_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.int64)
    if spectra.shape[0] != labels.shape[0]:
        raise ValueError("spectra and labels length mismatch")

    id_to_label: Dict[int, str] = {}
    if label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_map = payload.get("id_to_label", {})
        id_to_label = {int(k): str(v) for k, v in raw_map.items()}
    else:
        for c in sorted(np.unique(labels).tolist()):
            id_to_label[int(c)] = f"class_{c}"

    return spectra, labels, id_to_label


def build_fewshot_split(
    labels: np.ndarray,
    n_way: int,
    k_shot: int,
    n_query: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    if n_way > 0:
        if n_way > len(classes):
            raise ValueError(f"n_way={n_way} is larger than available classes={len(classes)}")
        classes = np.sort(rng.choice(classes, size=n_way, replace=False))

    support_idx = []
    query_idx = []
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        if len(idx) <= k_shot:
            raise ValueError(
                f"Class {c} has {len(idx)} samples, not enough for k_shot={k_shot} + query"
            )
        sup = idx[:k_shot]
        rem = idx[k_shot:]
        if n_query <= 0:
            qry = rem
        else:
            qry = rem[: min(n_query, len(rem))]
        if len(qry) == 0:
            raise ValueError(f"Class {c} has no query samples after split")
        support_idx.extend(sup.tolist())
        query_idx.extend(qry.tolist())

    return classes.astype(np.int64), np.asarray(support_idx, dtype=np.int64), np.asarray(query_idx, dtype=np.int64)


def extract_embeddings(
    encoder: ResNet1DEncoder,
    spectra: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(spectra).unsqueeze(1))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    embs = []
    encoder.eval()
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            e = encoder(x).cpu().numpy()
            embs.append(e)
    return np.concatenate(embs, axis=0)


def map_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_pos = {int(c): i for i, c in enumerate(classes.tolist())}
    return np.asarray([class_to_pos[int(y)] for y in labels.tolist()], dtype=np.int64)


def calc_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_prototype_mode(
    support_emb: np.ndarray,
    support_y_raw: np.ndarray,
    query_emb: np.ndarray,
    query_y_raw: np.ndarray,
    classes: np.ndarray,
    temperature: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    support_emb_t = torch.from_numpy(support_emb).float()
    query_emb_t = torch.from_numpy(query_emb).float()
    support_y = torch.from_numpy(map_labels(support_y_raw, classes)).long()
    query_y = torch.from_numpy(map_labels(query_y_raw, classes)).long()

    protos = []
    for c_pos in range(len(classes)):
        m = support_y == c_pos
        protos.append(support_emb_t[m].mean(dim=0))
    protos = torch.stack(protos, dim=0)

    query_n = F.normalize(query_emb_t, dim=1)
    proto_n = F.normalize(protos, dim=1)
    logits = (query_n @ proto_n.t()) / max(temperature, 1e-6)
    loss = F.cross_entropy(logits, query_y).item()
    pred_pos = torch.argmax(logits, dim=1).numpy()
    pred_raw = classes[pred_pos]

    scores = calc_scores(query_y_raw, pred_raw)
    scores["query_loss"] = float(loss)
    return scores, pred_raw


def run_linear_head_mode(
    support_emb: np.ndarray,
    support_y_raw: np.ndarray,
    query_emb: np.ndarray,
    query_y_raw: np.ndarray,
    classes: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray, nn.Linear]:
    x_sup = torch.from_numpy(support_emb).float().to(device)
    y_sup = torch.from_numpy(map_labels(support_y_raw, classes)).long().to(device)
    x_q = torch.from_numpy(query_emb).float().to(device)
    y_q = torch.from_numpy(map_labels(query_y_raw, classes)).long().to(device)

    head = nn.Linear(support_emb.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        head.train()
        logits_sup = head(x_sup)
        train_loss = F.cross_entropy(logits_sup, y_sup)
        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            logits_q = head(x_q)
            val_loss = F.cross_entropy(logits_q, y_q).item()
            pred = torch.argmax(logits_q, dim=1)
            val_acc = float((pred == y_q).float().mean().item())
            train_acc = float((torch.argmax(logits_sup, dim=1) == y_sup).float().mean().item())

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "train_acc": train_acc,
                "val_loss": float(val_loss),
                "val_acc": val_acc,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss.item():.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}",
                flush=True,
            )

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        logits_q = head(x_q)
        q_loss = float(F.cross_entropy(logits_q, y_q).item())
        pred_pos = torch.argmax(logits_q, dim=1).cpu().numpy()
    pred_raw = classes[pred_pos]
    scores = calc_scores(query_y_raw, pred_raw)
    scores["query_loss"] = q_loss
    return scores, history, pred_raw, head


def save_confusion_png(path: Path, y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, id_to_label: Dict[int, str]) -> None:
    import matplotlib.pyplot as plt

    labels = classes.tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick = [id_to_label.get(int(c), str(int(c))) for c in labels]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(tick)))
    ax.set_yticks(np.arange(len(tick)))
    ax.set_xticklabels(tick, rotation=45, ha="right")
    ax.set_yticklabels(tick)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Few-shot Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_history_csv(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    out_dir = resolve_save_dir(args.save_dir)

    data_dir = Path(args.data_dir)
    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    print("Loading real dataset...", flush=True)
    spectra, labels, id_to_label = load_real_dataset(data_dir)
    classes, support_idx, query_idx = build_fewshot_split(
        labels=labels,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        seed=args.seed,
    )
    print(
        f"Few-shot split ready | classes={len(classes)} | support={len(support_idx)} | query={len(query_idx)}",
        flush=True,
    )

    print("Loading encoder and extracting embeddings...", flush=True)
    encoder = ResNet1DEncoder(embedding_dim=128)
    encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu"))
    encoder.to(device)

    support_emb = extract_embeddings(
        encoder,
        spectra[support_idx],
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    query_emb = extract_embeddings(
        encoder,
        spectra[query_idx],
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    support_y_raw = labels[support_idx]
    query_y_raw = labels[query_idx]

    history: List[Dict[str, float]] = []
    if args.mode == "prototype":
        print("Running prototype inference...", flush=True)
        scores, query_pred_raw = run_prototype_mode(
            support_emb=support_emb,
            support_y_raw=support_y_raw,
            query_emb=query_emb,
            query_y_raw=query_y_raw,
            classes=classes,
            temperature=args.temperature,
        )
        print(f"Prototype query acc={scores['acc']:.4f}, macro_f1={scores['macro_f1']:.4f}", flush=True)
        np.save(out_dir / "prototypes_classes.npy", classes.astype(np.int64))
    else:
        print("Training linear head on support embeddings...", flush=True)
        scores, history, query_pred_raw, head = run_linear_head_mode(
            support_emb=support_emb,
            support_y_raw=support_y_raw,
            query_emb=query_emb,
            query_y_raw=query_y_raw,
            classes=classes,
            args=args,
            device=device,
        )
        torch.save(head.state_dict(), out_dir / "adapted_head.pth")
        print(f"Linear-head query acc={scores['acc']:.4f}, macro_f1={scores['macro_f1']:.4f}", flush=True)

    conf_path = out_dir / "confusion_matrix.png"
    save_confusion_png(conf_path, query_y_raw, query_pred_raw, classes, id_to_label)

    metrics_payload = {
        "config": vars(args),
        "data_dir": str(data_dir.resolve()),
        "encoder_path": str(encoder_path.resolve()),
        "save_dir": str(out_dir.resolve()),
        "device_used": str(device),
        "split": {
            "classes": classes.astype(int).tolist(),
            "num_support": int(len(support_idx)),
            "num_query": int(len(query_idx)),
            "support_indices": support_idx.astype(int).tolist(),
            "query_indices": query_idx.astype(int).tolist(),
        },
        "scores": scores,
        "history": history,
    }

    metrics_json = out_dir / "fewshot_metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    if history:
        save_history_csv(out_dir / "fewshot_metrics.csv", history)

    print("Saved outputs:", flush=True)
    print(f"  {metrics_json}", flush=True)
    if history:
        print(f"  {out_dir / 'fewshot_metrics.csv'}", flush=True)
    print(f"  {conf_path}", flush=True)


if __name__ == "__main__":
    main()
