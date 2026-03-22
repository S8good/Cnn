import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain 1D-ResNet on LSPR spectra")
    parser.add_argument("--data-dir", type=str, default="data/pretrain")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-threads", type=int, default=0, help="0 keeps PyTorch default")
    parser.add_argument("--log-interval", type=int, default=10, help="Batches between progress logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="outputs/lspr_encoder_v1.pth")
    parser.add_argument("--metrics-csv", type=str, default="")
    parser.add_argument("--metrics-json", type=str, default="")
    parser.add_argument("--metrics-png", type=str, default="")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, criterion, log_interval: int):
    model.train()
    running_loss = 0.0
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        if log_interval > 0 and (step % log_interval == 0 or step == 1):
            avg = running_loss / (step * x.size(0))
            print(f"  step {step:04d} | running_loss={avg:.4f}")
    return running_loss / len(loader.dataset)


def eval_one_epoch(model, loader, device, criterion):
    import torch
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    loss = running_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return loss, acc


def main() -> None:
    args = parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.num_threads)

    print("Starting training script...", flush=True)
    print("Importing torch...", flush=True)

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from lspr.data import PretrainSpectraDataset, make_subset, split_indices
    from lspr.model import ResNet1DClassifier

    torch.manual_seed(args.seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    data_dir = Path(args.data_dir)
    spectra_path = data_dir / "pretrain_spectra.npy"
    labels_path = data_dir / "pretrain_labels.npy"

    print("Loading dataset...", flush=True)
    dataset = PretrainSpectraDataset(str(spectra_path), str(labels_path))
    train_idx, val_idx = split_indices(len(dataset), val_frac=args.val_frac, seed=args.seed)
    train_set = make_subset(dataset, train_idx)
    val_set = make_subset(dataset, val_idx)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_memory = device.type == "cuda"
    print("Building dataloaders...", flush=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    num_classes = int(dataset.labels.max()) + 1
    model = ResNet1DClassifier(num_classes=num_classes, embedding_dim=128)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []

    print(
        f"Device: {device} | train={len(train_set)} val={len(val_set)} "
        f"| batch={args.batch_size} | workers={args.num_workers}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, args.log_interval)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device, criterion)
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        epoch_sec = float(time.time() - t0)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": lr,
                "epoch_sec": epoch_sec,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | lr={lr:.3e}",
            flush=True,
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), str(save_path))
    print(f"Saved encoder to: {save_path}", flush=True)

    default_base = save_path.with_suffix("")
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else default_base.with_name(default_base.name + "_metrics.csv")
    metrics_json = Path(args.metrics_json) if args.metrics_json else default_base.with_name(default_base.name + "_metrics.json")
    metrics_png = Path(args.metrics_png) if args.metrics_png else default_base.with_name(default_base.name + "_metrics.png")
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_png.parent.mkdir(parents=True, exist_ok=True)

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "lr", "epoch_sec"])
        writer.writeheader()
        writer.writerows(history)

    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
            "device": args.device,
            "seed": args.seed,
        },
        "history": history,
        "best": min(history, key=lambda x: x["val_loss"]) if history else None,
    }
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save a quick training curve PNG for experiment tracking.
    try:
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        tr = [h["train_loss"] for h in history]
        va = [h["val_loss"] for h in history]
        plt.figure(figsize=(7, 4), dpi=140)
        plt.plot(epochs, tr, label="train_loss", linewidth=2)
        plt.plot(epochs, va, label="val_loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Pretrain Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(metrics_png)
        print(f"Saved metrics PNG to: {metrics_png}", flush=True)
    except Exception as exc:
        print(f"WARNING: failed to save metrics PNG: {exc}", flush=True)

    print(f"Saved metrics CSV to: {metrics_csv}", flush=True)
    print(f"Saved metrics JSON to: {metrics_json}", flush=True)


if __name__ == "__main__":
    main()
