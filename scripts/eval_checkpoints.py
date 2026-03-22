import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from lspr.data import PretrainSpectraDataset, make_subset, split_indices
from lspr.model import ResNet1DClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate encoder checkpoints on their datasets")
    parser.add_argument(
        "--items",
        type=str,
        default=(
            "small:data/pretrain_small:outputs/lspr_encoder_v1_small.pth,"
            "mid:data/pretrain_mid:outputs/lspr_encoder_v1_mid.pth,"
            "full:data/pretrain_full:outputs/lspr_encoder_v1_full.pth"
        ),
        help="Comma separated entries: name:data_dir:encoder_path",
    )
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def parse_items(raw: str):
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        segs = part.split(":")
        if len(segs) != 3:
            raise ValueError(f"Bad item: {part}. Expected name:data_dir:encoder_path")
        items.append((segs[0], Path(segs[1]), Path(segs[2])))
    return items


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_split(model, subset, device, batch_size: int):
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    return total_loss / len(subset), correct / len(subset)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    items = parse_items(args.items)

    print(f"Device: {device}")
    for name, data_dir, encoder_path in items:
        spectra_path = data_dir / "pretrain_spectra.npy"
        labels_path = data_dir / "pretrain_labels.npy"
        if not spectra_path.exists() or not labels_path.exists() or not encoder_path.exists():
            print(f"{name}: missing files")
            continue

        dataset = PretrainSpectraDataset(str(spectra_path), str(labels_path))
        train_idx, val_idx = split_indices(len(dataset), val_frac=args.val_frac, seed=args.seed)
        train_set = make_subset(dataset, train_idx)
        val_set = make_subset(dataset, val_idx)

        num_classes = int(dataset.labels.max()) + 1
        model = ResNet1DClassifier(num_classes=num_classes, embedding_dim=128)
        model.encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu"))
        model.to(device)

        train_loss, train_acc = eval_split(model, train_set, device, args.batch_size)
        val_loss, val_acc = eval_split(model, val_set, device, args.batch_size)

        print(
            f"{name}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
