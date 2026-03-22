import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command pretrain bundle: save pth/png/csv/json into a timestamped run folder"
    )
    parser.add_argument("--data-dir", type=str, default="data/pretrain_full")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-n", type=int, default=100)
    parser.add_argument("--unique-bases-per-n", type=int, default=12)
    parser.add_argument("--n-values", type=str, default="1.335,1.355,1.375")
    parser.add_argument("--gold-nk-csv", type=str, default="data/au_johnson_nk.csv")
    return parser.parse_args()


def run_cmd(cmd: list) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() if args.run_name.strip() else f"run_{stamp}"
    run_dir = (PROJECT_ROOT / args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = run_dir / "lspr_encoder_v1.pth"
    metrics_csv = run_dir / "train_metrics.csv"
    metrics_json = run_dir / "train_metrics.json"
    metrics_png = run_dir / "train_metrics.png"
    tsne_png = run_dir / "tsne_validation.png"

    train_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_pretrain.py"),
        "--data-dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--val-frac",
        str(args.val_frac),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--num-threads",
        str(args.num_threads),
        "--log-interval",
        str(args.log_interval),
        "--seed",
        str(args.seed),
        "--save-path",
        str(encoder_path),
        "--metrics-csv",
        str(metrics_csv),
        "--metrics-json",
        str(metrics_json),
        "--metrics-png",
        str(metrics_png),
    ]
    run_cmd(train_cmd)

    tsne_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tsne_validate.py"),
        "--encoder",
        str(encoder_path),
        "--out",
        str(tsne_png),
        "--samples-per-n",
        str(args.samples_per_n),
        "--unique-bases-per-n",
        str(args.unique_bases_per_n),
        "--n-values",
        args.n_values,
        "--gold-nk-csv",
        args.gold_nk_csv,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    run_cmd(tsne_cmd)

    manifest = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "files": {
            "encoder_pth": str(encoder_path),
            "train_metrics_csv": str(metrics_csv),
            "train_metrics_json": str(metrics_json),
            "train_metrics_png": str(metrics_png),
            "tsne_png": str(tsne_png),
        },
        "config": vars(args),
    }
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}", flush=True)
    print(f"Bundle done. Run folder: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
